"""Invocation-time renderer configuration and coordinate presets."""

from .sensor import Sensor, Perspective
from .film import Film
from .sampler import Sampler, Independent
from .emitter import Emitter, Envmap
from .integrator import Integrator, Path
from .render_pass import get_render_pass

import numpy as np
from dataclasses import dataclass, field
from typing import Literal


@dataclass(kw_only=True, slots=True)
class Config:
    """Invocation-time renderer policy, separate from visualization intent.

    A :class:`~hakowan.grammar.figure.Figure` stores portable camera, lighting,
    environment, and output intent. Passing an explicit ``Config`` to rendering
    selects operational backend settings and overrides the complete
    Figure-derived configuration rather than merging with it.

    Attributes:
        sensor: Sensor settings.
        film: Film settings.
        sampler: Sampler settings.
        emitters: Emitter settings.
        integrator: Integrator settings.
        render_passes: Set of active render passes.  Recognised values:

            - ``"albedo"``    – diffuse color without shading.
            - ``"depth"``     – depth buffer.
            - ``"normal"``    – shading-normal pass.
            - ``"facet_id"``  – per-face index encoded as RGB (Blender only).

            The convenience properties :attr:`albedo`, :attr:`depth`,
            :attr:`normal`, and :attr:`facet_id` are thin aliases that add or
            remove the corresponding string from this set.  Assigning an
            unrecognised pass name raises :class:`ValueError` (validated against
            :data:`hakowan.setup.render_pass.RENDER_PASSES`).

            Backend support varies: requesting a pass the chosen backend cannot
            honor logs a warning and is otherwise ignored.  Each honored pass is
            written to a ``<stem>_<pass><ext>`` sidecar file (or exposed as a
            live viewer toggle for WebGL); see
            :class:`hakowan.render.RenderResult` for the per-render manifest.
        environment_visible: Show an environment map to the camera when true.
        background: Optional WebGL/raster light or dark background override.

    """

    sensor: Sensor = field(default_factory=Perspective)
    film: Film = field(default_factory=Film)
    sampler: Sampler = field(default_factory=Independent)
    emitters: list[Emitter] = field(default_factory=lambda: [Envmap()])
    integrator: Integrator = field(default_factory=Path)
    _render_passes: set[str] = field(default_factory=set)
    environment_visible: bool = False
    background: Literal["light", "dark"] | None = None

    def __setattr__(self, name, value):
        if name == "background" and value not in {None, "light", "dark"}:
            raise ValueError("Config.background must be 'light', 'dark', or None.")
        object.__setattr__(self, name, value)

    def z_up(self) -> None:
        """Update configuration for z-up coordinate system."""
        self.sensor.location = np.array([0, -5, 0])
        self.sensor.up = np.array([0, 0, 1])
        for emitter in self.emitters:
            if isinstance(emitter, Envmap):
                emitter.up = np.array([0, 0, 1])
                emitter.rotation = 180.0

    def z_down(self) -> None:
        """Update configuration for z-down coordinate system."""
        self.sensor.location = np.array([0, 5, 0])
        self.sensor.up = np.array([0, 0, -1])
        for emitter in self.emitters:
            if isinstance(emitter, Envmap):
                emitter.up = np.array([0, 0, -1])
                emitter.rotation = 180.0

    def y_up(self) -> None:
        """Update configuration for y-up coordinate system."""
        self.sensor.location = np.array([0, 0, 5])
        self.sensor.up = np.array([0, 1, 0])
        for emitter in self.emitters:
            if isinstance(emitter, Envmap):
                emitter.up = np.array([0, 1, 0])
                emitter.rotation = 180.0

    def y_down(self) -> None:
        """Update configuration for y-down coordinate system."""
        self.sensor.location = np.array([0, 0, -5])
        self.sensor.up = np.array([0, -1, 0])
        for emitter in self.emitters:
            if isinstance(emitter, Envmap):
                emitter.up = np.array([0, -1, 0])
                emitter.rotation = 180.0

    # ------------------------------------------------------------------ #
    # render_passes – primary interface                                    #
    # ------------------------------------------------------------------ #

    @property
    def render_passes(self) -> frozenset[str]:
        """Immutable set of requested semantic render passes.

        Valid pass names are ``"albedo"``, ``"depth"``, ``"normal"``, and
        ``"facet_id"``. Assign a collection to replace the requests. Backends
        derive their native pass configuration at render time, keeping this set
        as the single source of truth.

        Example::

            config.render_passes = {"albedo", "depth"}
        """
        return frozenset(self._render_passes)

    @render_passes.setter
    def render_passes(self, value: set[str] | list[str] | frozenset[str]) -> None:
        """Replace the requested render passes after validating every name."""
        names = set(value)
        for name in names:
            get_render_pass(name)  # type: ignore[arg-type]
        self._render_passes = names

    # ------------------------------------------------------------------ #
    # Convenience boolean aliases                                          #
    # ------------------------------------------------------------------ #

    @property
    def albedo(self) -> bool:
        """Whether the albedo pass is active.  Alias for ``"albedo" in render_passes``."""
        return "albedo" in self._render_passes

    @albedo.setter
    def albedo(self, value: bool) -> None:
        """Add or remove the albedo pass request."""
        if value:
            self._render_passes.add("albedo")
        else:
            self._render_passes.discard("albedo")

    @property
    def depth(self) -> bool:
        """Whether the depth pass is active.  Alias for ``"depth" in render_passes``."""
        return "depth" in self._render_passes

    @depth.setter
    def depth(self, value: bool) -> None:
        """Add or remove the depth pass request."""
        if value:
            self._render_passes.add("depth")
        else:
            self._render_passes.discard("depth")

    @property
    def normal(self) -> bool:
        """Whether the shading-normal pass is active.  Alias for ``"normal" in render_passes``."""
        return "normal" in self._render_passes

    @normal.setter
    def normal(self, value: bool) -> None:
        """Add or remove the normal pass request."""
        if value:
            self._render_passes.add("normal")
        else:
            self._render_passes.discard("normal")

    @property
    def facet_id(self) -> bool:
        """Whether the facet-ID pass is active.  Alias for ``"facet_id" in render_passes``.

        When active the Blender backend performs a second render after the
        main one.  Every mesh face is colored with the RGB encoding of its
        zero-based index (R = high byte, G = mid byte, B = low byte) using a
        flat Emission shader so lighting has no effect.  The output is written
        to ``<stem>_facet_id<ext>`` with gamma correction, temporal blending,
        pixel filtering, and dithering all disabled so pixel values can be
        decoded directly::

            fid = (R << 16) | (G << 8) | B

        Background pixels have ``A = 0`` and can be masked out.  Supports up
        to 2**24 − 1 ≈ 16.7 M faces.
        """
        return "facet_id" in self._render_passes

    @facet_id.setter
    def facet_id(self, value: bool) -> None:
        """Add or remove the facet-ID pass."""
        if value:
            self._render_passes.add("facet_id")
        else:
            self._render_passes.discard("facet_id")
