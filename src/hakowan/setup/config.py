from .sensor import Sensor, Perspective
from .film import Film
from .sampler import Sampler, Independent
from .emitter import Emitter, Envmap
from .integrator import Integrator, Path, AOV
from ..common import logger

import numpy as np
from dataclasses import dataclass, field


@dataclass(kw_only=True, slots=True)
class Config:
    """Configuration for rendering.

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
            remove the corresponding string from this set.
    """

    sensor: Sensor = field(default_factory=Perspective)
    film: Film = field(default_factory=Film)
    sampler: Sampler = field(default_factory=Independent)
    emitters: list[Emitter] = field(default_factory=lambda: [Envmap()])
    integrator: Integrator = field(default_factory=Path)
    _render_passes: set = field(default_factory=set)

    def z_up(self):
        """Update configuration for z-up coordinate system."""
        self.sensor.location = np.array([0, -5, 0])
        self.sensor.up = np.array([0, 0, 1])
        for emitter in self.emitters:
            if isinstance(emitter, Envmap):
                emitter.up = np.array([0, 0, 1])
                emitter.rotation = 180.0

    def z_down(self):
        """Update configuration for z-down coordinate system."""
        self.sensor.location = np.array([0, 5, 0])
        self.sensor.up = np.array([0, 0, -1])
        for emitter in self.emitters:
            if isinstance(emitter, Envmap):
                emitter.up = np.array([0, 0, -1])
                emitter.rotation = 180.0

    def y_up(self):
        """Update configuration for y-up coordinate system."""
        self.sensor.location = np.array([0, 0, 5])
        self.sensor.up = np.array([0, 1, 0])
        for emitter in self.emitters:
            if isinstance(emitter, Envmap):
                emitter.up = np.array([0, 1, 0])
                emitter.rotation = 180

    def y_down(self):
        """Update configuration for y-down coordinate system."""
        self.sensor.location = np.array([0, 0, -5])
        self.sensor.up = np.array([0, -1, 0])
        for emitter in self.emitters:
            if isinstance(emitter, Envmap):
                emitter.up = np.array([0, -1, 0])
                emitter.rotation = 180

    # ------------------------------------------------------------------ #
    # render_passes – primary interface                                    #
    # ------------------------------------------------------------------ #

    @property
    def render_passes(self) -> set[str]:
        """Set of active render passes.

        Valid pass names are ``"albedo"``, ``"depth"``, ``"normal"``, and
        ``"facet_id"``.  Assigning a new collection replaces the entire set
        and re-synchronises the Mitsuba AOV integrator accordingly.

        Example::

            config.render_passes = {"albedo", "depth"}
        """
        return self._render_passes

    @render_passes.setter
    def render_passes(self, value: set[str] | list[str]):
        """Replace the active render-pass set and re-sync AOV integrator."""
        self._render_passes = set(value)
        self.__sync_aovs()

    # ------------------------------------------------------------------ #
    # Convenience boolean aliases                                          #
    # ------------------------------------------------------------------ #

    @property
    def albedo(self) -> bool:
        """Whether the albedo pass is active.  Alias for ``"albedo" in render_passes``."""
        return "albedo" in self._render_passes

    @albedo.setter
    def albedo(self, value: bool):
        """Add or remove the albedo pass.  Also updates the Mitsuba AOV integrator."""
        if value:
            self._render_passes.add("albedo")
        else:
            self._render_passes.discard("albedo")
        self.__sync_aovs()

    @property
    def depth(self) -> bool:
        """Whether the depth pass is active.  Alias for ``"depth" in render_passes``."""
        return "depth" in self._render_passes

    @depth.setter
    def depth(self, value: bool):
        """Add or remove the depth pass.  Also updates the Mitsuba AOV integrator."""
        if value:
            self._render_passes.add("depth")
        else:
            self._render_passes.discard("depth")
        self.__sync_aovs()

    @property
    def normal(self) -> bool:
        """Whether the shading-normal pass is active.  Alias for ``"normal" in render_passes``."""
        return "normal" in self._render_passes

    @normal.setter
    def normal(self, value: bool):
        """Add or remove the normal pass.  Also updates the Mitsuba AOV integrator."""
        if value:
            self._render_passes.add("normal")
        else:
            self._render_passes.discard("normal")
        self.__sync_aovs()

    @property
    def facet_id(self) -> bool:
        """Whether the facet-ID pass is active.  Alias for ``"facet_id" in render_passes``.

        When active the Blender backend performs a second render after the
        main one.  Every mesh face is colored with the RGB encoding of its
        zero-based index (R = high byte, G = mid byte, B = low byte) using a
        flat Emission shader so lighting has no effect.  The output is written
        to ``<stem>_facet_id<ext>`` with gamma correction, temporal blending,
        and pixel filtering all disabled so pixel values can be decoded
        directly::

            fid = (R << 16) | (G << 8) | B

        Background pixels have ``A = 0`` and can be masked out.  Supports up
        to 2**24 − 1 ≈ 16.7 M faces.
        """
        return "facet_id" in self._render_passes

    @facet_id.setter
    def facet_id(self, value: bool):
        """Add or remove the facet-ID pass."""
        if value:
            self._render_passes.add("facet_id")
        else:
            self._render_passes.discard("facet_id")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def __sync_aovs(self):
        """Rebuild the Mitsuba AOV integrator from the current render-pass set.

        Strips any existing AOV wrapper and re-adds only the passes that are
        currently active, ensuring the integrator always reflects the exact
        state of ``_render_passes``.
        """
        # Strip the AOV wrapper (if any) to start from the base integrator.
        if isinstance(self.integrator, AOV):
            self.integrator = self.integrator.integrator or Path()

        # Re-add AOVs for every active pass that has a Mitsuba counterpart.
        _pass_to_aov = {
            "albedo": "albedo:albedo",
            "depth": "depth:depth",
            "normal": "sh_normal:sh_normal",
        }
        for pass_name, aov_str in _pass_to_aov.items():
            if pass_name in self._render_passes:
                if not isinstance(self.integrator, AOV):
                    self.integrator = AOV(aovs=[aov_str], integrator=self.integrator)
                elif aov_str not in self.integrator.aovs:
                    self.integrator.aovs.append(aov_str)
