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
        albedo: Whether to render albedo (i.e. diffuse color without shading).
        depth: Whether to render a depth pass.
        normal: Whether to render a shading-normal pass.
        facet_id: Whether to render a facet-ID pass (Blender backend only).
            Each face is colored by its zero-based index encoded as RGB so that
            the original index can be recovered from the output pixel values.
            The pass is written as ``<stem>_facet_id<ext>`` alongside the main
            image.  Gamma correction, blending, and anti-aliasing are all
            disabled for this pass.
    """

    sensor: Sensor = field(default_factory=Perspective)
    film: Film = field(default_factory=Film)
    sampler: Sampler = field(default_factory=Independent)
    emitters: list[Emitter] = field(default_factory=lambda: [Envmap()])
    integrator: Integrator = field(default_factory=Path)
    _albedo: bool = False
    _depth: bool = False
    _normal: bool = False
    _facet_id: bool = False

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

    @property
    def albedo(self) -> bool:
        """Whether to render albedo (i.e. without shading)."""
        return self._albedo

    @albedo.setter
    def albedo(self, value: bool):
        """Whether to render albedo (i.e. without shading).

        Note that this setting will modify Config.integrator property.
        """
        self._albedo = value
        if self._albedo:
            self.__add_aov("albedo:albedo")
        else:
            self.__reset_aov()

    @property
    def depth(self) -> bool:
        """Whether to render depth."""
        return self._depth

    @depth.setter
    def depth(self, value: bool):
        """Whether to render depth.

        Note that this setting will modify Config.integrator property.
        """
        self._depth = value
        if self._depth:
            self.__add_aov("depth:depth")
        else:
            self.__reset_aov()

    @property
    def normal(self) -> bool:
        """Whether to render normal."""
        return self._normal

    @normal.setter
    def normal(self, value: bool):
        """Whether to render normal.

        Note that this setting will modify Config.integrator property.
        """
        self._normal = value
        if self._normal:
            self.__add_aov("sh_normal:sh_normal")
        else:
            self.__reset_aov()

    @property
    def facet_id(self) -> bool:
        """Whether to render a facet-ID pass alongside the main image.

        When ``True`` the Blender backend performs a second render after the
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
        return self._facet_id

    @facet_id.setter
    def facet_id(self, value: bool):
        """Enable or disable the facet-ID render pass."""
        self._facet_id = value

    def __add_aov(self, aov: str):
        """Add an AOV to the integrator.

        An AOV integrator is created if one does not already exist. Otherwise, the specific output
        variable will be added to the existing AOV integrator.
        """
        if not isinstance(self.integrator, AOV):
            self.integrator = AOV(aovs=[aov], integrator=self.integrator)
        elif aov not in self.integrator.aovs:
            self.integrator.aovs.append(aov)

    def __reset_aov(self):
        if isinstance(self.integrator, AOV):
            if self.integrator.integrator is not None:
                self.integrator = self.integrator.integrator
            else:
                self.integrator = Path()
