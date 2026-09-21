"""Figure-level camera, lighting, environment, and output grammar."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, cast

from ..common.color import ColorLike
from ..setup import Config
from ..setup.emitter import Directional as DirectionalEmitter
from ..setup.emitter import Envmap, Point as PointEmitter
from ..setup.sensor import Orthographic, Perspective, ThinLens
from .layer import Layer


FovAxis = Literal["x", "y", "diagonal", "smaller", "larger"]
RenderPassName = Literal["beauty", "albedo", "depth", "normal", "facet_id"]


@dataclass(frozen=True, slots=True)
class PerspectiveCamera:
    eye: tuple[float, float, float] = (0.0, 0.0, 5.0)
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov: float = 28.8415
    fov_axis: FovAxis = "smaller"
    near: float = 0.01
    far: float = 10000.0

    def __post_init__(self) -> None:
        _validate_camera(self.eye, self.target, self.up, self.near, self.far)
        if not 0.0 < self.fov < 180.0:
            raise ValueError("Camera field of view must be in (0, 180).")


@dataclass(frozen=True, slots=True)
class OrthographicCamera:
    eye: tuple[float, float, float] = (0.0, 0.0, 5.0)
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    near: float = 0.01
    far: float = 10000.0
    scale: float = 2.0

    def __post_init__(self) -> None:
        _validate_camera(self.eye, self.target, self.up, self.near, self.far)
        if self.scale <= 0.0:
            raise ValueError("OrthographicCamera.scale must be positive.")


@dataclass(frozen=True, slots=True)
class ThinLensCamera(PerspectiveCamera):
    aperture_radius: float = 0.1
    focus_distance: float = 0.0

    def __post_init__(self) -> None:
        super(ThinLensCamera, self).__post_init__()
        if self.aperture_radius < 0.0:
            raise ValueError("ThinLensCamera.aperture_radius must be non-negative.")
        if self.focus_distance < 0.0:
            raise ValueError("ThinLensCamera.focus_distance must be non-negative.")


Camera = PerspectiveCamera | OrthographicCamera | ThinLensCamera


@dataclass(frozen=True, slots=True)
class PointLight:
    position: tuple[float, float, float] = (0.0, 0.0, 5.0)
    color: ColorLike = "white"
    intensity: float = 1.0

    def __post_init__(self) -> None:
        if self.intensity < 0.0:
            raise ValueError("PointLight.intensity must be non-negative.")


@dataclass(frozen=True, slots=True)
class DirectionalLight:
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    color: ColorLike = "white"
    intensity: float = 1.0

    def __post_init__(self) -> None:
        if self.intensity < 0.0:
            raise ValueError("DirectionalLight.intensity must be non-negative.")
        if sum(value * value for value in self.direction) <= 1e-20:
            raise ValueError("DirectionalLight.direction must be non-zero.")


Light = PointLight | DirectionalLight


@dataclass(frozen=True, slots=True)
class Environment:
    path: Path | None = None
    scale: float = 1.0
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    rotation: float = 180.0
    visible: bool = False
    enabled: bool = True
    _source: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.scale < 0.0:
            raise ValueError("Environment.scale must be non-negative.")


@dataclass(frozen=True, slots=True)
class OutputSettings:
    width: int = 1024
    height: int = 800
    background: Literal["light", "dark"] = "dark"
    passes: tuple[RenderPassName, ...] = ("beauty",)
    sampler_seed: int = 0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Output width and height must be positive.")
        if len(set(self.passes)) != len(self.passes):
            raise ValueError("Output passes must not contain duplicates.")


@dataclass(frozen=True, slots=True)
class SceneSettings:
    camera: Camera | None = None
    lights: tuple[Light, ...] | None = None
    environment: Environment | None = None
    output: OutputSettings | None = None

    def to_config(self, base: Config | None = None) -> Config:
        """Resolve declarative settings over Config defaults or a supplied base."""
        config = copy.deepcopy(base) if base is not None else Config()
        if self.camera is not None:
            config.sensor = _sensor(self.camera)
        if self.lights is not None:
            config.emitters = [
                emitter
                for emitter in config.emitters
                if isinstance(emitter, Envmap)
            ]
            config.emitters.extend(_emitter(light) for light in self.lights)
        if self.environment is not None:
            config.emitters = [
                emitter
                for emitter in config.emitters
                if not isinstance(emitter, Envmap)
            ]
            if self.environment.enabled:
                filename = self.environment.path
                if filename is None:
                    filename = Envmap().filename
                config.emitters.insert(
                    0,
                    Envmap(
                        filename=filename,
                        scale=self.environment.scale,
                        up=list(self.environment.up),
                        rotation=self.environment.rotation,
                    ),
                )
            config.environment_visible = self.environment.visible
            config.integrator.hide_emitters = not self.environment.visible
        if self.output is not None:
            config.film.width = self.output.width
            config.film.height = self.output.height
            config.sampler.seed = self.output.sampler_seed
            config.background = self.output.background
            passes: set[str] = {str(item) for item in self.output.passes}
            config.render_passes = passes - {"beauty"}
        return config


@dataclass(frozen=True, slots=True)
class Figure:
    """A composed layer plus reproducible scene-level rendering intent."""

    layer: Layer
    scene: SceneSettings = field(default_factory=SceneSettings)

    def camera(self, camera: Camera | str = "perspective", **kwargs) -> "Figure":
        if isinstance(camera, str):
            kind = camera
            if kind in {"fit", "principal_axis", "attribute_extremum", "section"}:
                from ..framing import resolve_camera

                output = self.scene.output
                kwargs.setdefault(
                    "resolution",
                    (output.width, output.height) if output is not None else (1024, 800),
                )
                resolved = resolve_camera(
                    self.layer,
                    cast(Literal["fit", "principal_axis", "attribute_extremum", "section"], kind),
                    **kwargs,
                )
            elif kind == "perspective":
                resolved = PerspectiveCamera(**kwargs)
            elif kind == "orthographic":
                resolved = OrthographicCamera(**kwargs)
            elif kind == "thin_lens":
                resolved = ThinLensCamera(**kwargs)
            else:
                raise ValueError(f"Unknown camera kind: {kind!r}")
        else:
            if kwargs:
                raise TypeError("Keyword camera options require a string camera kind.")
            resolved = camera
        return replace(self, scene=replace(self.scene, camera=resolved))

    def turntable(self, **kwargs) -> tuple["Figure", ...]:
        """Return figures with evenly spaced fitted cameras around the scene."""
        from ..framing import turntable_cameras

        output = self.scene.output
        kwargs.setdefault(
            "resolution",
            (output.width, output.height) if output is not None else (1024, 800),
        )
        return tuple(
            replace(self, scene=replace(self.scene, camera=camera))
            for camera in turntable_cameras(self.layer, **kwargs)
        )

    def light(self, light: Light | str = "point", **kwargs) -> "Figure":
        if isinstance(light, str):
            kind = light
            if kind == "point":
                resolved: Light = PointLight(**kwargs)
            elif kind == "directional":
                resolved = DirectionalLight(**kwargs)
            else:
                raise ValueError(f"Unknown light kind: {kind!r}")
        else:
            if kwargs:
                raise TypeError("Keyword light options require a string light kind.")
            resolved = light
        lights = self.scene.lights or ()
        return replace(self, scene=replace(self.scene, lights=(*lights, resolved)))

    def clear_lights(self) -> "Figure":
        return replace(self, scene=replace(self.scene, lights=()))

    def environment(
        self, environment: Environment | str | Path | None = None, **kwargs
    ) -> "Figure":
        if isinstance(environment, (str, Path)):
            environment = Environment(path=Path(environment), **kwargs)
        elif environment is None:
            environment = Environment(**kwargs)
        elif kwargs:
            raise TypeError("Keyword environment options require a path or None.")
        return replace(self, scene=replace(self.scene, environment=environment))

    def output(self, output: OutputSettings | None = None, **kwargs) -> "Figure":
        if output is None:
            output = OutputSettings(**kwargs)
        elif kwargs:
            raise TypeError("Keyword output options cannot accompany OutputSettings.")
        return replace(self, scene=replace(self.scene, output=output))

    def to_config(self, base: Config | None = None) -> Config:
        return self.scene.to_config(base)

    def to_spec(self, *, data_ids=None, function_ids=None):
        from ..spec import to_spec

        return to_spec(self, data_ids=data_ids, function_ids=function_ids)

    def to_json(self, *, data_ids=None, function_ids=None, indent=2, canonical=False):
        return self.to_spec(
            data_ids=data_ids, function_ids=function_ids
        ).to_json(indent=indent, canonical=canonical)


def _validate_camera(eye, target, up, near: float, far: float) -> None:
    if eye == target:
        raise ValueError("Camera eye and target must differ.")
    if sum(value * value for value in up) <= 1e-20:
        raise ValueError("Camera up vector must be non-zero.")
    if near <= 0.0 or far <= near:
        raise ValueError("Camera clipping planes require 0 < near < far.")


def _sensor(camera: Camera):
    if isinstance(camera, ThinLensCamera):
        return ThinLens(
            location=list(camera.eye),
            target=list(camera.target),
            up=list(camera.up),
            near_clip=camera.near,
            far_clip=camera.far,
            fov=camera.fov,
            fov_axis=camera.fov_axis,
            aperture_radius=camera.aperture_radius,
            focus_distance=camera.focus_distance,
        )
    if isinstance(camera, PerspectiveCamera):
        return Perspective(
            location=list(camera.eye),
            target=list(camera.target),
            up=list(camera.up),
            near_clip=camera.near,
            far_clip=camera.far,
            fov=camera.fov,
            fov_axis=camera.fov_axis,
        )
    return Orthographic(
        location=list(camera.eye),
        target=list(camera.target),
        up=list(camera.up),
        scale=camera.scale,
        near_clip=camera.near,
        far_clip=camera.far,
    )


def _emitter(light: Light):
    if isinstance(light, PointLight):
        return PointEmitter(
            position=list(light.position), intensity=light.intensity, color=light.color
        )
    return DirectionalEmitter(
        direction=list(light.direction), intensity=light.intensity, color=light.color
    )


def as_figure(value: Layer | Figure) -> Figure:
    return value if isinstance(value, Figure) else Figure(value)


__all__ = [
    "Camera",
    "DirectionalLight",
    "Environment",
    "Figure",
    "Light",
    "OrthographicCamera",
    "OutputSettings",
    "PerspectiveCamera",
    "PointLight",
    "SceneSettings",
    "ThinLensCamera",
    "as_figure",
]
