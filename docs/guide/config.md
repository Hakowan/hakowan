# Configuration

The configuration class defines a number of scene-related settings that mapped directly from Mitsuba.

* `sensor`: Camera-related settings.
* `film`: The output image format.
* `sampler`: Sampler settings.
* `emitters`: An array of light settings.
* `integrator`: Settings for different rendering techniques.

The configuration object provides a number of handy functions for commonly used configurations. For
example, the default scene using Y axis as the up direction. To change the up direction:

```py
config = hkw.config()

config.z_up() # Change the up direction to +Z axis.
config.z_down() # Change the up direction to -Z axis.
config.y_up() # Change the up direction to +Y axis.
config.y_down() # Change the up direction to -Y axis.
```

Another useful feature is to enable albedo-only rendering:

```py
config.albedo_only = True
```

See the [Penny](../examples/penny.md) example for a usage of albedo-only rendering.

## Sensor settings

Sensor defines the camera setting. All sensors supports the following settings:

| Setting | Type | Description |
|---------|------|-------------|
| `location` | `NDArray` | The camera location. (default: `[0, 0, 5]`) |
| `target` | `NDArray` | The look-at location. (default: `[0, 0, 0]`) |
| `up` | `NDArray` | The up direction. (default: `[0, 1, 0]`) |
| `near_clip` | `float` | The near clipping plane (default: `1e-2`) |
| `far_clip` | `float` | The far clipping plane (default: `1e4`) |

Note that `location` and `target` are location after applying global transformation that put the
bounding box of the scene within a unit sphere centered at the origin.
Typically, only `location` needs to be changed based on need.

### Perspective sensor

Perspective sensor models the traditional pin-hole camera. It support the following attributes.

| Setting | Type | Description |
|---------|------|-------------|
| `fov` | `float` | The field of view in degrees. (default: 28.8415) |
| `fov_axis` | `str` | The fox axis (default: `smaller`) |

```py
config.sensor = hkw.setup.sensor.Perspective(fov=30)
```

### Orthographic sensor

Orthographic sensor uses the orthographic projection. This sensor type does not produce
foreshortening effect.

```py
config.sensor = hkw.setup.sensor.Orthograpic()
```

See the [Incremental Potential Contact](../examples/ipc.md) example for an usage of orthographic
sensor.

### Thin lens sensor

This sensor models the thin lens camera model which allows rendering with a specific depth of field.

| Setting | Type | Description |
|---------|------|-------------|
| `aperture_radius` | `float` | The aperture radius. (default: 0.1) |
| `focus_distance` | `float` | The focus distance. (default: 0.0) |

## Film settings

Film settings provide output image specification.

| Setting | Type | Description |
|---------|------|-------------|
| `width` | `int` | The output image width (default: 1024) |
| `height` | `int` | The output image height (default: 800) |
| `file_format` | `str` | The output image file format (default: `openexr`) |
| `pixel_format` | `str` | The output image pixel format (default: `rgba`) |
| `component_format` | `str` | The output image pixel format (default: `float16`) |
| `crop_offset` | `NDArray` | The top left corner of the crop region (default: `None`) |
| `crop_size` | `NDArray` | The size of the crop region (default: `None`) |

Note that `crop_offset` and `crop_size` together define the crop region. Setting either to `None`
indicate no cropping.
`file_format`, `pixel_format` and `component_format` are for expert-usage only.

```py
# Generate 4K rendering.
config.film.width = 3840
config.film.height = 2160
```

## Sampler

Sampler defines the sample strategy used for rendering. It supports the following parameters:

| Setting | Type | Description |
|---------|------|-------------|
| `sample_count` | `int` | Number of samples per pixel (default: 256) |
| `seed` | `int` | Random number seed (default: 0) |

Note that sampler settings typically do not need to be changed unless the rendered image is too noisy.

```py
config.sampler.sample_count = 128 # Reduce sample count
```

## Emitter settings

Emitters represent the light source in the scene. The emitting setting directly influence the shading
and shadow of the scene. Multiple emitters can be used at the same time.

### Point emitter

A point emitter is a point light source, which tends to generate share shadow boundaries.

```py
l = hkw.setup.emitter.Point(position=[0, 0, 5], intensity="#FFEAC5")
config.emitters.append(l)
```

### Environment emitter

A environment emitter defines [image-based lighting](https://en.wikipedia.org/wiki/Image-based_lighting) (IBL).
It is the preferred emitter setting in Hakowan. By default, we use "At the Window" environment map
from [Bernhard Vogl](https://dativ.at/lightprobes/) (free for non-commercial usage).

```py
l = hkw.setup.emitter.Envmap(filename="envmap.exr")
config.emitters.append(l)
```

## Integrator settings

Integrator settings defines the render technique used.

| Setting | Type | Description |
|---------|------|-------------|
| `hide_emitters` | `bool` | Whether to render emitters (default: `False`) |

### Path integrator

The `Path` integrator mirrors the [Mitsuba's `Path` integrator](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html#path-tracer-path). It is the default integrator used in Hakowan.

### AOV integrator

The `AOV` integrator mirrors the [Mitsuba's `AOV`
integrator](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html#arbitrary-output-variables-integrator-aov).
It is useful for visualizing specific attribute without shading.

### VolPath integrator

The `VolPath` integrator mirrors the [Mitsuba's `VolPath` integrator](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html#volumetric-path-tracer-volpath).
It is useful for rendering scene with volumetric elements (e.g. with
[Dielectric][hakowan.material.Dielectric] material).
