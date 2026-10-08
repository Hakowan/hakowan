#!/usr/bin/env python

import math
import tempfile
import webbrowser
from pathlib import Path
from typing import Literal

import hakowan as hkw
import lagrange
import numpy as np
from PIL import Image
from tqdm import tqdm

from .cli.arguments import InspectArgs, parse_args
from .cli.camera import compute_camera_matrix, save_camera_matrix
from .cli.layers import build_layer
from .cli.materials import get_tmp_image_name


def main() -> None:
    """
    Entry point for the command-line interface for mesh rendering.
    Parses command-line arguments and orchestrates the mesh rendering process.
    """
    args = parse_args()

    if isinstance(args, InspectArgs):
        summary = hkw.inspect(args.input_mesh)
        print(f"Attributes in {args.input_mesh}:")
        if not summary.attributes:
            print("  (none)")
        for attribute in summary.attributes:
            minimum = (
                f"({', '.join(map(str, attribute.minimum))})"
                if isinstance(attribute.minimum, list)
                else str(attribute.minimum)
            )
            maximum = (
                f"({', '.join(map(str, attribute.maximum))})"
                if isinstance(attribute.maximum, list)
                else str(attribute.maximum)
            )
            channel_label = "channel" if attribute.channels == 1 else "channels"
            print(
                f"  {attribute.name} ({attribute.element}, "
                f"{attribute.channels} {channel_label}): min={minimum} max={maximum}"
            )
        return

    if len(args.input_mesh) > 6:
        raise SystemExit(
            f"At most 6 input meshes are supported, got {len(args.input_mesh)}."
        )

    hkw.logger.setLevel(args.log_level.upper())
    lagrange.logger.setLevel(args.log_level.upper())

    # Load each mesh once so facet-free inputs can select point rendering before
    # their layers are styled. Multiple meshes are arranged in a near-square grid
    # facing the camera (rows stacked along the up-axis). With more than one mesh,
    # normalize each to unit size so they show at the same on-screen scale despite
    # unrelated source coordinate systems.
    meshes = [
        lagrange.io.load_mesh(mesh_path, quiet=True, stitch_vertices=True)
        for mesh_path in args.input_mesh
    ]
    if not args.point_cloud and all(mesh.num_facets == 0 for mesh in meshes):
        args.point_cloud = True

    normalize = len(meshes) > 1
    layers = [
        build_layer(args, mesh_path, normalize=normalize, mesh=mesh)
        for mesh_path, mesh in zip(args.input_mesh, meshes, strict=True)
    ]
    layer = hkw.grid(
        layers,
        columns=2 if len(layers) == 4 else min(len(layers), 3),
        row_axis="z" if args.z_up else "y",
    )

    config = hkw.config()
    [config.film.width, config.film.height] = args.resolution
    if args.z_up:
        config.z_up()
    if args.camera is not None:
        config.sensor.location = args.camera

    if args.albedo:
        config.albedo = True
    if args.depth:
        config.depth = True
    if args.shading_normal:
        config.normal = True
    if args.facet_id:
        config.facet_id = True

    if args.output:
        output_file = Path(args.output)
    elif args.action == "view" and args.turn_table == 0 and not args.no_open:
        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False
        ) as temporary_file:
            output_file = Path(temporary_file.name)
    else:
        default_suffix = ".html" if args.action == "view" else ".png"
        output_file = Path(args.input_mesh[0]).with_suffix(default_suffix)

    if args.camera_matrix:
        cam_data = compute_camera_matrix(config)
        save_camera_matrix(cam_data, Path(args.camera_matrix))

    turntable_pass: Literal["beauty", "albedo", "depth", "normal"] = "beauty"
    if args.backend == "webgl" and args.turn_table:
        requested_passes: list[Literal["albedo", "depth", "normal"]] = []
        if args.albedo:
            requested_passes.append("albedo")
        if args.depth:
            requested_passes.append("depth")
        if args.shading_normal:
            requested_passes.append("normal")
        if len(requested_passes) > 1:
            raise SystemExit(
                "WebGL turn-table capture supports at most one render pass."
            )
        if requested_passes:
            turntable_pass = requested_passes[0]

    if args.turn_table == 0:
        kwargs = {}
        if args.serialize:
            kwargs["yaml_file"] = output_file.with_suffix(".yaml")
            if args.backend == "blender":
                kwargs["blend_file"] = output_file.with_suffix(".blend")
        result = hkw.render(
            layer,
            config,
            filename=output_file,
            backend=args.backend,
            **kwargs,
        )
        if args.action == "view" and not args.no_open:
            opened_path = (
                Path(result)
                if isinstance(result, (str, Path))
                else output_file.with_suffix(".html")
            )
            opened = webbrowser.open(opened_path.resolve().as_uri())
            if not opened:
                hkw.logger.warning(
                    "Could not open WebGL viewer; the HTML file is saved at %s",
                    opened_path.resolve(),
                )
    else:
        if args.z_up:
            axis = np.array([0, 0, 1], dtype=float)
        else:
            axis = np.array([0, 1, 0], dtype=float)

        # Get initial camera position in normalized coordinates
        # (object is at origin with bounding radius 1)
        initial_camera = np.array(config.sensor.location, dtype=float)

        # Render frames to temporary files
        frames = []
        temp_files = []
        try:
            for i in tqdm(
                range(args.turn_table), desc="Rendering frames", unit="frame"
            ):
                # Rotate camera position around origin in normalized coordinates
                angle = i * 2 * math.pi / args.turn_table
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)

                # Rodrigues' rotation formula to rotate camera position around axis
                rotated_camera = (
                    initial_camera * cos_a
                    + np.cross(axis, initial_camera) * sin_a
                    + axis * np.dot(axis, initial_camera) * (1 - cos_a)
                )

                # Update camera position for this frame
                frame_config = hkw.config()
                frame_config.film.width = config.film.width
                frame_config.film.height = config.film.height
                if args.z_up:
                    frame_config.z_up()
                frame_config.sensor.location = list(rotated_camera)
                frame_config.sensor.target = [0.0, 0.0, 0.0]  # Look at origin

                if args.albedo:
                    frame_config.albedo = True
                if args.depth:
                    frame_config.depth = True
                if args.shading_normal:
                    frame_config.normal = True
                if args.facet_id:
                    frame_config.facet_id = True

                frame_file = get_tmp_image_name()
                temp_files.append(frame_file)
                if args.backend == "webgl":
                    camera_eye = (
                        float(rotated_camera[0]),
                        float(rotated_camera[1]),
                        float(rotated_camera[2]),
                    )
                    up_vector = np.asarray(frame_config.sensor.up, dtype=np.float64)
                    camera_up = (
                        float(up_vector[0]),
                        float(up_vector[1]),
                        float(up_vector[2]),
                    )
                    hkw.snapshot(
                        layer,
                        pass_name=turntable_pass,
                        camera=hkw.CameraState(
                            eye=camera_eye,
                            target=(0.0, 0.0, 0.0),
                            up=camera_up,
                            fov=float(getattr(frame_config.sensor, "fov", 28.8415)),
                            near=float(frame_config.sensor.near_clip),
                            far=float(frame_config.sensor.far_clip),
                        ),
                        resolution=(frame_config.film.width, frame_config.film.height),
                        config=frame_config,
                        filename=frame_file,
                    )
                else:
                    hkw.render(
                        layer, frame_config, filename=frame_file, backend=args.backend
                    )
                # Load frame and ensure solid white background
                with Image.open(frame_file) as img:
                    if img.mode == "RGBA":
                        # Create a white background
                        white_bg = Image.new("RGB", img.size, (255, 255, 255))
                        white_bg.paste(
                            img, mask=img.split()[3]
                        )  # Use alpha channel as mask
                        frames.append(white_bg)
                    else:
                        frames.append(img.convert("RGB"))

            # Save as animated GIF
            gif_file = output_file.with_suffix(".gif")
            frames[0].save(
                gif_file,
                save_all=True,
                append_images=frames[1:],
                duration=100,  # milliseconds per frame
                loop=0,
            )
        finally:
            for temp_image in temp_files:
                temp_image.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
