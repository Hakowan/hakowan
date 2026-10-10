"""Command-line argument parsing and option types."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from typing import Literal

from ..backends import BackendName

import hakowan as hkw


class InspectArgs(argparse.Namespace):
    action: Literal["inspect"]
    input_mesh: str


class RenderArgs(argparse.Namespace):
    action: Literal["render", "view"]
    backend: BackendName
    point_cloud: bool
    point_size: float
    z_up: bool
    comp: bool
    explode: float | None
    normal: str | None
    input_mesh: list[str]
    camera: Sequence[float] | None
    output: str | None
    wireframe: bool
    wire_thickness: float
    resolution: Sequence[int]
    material: str
    color: str
    back_color: str | None
    orient_pca: bool
    rotate: float | None
    turn_table: int
    two_sided: bool
    seams: bool
    uv: bool
    albedo: bool
    depth: bool
    shading_normal: bool
    facet_id: bool
    isoline: bool
    clip: tuple[str, float] | None
    singularity: bool
    vector_field: str | None
    cross_field: str | None
    field_style: Literal["streamline", "arrow", "fur"]
    num_streamlines: int
    streamline_length: float | None
    streamline_color: str
    quad_only: bool
    uv_scale: float
    categorical: bool
    saturation: float
    whiteness: float
    log_level: str
    serialize: bool
    no_open: bool
    camera_matrix: str | None


_CLIP_AXES = {"x", "-x", "y", "-y", "z", "-z"}


def _render_backend(*, facet_id: bool = False) -> BackendName:
    """Choose an installed native renderer for the CLI still-image action."""
    available = set(hkw.list_backends())
    if facet_id:
        if "blender" in available:
            return "blender"
        raise SystemExit("The --facet-id pass requires the Blender backend.")
    for backend in ("mitsuba", "blender"):
        if backend in available:
            return backend
    raise SystemExit(
        "hkw render requires Mitsuba or Blender. Install 'hakowan[mitsuba]' or "
        "'hakowan[blender]', or use 'hkw view' for a WebGL viewer."
    )


def _clip_arg(value: str) -> tuple[str, float]:
    """Parse AXIS or AXIS:VALUE (e.g. 'x', '-y:0.3')."""
    if ":" in value:
        axis, val = value.rsplit(":", 1)
        try:
            fval = float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"clip value must be a number, got {val!r}"
            )
        if not (0.0 <= fval <= 1.0):
            raise argparse.ArgumentTypeError(
                f"clip value must be in [0, 1], got {fval}"
            )
    else:
        axis, fval = value, 0.5
    if axis not in _CLIP_AXES:
        raise argparse.ArgumentTypeError(
            f"clip axis must be one of {sorted(_CLIP_AXES)}, got {axis!r}"
        )
    return axis, fval


def _saturation_arg(value: str) -> float:
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"saturation must be a number, got {value!r}")
    if not math.isfinite(v) or v < 0:
        raise argparse.ArgumentTypeError(
            f"saturation must be non-negative and finite, got {v}"
        )
    return v


def _explode_arg(value: str) -> float:
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"explode magnitude must be a number, got {value!r}"
        )
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"explode magnitude must be finite, got {v}")
    return v


def _whiteness_arg(value: str) -> float:
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"whiteness must be a number, got {value!r}")
    if not math.isfinite(v) or not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"whiteness must be in [0, 1], got {v}")
    return v


def _turn_table_arg(value: str) -> int:
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"turn-table count must be a non-negative integer, got {value!r}"
        )
    if count < 0:
        raise argparse.ArgumentTypeError(
            f"turn-table count must be non-negative, got {count}"
        )
    return count


def parse_args(argv: Sequence[str] | None = None) -> InspectArgs | RenderArgs:
    """Parse the action and mesh-rendering options."""
    argv = sys.argv[1:] if argv is None else list(argv)
    action_parser = argparse.ArgumentParser(
        description="Render, view, or inspect a mesh."
    )
    action_parser.add_argument("action", choices=("render", "view", "inspect"))
    action: Literal["render", "view", "inspect"] = action_parser.parse_args(
        argv[:1]
    ).action

    if action == "inspect":
        parser = argparse.ArgumentParser(
            description="List attributes in a mesh.",
            prog=f"{action_parser.prog} inspect",
        )
        parser.add_argument("input_mesh", help="Input mesh file to inspect.")
        inspect_args = parser.parse_args(argv[1:], namespace=InspectArgs())
        inspect_args.action = "inspect"
        return inspect_args

    parser = argparse.ArgumentParser(
        description="Render a mesh.", prog=f"{action_parser.prog} {action}"
    )
    parser.add_argument(
        "--point-cloud",
        help="Render point cloud (enabled automatically for facet-free inputs)",
        action="store_true",
    )
    parser.add_argument(
        "--point-size",
        help="Point diameter in output pixels",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--z-up", help="Use Z-up coordinate system.", action="store_true"
    )
    parser.add_argument("--comp", help="Visualize components", action="store_true")
    parser.add_argument(
        "--explode",
        help=(
            "Explode the mesh: displace each connected component outward from the "
            "mesh center by VAL times its offset (0 = no move, 1 = strong). "
            "Negative values implode."
        ),
        type=_explode_arg,
        default=None,
        metavar="VAL",
    )
    parser.add_argument(
        "--normal",
        help=(
            "Normal field for shading. 'facet' or 'vertex' computes normals "
            "from geometry; any other value is treated as the name of a "
            "pre-existing normal attribute on the mesh."
        ),
        default=None,
        metavar="FIELD",
    )
    parser.add_argument(
        "input_mesh",
        nargs="+",
        help="Input mesh file(s), up to 6, arranged in a grid (3 columns max).",
    )
    parser.add_argument("--camera", help="Camera location", nargs=3, type=float)
    parser.add_argument("-o", "--output", help="Output image file.")
    parser.add_argument(
        "-w", "--wireframe", help="Render wireframe", action="store_true"
    )
    parser.add_argument(
        "--wire-thickness",
        help="Wireframe, seam, and field-line diameter in pixels (default: 0.5)",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--resolution", help="Resolution", nargs=2, type=int, default=(1024, 800)
    )
    parser.add_argument(
        "--material",
        default="plastic",
        help=(
            "Material types: plastic, roughplastic, glass, diffuse, uv, normal, texture, vertex_color. "
            "If none of these, it is treated as a scalar field attribute."
        ),
    )
    parser.add_argument("--color", help="Material color", type=str, default="ivory")
    parser.add_argument(
        "--back-color",
        help="Back-face color (enables two-sided rendering with a distinct back material).",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--orient-pca",
        action="store_true",
        help=(
            "Rotate and translate the mesh so PCA principal axes align with world axes: "
            "largest variance along +Y (default up), or +Z when --z-up is set; "
            "second/third variance along the remaining axes (right-handed). Applied before --rotate."
        ),
    )
    parser.add_argument(
        "--rotate", help="Rotate the mesh (degrees)", type=float, default=None
    )
    parser.add_argument(
        "--turn-table",
        help=(
            "Turn table animation (non-negative frame count; 0 disables). WebGL "
            "capture requires hakowan[observe] and Playwright Chromium."
        ),
        type=_turn_table_arg,
        default=0,
    )
    parser.add_argument(
        "--two-sided", help="Render both sides of the mesh", action="store_true"
    )
    parser.add_argument("--seams", help="Render uv seams", action="store_true")
    parser.add_argument("--uv", help="Render uv map", action="store_true")
    parser.add_argument("--albedo", help="Render albedo", action="store_true")
    parser.add_argument("--depth", help="Render depth", action="store_true")
    parser.add_argument("--shading-normal", help="Render normal", action="store_true")
    parser.add_argument(
        "--facet-id",
        help=(
            "Render each facet colored by its index (RGB-encoded). "
            "No gamma correction, no blending, no anti-aliasing. "
            "Blender backend only."
        ),
        action="store_true",
    )
    parser.add_argument("--isoline", help="Render isoline", action="store_true")
    parser.add_argument(
        "--clip",
        help="Slice the mesh at AXIS or AXIS:VALUE, keeping the selected side. AXIS is x/y/z/-x/-y/-z; VALUE is a fraction of the bbox size in [0,1] (default 0.5). The cut is left open. E.g. '--clip x' or '--clip -y:0.3'.",
        type=_clip_arg,
        default=None,
        metavar="AXIS[:VALUE]",
    )
    parser.add_argument("--singularity", help="Show singularity", action="store_true")
    streamline_group = parser.add_mutually_exclusive_group()
    streamline_group.add_argument(
        "--vector-field",
        metavar="ATTR",
        help="Overlay a plain vector field attribute (streamlines or arrows, see --field-style).",
        default=None,
    )
    streamline_group.add_argument(
        "--cross-field",
        metavar="ATTR",
        help="Overlay a 4-RoSy cross-field attribute (streamlines or arrows, see --field-style).",
        default=None,
    )
    parser.add_argument(
        "--field-style",
        choices=["streamline", "arrow", "fur"],
        default="streamline",
        help=(
            "How to visualize --vector-field / --cross-field: 'streamline' "
            "(default), 'arrow', or 'fur' (hair strands combed along the field; "
            "--vector-field only). For fur, --num-streamlines sets the strand "
            "count and --streamline-color sets the coat color."
        ),
    )

    parser.add_argument(
        "--num-streamlines",
        help="Number of streamlines to render (default: 500).",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--streamline-length",
        help="Max streamline half-length as a fraction of bbox diagonal (default: 0.5).",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--streamline-color",
        help="Streamline/arrow color (default black).",
        type=str,
        default="black",
    )
    parser.add_argument(
        "--quad-only", help="Only render quad facets", action="store_true"
    )
    parser.add_argument("--uv-scale", help="UV scale factor", type=float, default=1.0)
    parser.add_argument(
        "--categorical",
        help="Treat scalar attribute field as categorical (uses discrete colormap).",
        action="store_true",
    )
    parser.add_argument(
        "--saturation",
        help="Texture image saturation (1.0=full color, 0.0=grayscale, must be non-negative). Only applies when --material is a texture image.",
        type=_saturation_arg,
        default=1.0,
    )
    parser.add_argument(
        "--whiteness",
        help="Blend texture image toward pure white (0.0=original, 1.0=white, must be in [0,1]). Only applies when --material is a texture image.",
        type=_whiteness_arg,
        default=0.0,
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="warning",
        help="Logging level",
    )
    parser.add_argument("--serialize", help="Serialize the config", action="store_true")
    parser.add_argument(
        "--no-open",
        help="Do not auto-open the WebGL viewer.",
        action="store_true",
    )
    parser.add_argument(
        "--camera-matrix",
        help=(
            "Save the world-to-camera transform and intrinsic matrix to this file "
            "(.npz or .json) for use in back-projecting pixels onto the mesh."
        ),
        default=None,
    )
    render_args = parser.parse_args(argv[1:], namespace=RenderArgs())
    render_args.action = action
    if action == "view" and render_args.facet_id:
        raise SystemExit("The --facet-id pass is only available with hkw render.")
    render_args.backend = (
        _render_backend(facet_id=render_args.facet_id)
        if action == "render"
        else "webgl"
    )
    return render_args
