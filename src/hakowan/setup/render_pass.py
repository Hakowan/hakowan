"""Canonical registry of auxiliary render passes (AOVs).

Each pass is described once, here, by a :class:`RenderPass` descriptor carrying
the metadata every backend needs:

- ``name``        – the canonical string used in :attr:`Config.render_passes`.
- ``channels``    – number of image channels the pass produces (1 or 3).
- ``discrete``    – ``True`` for integer-ID passes (e.g. ``facet_id``) that must
  be rendered losslessly: no anti-aliasing, pixel filtering, dithering, or
  tone-mapping.  Continuous shading passes (albedo/depth/normal) leave this
  ``False``.
- ``mitsuba_aov`` – the Mitsuba AOV spec string (e.g. ``"albedo:albedo"``) or
  ``None`` when the pass has no native Mitsuba AOV counterpart.

Backends declare the subset they support via ``SUPPORTED_PASSES`` and the
render dispatcher warns about any requested pass a backend cannot honor, so a
pass is never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class RenderPass:
    """Descriptor for one auxiliary render pass."""

    name: str
    channels: int
    discrete: bool = False
    mitsuba_aov: str | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


# Canonical pass instances. Declaration order is preserved by ``RENDER_PASSES``
# below and used wherever a deterministic ordering matters (e.g. building the
# Mitsuba AOV channel layout).
ALBEDO = RenderPass("albedo", channels=3, mitsuba_aov="albedo:albedo")
DEPTH = RenderPass("depth", channels=1, mitsuba_aov="depth:depth")
NORMAL = RenderPass("normal", channels=3, mitsuba_aov="sh_normal:sh_normal")
FACET_ID = RenderPass("facet_id", channels=3, discrete=True)

# Name -> descriptor lookup (insertion order = declaration order above).
RENDER_PASSES: dict[str, RenderPass] = {
    p.name: p for p in (ALBEDO, DEPTH, NORMAL, FACET_ID)
}


def get_render_pass(
    name: Literal["albedo", "depth", "normal", "facet_id"],
) -> RenderPass:
    """Look up a :class:`RenderPass` by name.

    Args:
        name: Canonical pass name.

    Returns:
        The matching :class:`RenderPass` descriptor.

    Raises:
        ValueError: If ``name`` is not a recognised render pass.
    """
    try:
        return RENDER_PASSES[name]
    except KeyError:
        raise ValueError(
            f"Unknown render pass {name!r}. Valid passes: {sorted(RENDER_PASSES)}."
        ) from None


def aov_path(filename: Path | str, render_pass: RenderPass | str) -> Path:
    """Derive the sidecar output path for a render pass.

    Produces ``<stem>_<pass><ext>`` next to *filename* — the single naming
    convention shared by every file-writing backend (e.g. ``bust.png`` →
    ``bust_albedo.png``, ``bust_facet_id.png``).

    Args:
        filename: Main render output path.
        render_pass: A :class:`RenderPass` descriptor or its canonical name.

    Returns:
        The derived sidecar path.
    """
    name = render_pass.name if isinstance(render_pass, RenderPass) else render_pass
    p = Path(filename)
    return p.with_name(f"{p.stem}_{name}{p.suffix}")


__all__ = [
    "RenderPass",
    "ALBEDO",
    "DEPTH",
    "NORMAL",
    "FACET_ID",
    "RENDER_PASSES",
    "get_render_pass",
    "aov_path",
]
