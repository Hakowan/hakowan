"""Load canonical recipe artifacts from the sibling Hakowan gallery."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any


def find_gallery(explicit: str | Path | None = None) -> Path | None:
    """Find a gallery root from an argument, environment, or sibling checkout."""
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    if os.environ.get("HAKOWAN_GALLERY"):
        candidates.append(Path(os.environ["HAKOWAN_GALLERY"]))
    candidates.append(Path(__file__).resolve().parents[3] / "hakowan-gallery")
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "gallery" / "README.md").is_file():
            return root
    return None


def load_gallery(root: str | Path | None = None) -> tuple[dict[str, Any], ...]:
    """Return manifests and canonical artifacts for every migrated recipe."""
    gallery_root = find_gallery(root)
    if gallery_root is None:
        return ()
    recipes = []
    for manifest_path in sorted((gallery_root / "gallery").glob("*/recipe.toml")):
        folder = manifest_path.parent
        artifacts = folder / "artifacts"
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        recipe: dict[str, Any] = {"folder": folder.name, "manifest": manifest}
        for name in ("inspect", "figure", "validation", "render"):
            path = artifacts / f"{name}.json"
            if path.is_file():
                import json

                recipe[name] = json.loads(path.read_text(encoding="utf-8"))
        recipes.append(recipe)
    return tuple(recipes)


def few_shots(
    tags: tuple[str, ...], recipes: tuple[dict[str, Any], ...], limit: int = 3
) -> list[dict[str, Any]]:
    """Select compact canonical examples sharing the most feature tags."""
    requested = set(tags)
    ranked = sorted(
        recipes,
        key=lambda recipe: (
            -len(requested & set(recipe["manifest"].get("features", ()))),
            recipe["manifest"]["id"],
        ),
    )
    result = []
    for recipe in ranked:
        if "figure" not in recipe:
            continue
        result.append(
            {
                "id": recipe["manifest"]["id"],
                "summary": recipe["manifest"]["summary"],
                "features": recipe["manifest"]["features"],
                "figure": recipe["figure"],
            }
        )
        if len(result) == limit:
            break
    return result
