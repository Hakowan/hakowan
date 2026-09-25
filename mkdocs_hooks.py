"""MkDocs hooks for content imported from the gallery repository."""

from __future__ import annotations

import re
import tempfile
import urllib.error
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlsplit
from urllib.request import urlopen

_GALLERY_README = re.compile(
    r'^--8<-- "(?P<url>https://github\.com/Hakowan/hakowan-gallery/raw/main/'
    r'gallery/(?P<recipe>[^/]+)/README\.md)"\s*$',
    re.MULTILINE,
)
_GALLERY_REVISION = "6e250d5907d22f491d0c13ec6f469489e74c679c"
_MAX_GALLERY_FILE_BYTES = 128 * 1024 * 1024
_MAX_README_BYTES = 2 * 1024 * 1024
_RAW_MAIN_URL = re.compile(
    r"https://github\.com/(?:Hakowan|qnzhou)/hakowan-gallery/raw/main/"
)
_MARKDOWN_LINK = re.compile(r"(?<!!)\[([^]]+)]\(([^)]+)\)")
_MARKDOWN_IMAGE = re.compile(r"!\[([^]]*)]\(([^)]+)\)")
_HTML_IMAGE = re.compile(
    r'(?P<prefix><img\b[^>]*?\bsrc=["\'])(?P<target>[^"\']+)(?P<suffix>["\'])',
    re.IGNORECASE,
)
_LOCAL_SUFFIXES = frozenset(
    {".html", ".json", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".avif"}
)

_GALLERY_FILES: dict[PurePosixPath, str] = {}


def _gallery_asset_path(
    target: str, recipe: str | None
) -> tuple[str, PurePosixPath] | None:
    parsed = urlsplit(target)
    relative_path = ""
    resolved_recipe = recipe
    if not parsed.netloc and recipe is not None:
        relative_path = parsed.path
    elif parsed.netloc:
        if parsed.netloc.lower() not in {
            "github.com",
            "raw.githubusercontent.com",
            "qnzhou.github.io",
            "hakowan.github.io",
        }:
            return None
        prefixes = (
            "/qnzhou/hakowan-gallery/blob/main/gallery/",
            "/Hakowan/hakowan-gallery/blob/main/gallery/",
            "/qnzhou/hakowan-gallery/raw/main/gallery/",
            "/Hakowan/hakowan-gallery/raw/main/gallery/",
            "/qnzhou/hakowan-gallery/main/gallery/",
            "/Hakowan/hakowan-gallery/main/gallery/",
            "/hakowan-gallery/gallery/",
        )
        gallery_path = next(
            (
                parsed.path.removeprefix(prefix)
                for prefix in prefixes
                if parsed.path.startswith(prefix)
            ),
            "",
        )
        parts = PurePosixPath(gallery_path).parts
        if len(parts) >= 2:
            resolved_recipe, relative_path = parts[0], "/".join(parts[1:])

    path = PurePosixPath(relative_path)
    recipe_is_safe = (
        resolved_recipe is not None
        and resolved_recipe not in {"", ".", ".."}
        and "/" not in resolved_recipe
        and "\\" not in resolved_recipe
    )
    if (
        not recipe_is_safe
        or not relative_path
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in relative_path
        or path.suffix.lower() not in _LOCAL_SUFFIXES
    ):
        return None
    return resolved_recipe, path


def _local_gallery_target(
    target: str, recipe: str | None, site_path: str
) -> str | None:
    asset = _gallery_asset_path(target, recipe)
    if asset is None:
        return None
    resolved_recipe, asset_path = asset
    directory = (
        "gallery-demos" if asset_path.suffix.lower() == ".html" else "gallery-assets"
    )
    output_path = PurePosixPath(directory, resolved_recipe, asset_path)
    _GALLERY_FILES[output_path] = (
        "https://raw.githubusercontent.com/Hakowan/hakowan-gallery/"
        f"{_GALLERY_REVISION}/gallery/{resolved_recipe}/{asset_path.as_posix()}"
    )
    fragment = urlsplit(target).fragment
    suffix = f"#{fragment}" if fragment else ""
    return f"{site_path}/{output_path.as_posix()}{suffix}"


def _resolve_gallery_links(markdown: str, recipe: str | None, site_path: str) -> str:
    source_base_url = (
        "https://github.com/Hakowan/hakowan-gallery/blob/"
        f"{_GALLERY_REVISION}/gallery/{recipe}/"
        if recipe is not None
        else ""
    )

    def rewrite_target(target: str) -> str:
        local = _local_gallery_target(target, recipe, site_path)
        if local is not None:
            return local
        if target.startswith(("https://", "http://", "mailto:", "#", "/")):
            return target
        return urljoin(source_base_url, target) if source_base_url else target

    markdown = _HTML_IMAGE.sub(
        lambda match: (
            f"{match.group('prefix')}{rewrite_target(match.group('target'))}"
            f"{match.group('suffix')}"
        ),
        markdown,
    )
    markdown = _MARKDOWN_IMAGE.sub(
        lambda match: f"![{match.group(1)}]({rewrite_target(match.group(2))})",
        markdown,
    )
    return _MARKDOWN_LINK.sub(
        lambda match: f"[{match.group(1)}]({rewrite_target(match.group(2))})",
        markdown,
    )


def _download(url: str, max_bytes: int) -> bytes:
    """Download one pinned gallery file with a strict size bound."""
    try:
        with urlopen(url, timeout=120) as response:
            payload = response.read(max_bytes + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Cannot download gallery file {url}: {exc}") from exc
    if len(payload) > max_bytes:
        raise RuntimeError(f"Gallery file exceeds {max_bytes} bytes: {url}")
    return payload


def _download_to(url: str, destination: Path) -> None:
    """Atomically write one bounded gallery download to the built site."""
    payload = _download(url, _MAX_GALLERY_FILE_BYTES)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, delete=False
        ) as output:
            output.write(payload)
            temporary = Path(output.name)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def on_config(config: dict[str, object]) -> dict[str, object]:
    """Reset discovered gallery files before each build."""
    _GALLERY_FILES.clear()
    return config


def on_page_markdown(markdown: str, config: dict[str, object], **_: object) -> str:
    """Inline remote gallery READMEs and resolve their repository-relative links."""
    site_path = urlsplit(str(config["site_url"])).path.rstrip("/")

    def replace(match: re.Match[str]) -> str:
        source_url = (
            "https://raw.githubusercontent.com/Hakowan/hakowan-gallery/"
            f"{_GALLERY_REVISION}/gallery/{match.group('recipe')}/README.md"
        )
        remote_markdown = _download(source_url, _MAX_README_BYTES).decode("utf-8")
        return _resolve_gallery_links(remote_markdown, match.group("recipe"), site_path)

    markdown = _GALLERY_README.sub(replace, markdown)
    pinned_raw = f"https://github.com/Hakowan/hakowan-gallery/raw/{_GALLERY_REVISION}/"
    markdown = _RAW_MAIN_URL.sub(pinned_raw, markdown)
    return _resolve_gallery_links(markdown, None, site_path)


def on_post_build(config: dict[str, object], **_: object) -> None:
    """Copy referenced gallery demos, images, and JSON into the generated site."""
    site_dir = Path(str(config["site_dir"]))
    for output_path, source_url in _GALLERY_FILES.items():
        destination = site_dir.joinpath(*output_path.parts)
        _download_to(source_url, destination)
