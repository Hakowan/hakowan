"""MkDocs hooks for content imported from the gallery repository."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlsplit
from urllib.request import urlopen

_GALLERY_README = re.compile(
    r'^--8<-- "(?P<url>https://github\.com/Hakowan/hakowan-gallery/raw/main/'
    r'gallery/(?P<recipe>[^/]+)/README\.md)"\s*$',
    re.MULTILINE,
)
_MARKDOWN_LINK = re.compile(r"(?<!!)\[([^]]+)]\(([^)]+)\)")

_GALLERY_DEMOS: dict[PurePosixPath, str] = {}


def _gallery_demo_path(target: str, recipe: str) -> PurePosixPath | None:
    parsed = urlsplit(target)
    if parsed.path.endswith(".html"):
        prefixes = (
            f"/qnzhou/hakowan-gallery/blob/main/gallery/{recipe}/",
            f"/Hakowan/hakowan-gallery/blob/main/gallery/{recipe}/",
            f"/qnzhou/hakowan-gallery/raw/main/gallery/{recipe}/",
            f"/Hakowan/hakowan-gallery/raw/main/gallery/{recipe}/",
            f"/hakowan-gallery/gallery/{recipe}/",
        )
        if not parsed.netloc:
            relative_path = parsed.path
        else:
            relative_path = next(
                (parsed.path.removeprefix(prefix) for prefix in prefixes if parsed.path.startswith(prefix)),
                "",
            )
        path = PurePosixPath(relative_path)
        if relative_path and not path.is_absolute() and ".." not in path.parts:
            return path
    return None


def _resolve_gallery_links(markdown: str, recipe: str, site_path: str) -> str:
    source_base_url = (
        "https://github.com/Hakowan/hakowan-gallery/blob/main/gallery/"
        f"{recipe}/"
    )

    def replace(match: re.Match[str]) -> str:
        label, target = match.groups()
        demo_path = _gallery_demo_path(target, recipe)
        if demo_path is not None:
            output_path = PurePosixPath("gallery-demos", recipe, demo_path)
            _GALLERY_DEMOS[output_path] = (
                "https://raw.githubusercontent.com/Hakowan/hakowan-gallery/"
                f"main/gallery/{recipe}/{demo_path}"
            )
            return f"[{label}]({site_path}/{output_path.as_posix()})"
        if target.startswith(("https://", "http://", "mailto:", "#", "/")):
            return match.group(0)
        return f"[{label}]({urljoin(source_base_url, target)})"

    return _MARKDOWN_LINK.sub(replace, markdown)


def on_config(config: dict[str, object]) -> dict[str, object]:
    """Reset discovered demos before each build."""
    _GALLERY_DEMOS.clear()
    return config


def on_page_markdown(markdown: str, config: dict[str, object], **_: object) -> str:
    """Inline remote gallery READMEs and resolve their repository-relative links."""
    site_path = urlsplit(str(config["site_url"])).path.rstrip("/")

    def replace(match: re.Match[str]) -> str:
        with urlopen(match.group("url"), timeout=30) as response:
            remote_markdown = response.read().decode("utf-8")
        return _resolve_gallery_links(remote_markdown, match.group("recipe"), site_path)

    return _GALLERY_README.sub(replace, markdown)


def on_post_build(config: dict[str, object], **_: object) -> None:
    """Copy referenced standalone gallery demos into the generated site."""
    site_dir = Path(str(config["site_dir"]))
    for output_path, source_url in _GALLERY_DEMOS.items():
        destination = site_dir.joinpath(*output_path.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(source_url, timeout=120) as response:
            destination.write_bytes(response.read())
