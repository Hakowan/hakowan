"""MkDocs hooks for content imported from the gallery repository."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlsplit

_GALLERY_README = re.compile(
    r'^--8<-- "(?P<url>https://github\.com/Hakowan/hakowan-gallery/raw/main/'
    r'gallery/(?P<recipe>[^/]+)/README\.md)"\s*$',
    re.MULTILINE,
)

_GALLERY_SOURCE_SNIPPET = re.compile(
    r'^--8<-- "https://github\.com/(?:Hakowan|qnzhou)/hakowan-gallery/raw/main/'
    r'gallery/(?P<path>[^"]+)"[ \t]*$',
    re.MULTILINE,
)
_GALLERY_REVISION = "2071eecf977bcdf988bc73bdad9e741a812a1b4c"
_GALLERY_REPOSITORY_URL = "https://github.com/Hakowan/hakowan-gallery.git"
_GALLERY_CHECKOUT_PATH = Path(tempfile.gettempdir()) / "hakowan-mkdocs-gallery"
_GALLERY_CHECKOUT_REVISION: str | None = None
_MAX_GALLERY_FILE_BYTES = 128 * 1024 * 1024
_MAX_README_BYTES = 2 * 1024 * 1024
_MAX_SOURCE_BYTES = 2 * 1024 * 1024
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

_GALLERY_FILES: dict[PurePosixPath, PurePosixPath] = {}


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
    _GALLERY_FILES[output_path] = PurePosixPath(resolved_recipe, *asset_path.parts)
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


def _ensure_gallery_checkout() -> Path:
    """Reuse a shallow Dulwich checkout of the pinned gallery revision."""
    global _GALLERY_CHECKOUT_REVISION

    try:
        from dulwich import porcelain
        from dulwich.client import get_transport_and_path
        from dulwich.repo import Repo
    except ImportError as exc:
        raise RuntimeError(
            "Gallery docs require Dulwich; install the docs extra with "
            "`pip install '.[docs]'`."
        ) from exc

    checkout = _GALLERY_CHECKOUT_PATH
    git_dir = checkout / ".git"
    if _GALLERY_CHECKOUT_REVISION == _GALLERY_REVISION and git_dir.is_dir():
        return checkout
    if checkout.exists() and not git_dir.is_dir() and any(checkout.iterdir()):
        raise RuntimeError(
            f"Gallery checkout path exists but is not a Git repository: {checkout}"
        )
    checkout.parent.mkdir(parents=True, exist_ok=True)
    checkout.mkdir(exist_ok=True)

    try:
        repo = (
            Repo(str(checkout)) if git_dir.is_dir() else porcelain.init(str(checkout))
        )
        revision = _GALLERY_REVISION.encode("ascii")
        try:
            current = repo.head()
        except KeyError:
            current = None

        if current != revision:
            if revision not in repo.object_store:
                client, remote_path = get_transport_and_path(_GALLERY_REPOSITORY_URL)
                client.fetch(
                    remote_path,
                    repo,
                    determine_wants=lambda _refs, depth=None: [revision],
                    progress=lambda _data: None,
                    depth=1,
                )
            if revision not in repo.object_store:
                raise RuntimeError(
                    f"Pinned gallery revision {_GALLERY_REVISION} was not fetched"
                )
            porcelain.checkout(repo, revision, force=True)
            current = repo.head()
            if current != revision:
                raise RuntimeError(
                    f"Gallery checkout has revision {current!r}, "
                    f"expected {_GALLERY_REVISION}"
                )
    except Exception as exc:
        raise RuntimeError(
            f"Cannot prepare gallery checkout at revision {_GALLERY_REVISION}: {exc}"
        ) from exc

    _GALLERY_CHECKOUT_REVISION = _GALLERY_REVISION
    return checkout


def _read_gallery_file(path: Path, max_bytes: int) -> bytes:
    try:
        with path.open("rb") as source:
            payload = source.read(max_bytes + 1)
    except OSError as exc:
        raise RuntimeError(f"Cannot read gallery file {path}: {exc}") from exc
    if len(payload) > max_bytes:
        raise RuntimeError(f"Gallery file exceeds {max_bytes} bytes: {path}")
    return payload


def _copy_gallery_file(source: Path, destination: Path, max_bytes: int) -> None:
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise RuntimeError(f"Cannot read gallery file {source}: {exc}") from exc
    if size > max_bytes:
        raise RuntimeError(f"Gallery file exceeds {max_bytes} bytes: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with (
            source.open("rb") as input_file,
            tempfile.NamedTemporaryFile(
                dir=destination.parent, delete=False
            ) as output_file,
        ):
            temporary = Path(output_file.name)
            copied = 0
            while chunk := input_file.read(1024 * 1024):
                copied += len(chunk)
                if copied > max_bytes:
                    raise RuntimeError(
                        f"Gallery file exceeds {max_bytes} bytes: {source}"
                    )
                output_file.write(chunk)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def on_config(config: dict[str, object]) -> dict[str, object]:
    """Reset discovered gallery files before each build."""
    _GALLERY_FILES.clear()
    return config


def on_page_markdown(markdown: str, config: dict[str, object], **_: object) -> str:
    """Inline gallery content from the pinned local checkout."""
    site_path = urlsplit(str(config["site_url"])).path.rstrip("/")

    def replace_readme(match: re.Match[str]) -> str:
        recipe = match.group("recipe")
        if recipe in {"", ".", ".."} or "\\" in recipe:
            raise RuntimeError(f"Unsafe gallery recipe name: {recipe!r}")
        checkout = _ensure_gallery_checkout()
        source = checkout / "gallery" / recipe / "README.md"
        readme = _read_gallery_file(source, _MAX_README_BYTES).decode("utf-8")
        return _resolve_gallery_links(readme, recipe, site_path)

    markdown = _GALLERY_README.sub(replace_readme, markdown)

    def replace_source(match: re.Match[str]) -> str:
        relative_path = PurePosixPath(match.group("path"))
        if (
            relative_path.is_absolute()
            or len(relative_path.parts) < 2
            or ".." in relative_path.parts
            or "\\" in match.group("path")
            or relative_path.suffix.lower() != ".py"
        ):
            raise RuntimeError(f"Unsafe gallery source path: {match.group('path')!r}")
        checkout = _ensure_gallery_checkout()
        source = checkout.joinpath("gallery", *relative_path.parts)
        return _read_gallery_file(source, _MAX_SOURCE_BYTES).decode("utf-8")

    markdown = _GALLERY_SOURCE_SNIPPET.sub(replace_source, markdown)
    pinned_raw = f"https://github.com/Hakowan/hakowan-gallery/raw/{_GALLERY_REVISION}/"
    markdown = _RAW_MAIN_URL.sub(pinned_raw, markdown)
    return _resolve_gallery_links(markdown, None, site_path)


def on_post_build(config: dict[str, object], **_: object) -> None:
    """Copy referenced gallery files from the pinned checkout into the site."""
    if not _GALLERY_FILES:
        return
    checkout = _ensure_gallery_checkout()
    site_dir = Path(str(config["site_dir"]))
    for output_path, source_path in _GALLERY_FILES.items():
        source = checkout.joinpath("gallery", *source_path.parts)
        destination = site_dir.joinpath(*output_path.parts)
        _copy_gallery_file(source, destination, _MAX_GALLERY_FILE_BYTES)
