from pathlib import PurePosixPath

import pytest

import mkdocs_hooks


def test_remote_readme_assets_are_rewritten_to_local_site_paths():
    mkdocs_hooks._GALLERY_FILES.clear()
    image_url = (
        "https://github.com/qnzhou/hakowan-gallery/blob/main/"
        "gallery/Flow/results/bust.webp?raw=true"
    )
    markdown = (
        f'[<img src="{image_url}" width="90%"/>]({image_url})\n'
        "[Interactive demo](https://qnzhou.github.io/hakowan-gallery/"
        "gallery/Flow/results/bust.html)\n"
        "[Inspection](artifacts/inspect.json) · [Python](flow.py)"
    )

    resolved = mkdocs_hooks._resolve_gallery_links(markdown, "Flow", "/hakowan")

    image_path = "/hakowan/gallery-assets/Flow/results/bust.webp"
    assert f'src="{image_path}"' in resolved
    assert f"]({image_path})" in resolved
    assert "(/hakowan/gallery-demos/Flow/results/bust.html)" in resolved
    assert "(/hakowan/gallery-assets/Flow/artifacts/inspect.json)" in resolved
    assert (
        f"(https://github.com/Hakowan/hakowan-gallery/blob/"
        f"{mkdocs_hooks._GALLERY_REVISION}/gallery/Flow/flow.py)"
    ) in resolved
    revision = mkdocs_hooks._GALLERY_REVISION
    assert mkdocs_hooks._GALLERY_FILES == {
        PurePosixPath("gallery-assets/Flow/results/bust.webp"): (
            "https://raw.githubusercontent.com/Hakowan/hakowan-gallery/"
            f"{revision}/gallery/Flow/results/bust.webp"
        ),
        PurePosixPath("gallery-demos/Flow/results/bust.html"): (
            "https://raw.githubusercontent.com/Hakowan/hakowan-gallery/"
            f"{revision}/gallery/Flow/results/bust.html"
        ),
        PurePosixPath("gallery-assets/Flow/artifacts/inspect.json"): (
            "https://raw.githubusercontent.com/Hakowan/hakowan-gallery/"
            f"{revision}/gallery/Flow/artifacts/inspect.json"
        ),
    }


def test_gallery_page_image_is_localized_and_keeps_display_fragment():
    mkdocs_hooks._GALLERY_FILES.clear()
    markdown = (
        "[![Sketch](https://github.com/Hakowan/hakowan-gallery/blob/main/"
        "gallery/Sketch/results/rough.webp?raw=true#only-dark)](examples/sketch.md)"
    )

    resolved = mkdocs_hooks._resolve_gallery_links(markdown, None, "/hakowan")

    assert resolved == (
        "[![Sketch](/hakowan/gallery-assets/Sketch/results/rough.webp#only-dark)]"
        "(examples/sketch.md)"
    )
    assert (
        PurePosixPath("gallery-assets/Sketch/results/rough.webp")
        in mkdocs_hooks._GALLERY_FILES
    )


def test_remaining_remote_snippets_are_pinned_to_gallery_revision():
    markdown = (
        '--8<-- "https://github.com/Hakowan/hakowan-gallery/raw/main/'
        'gallery/Flow/flow.py"'
    )

    resolved = mkdocs_hooks.on_page_markdown(
        markdown, {"site_url": "https://hakowan.github.io/hakowan/"}
    )

    assert f"/raw/{mkdocs_hooks._GALLERY_REVISION}/gallery/Flow/flow.py" in resolved
    assert "/raw/main/" not in resolved


def test_gallery_download_rejects_oversized_response(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, limit):
            return b"x" * limit

    monkeypatch.setattr(mkdocs_hooks, "urlopen", lambda *_args, **_kwargs: Response())

    with pytest.raises(RuntimeError, match="exceeds 8 bytes"):
        mkdocs_hooks._download("https://example.test/large", 8)


def test_gallery_download_is_atomically_replaced(monkeypatch, tmp_path):
    destination = tmp_path / "gallery" / "artifact.json"
    destination.parent.mkdir()
    destination.write_bytes(b"old")
    monkeypatch.setattr(mkdocs_hooks, "_download", lambda *_args: b"new")

    mkdocs_hooks._download_to("https://example.test/artifact", destination)

    assert destination.read_bytes() == b"new"
    assert list(destination.parent.iterdir()) == [destination]
