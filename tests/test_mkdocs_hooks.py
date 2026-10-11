import io
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
    assert mkdocs_hooks._GALLERY_FILES == {
        PurePosixPath("gallery-assets/Flow/results/bust.webp"): PurePosixPath(
            "Flow", "results", "bust.webp"
        ),
        PurePosixPath("gallery-demos/Flow/results/bust.html"): PurePosixPath(
            "Flow", "results", "bust.html"
        ),
        PurePosixPath("gallery-assets/Flow/artifacts/inspect.json"): PurePosixPath(
            "Flow", "artifacts", "inspect.json"
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


@pytest.mark.parametrize("recipe", ["..", ".", "..\\outside", "nested/recipe"])
def test_gallery_asset_rejects_unsafe_recipe(recipe):
    mkdocs_hooks._GALLERY_FILES.clear()

    assert mkdocs_hooks._gallery_asset_path("results/image.webp", recipe) is None
    assert (
        mkdocs_hooks._local_gallery_target("results/image.webp", recipe, "/hakowan")
        is None
    )
    assert mkdocs_hooks._GALLERY_FILES == {}


def test_gallery_asset_rejects_windows_style_path_traversal():
    assert mkdocs_hooks._gallery_asset_path("results\\..\\outside.webp", "Flow") is None


def test_gallery_readme_snippets_and_assets_use_local_checkout(monkeypatch, tmp_path):
    mkdocs_hooks._GALLERY_FILES.clear()
    checkout = tmp_path / "checkout"
    recipe = checkout / "gallery" / "Flow"
    (recipe / "results").mkdir(parents=True)
    (recipe / "README.md").write_text(
        "# Flow\n\n[Preview](results/preview.webp) · [Python](flow.py)\n",
        encoding="utf-8",
    )
    (recipe / "flow.py").write_text(
        'def render_flow():\n    return "local source"\n', encoding="utf-8"
    )
    (recipe / "results" / "preview.webp").write_bytes(b"local preview")
    monkeypatch.setattr(mkdocs_hooks, "_ensure_gallery_checkout", lambda: checkout)

    markdown = (
        '--8<-- "https://github.com/Hakowan/hakowan-gallery/raw/main/'
        'gallery/Flow/README.md"\n\n'
        "```py\n"
        '--8<-- "https://github.com/Hakowan/hakowan-gallery/raw/main/'
        'gallery/Flow/flow.py"\n'
        "```"
    )
    resolved = mkdocs_hooks.on_page_markdown(
        markdown, {"site_url": "https://hakowan.github.io/hakowan/"}
    )

    assert "# Flow" in resolved
    assert "def render_flow():" in resolved
    assert "--8<--" not in resolved
    assert "/hakowan/gallery-assets/Flow/results/preview.webp" in resolved
    assert (
        f"https://github.com/Hakowan/hakowan-gallery/blob/"
        f"{mkdocs_hooks._GALLERY_REVISION}/gallery/Flow/flow.py"
    ) in resolved

    destination = (
        tmp_path / "site" / "gallery-assets" / "Flow" / "results" / "preview.webp"
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old preview")
    mkdocs_hooks.on_post_build({"site_dir": str(tmp_path / "site")})
    assert destination.read_bytes() == b"local preview"
    assert list(destination.parent.iterdir()) == [destination]


def test_gallery_checkout_is_shallow_and_refreshes_changed_revision(
    monkeypatch, tmp_path
):
    pytest.importorskip("dulwich")
    from dulwich import client as dulwich_client
    from dulwich import porcelain
    from dulwich.repo import Repo

    source_path = tmp_path / "source"
    source_path.mkdir()
    source_repo = porcelain.init(str(source_path))

    def commit_readme(text, message):
        (source_path / "README.md").write_text(text, encoding="utf-8")
        porcelain.add(source_repo, ["README.md"])
        return porcelain.commit(
            source_repo,
            message=message.encode("ascii"),
            author=b"Gallery Test <gallery@example.test>",
            committer=b"Gallery Test <gallery@example.test>",
        )

    first_revision = commit_readme("revision one\n", "first revision").decode("ascii")
    remote_path = tmp_path / "gallery.git"
    porcelain.init(str(remote_path), bare=True)
    porcelain.push(
        source_repo,
        str(remote_path),
        refspecs="HEAD:refs/heads/main",
        outstream=io.BytesIO(),
        errstream=io.BytesIO(),
    )

    checkout = tmp_path / "cached-gallery"
    monkeypatch.setattr(mkdocs_hooks, "_GALLERY_CHECKOUT_PATH", checkout)
    monkeypatch.setattr(mkdocs_hooks, "_GALLERY_REPOSITORY_URL", str(remote_path))
    monkeypatch.setattr(mkdocs_hooks, "_GALLERY_REVISION", first_revision)
    monkeypatch.setattr(mkdocs_hooks, "_GALLERY_CHECKOUT_REVISION", None)

    assert mkdocs_hooks._ensure_gallery_checkout() == checkout
    repo = Repo(str(checkout))
    assert repo.head().decode("ascii") == first_revision
    repo.close()
    assert (checkout / ".git" / "shallow").is_file()

    get_transport_and_path = dulwich_client.get_transport_and_path
    monkeypatch.setattr(
        dulwich_client,
        "get_transport_and_path",
        lambda *_args, **_kwargs: pytest.fail("unexpected fetch for cached revision"),
    )
    assert mkdocs_hooks._ensure_gallery_checkout() == checkout
    monkeypatch.setattr(
        dulwich_client, "get_transport_and_path", get_transport_and_path
    )

    second_revision = commit_readme("revision two\n", "second revision").decode("ascii")
    porcelain.push(
        source_repo,
        str(remote_path),
        refspecs="HEAD:refs/heads/main",
        outstream=io.BytesIO(),
        errstream=io.BytesIO(),
    )
    monkeypatch.setattr(mkdocs_hooks, "_GALLERY_REVISION", second_revision)

    assert mkdocs_hooks._ensure_gallery_checkout() == checkout
    repo = Repo(str(checkout))
    assert repo.head().decode("ascii") == second_revision
    repo.close()
    assert (checkout / ".git" / "shallow").is_file()
    assert (checkout / "README.md").read_text(encoding="utf-8") == "revision two\n"


def test_remaining_raw_gallery_links_are_pinned_to_revision():
    markdown = (
        "[Source](https://github.com/Hakowan/hakowan-gallery/raw/main/"
        "gallery/Flow/flow.py)"
    )

    resolved = mkdocs_hooks.on_page_markdown(
        markdown, {"site_url": "https://hakowan.github.io/hakowan/"}
    )

    assert f"/raw/{mkdocs_hooks._GALLERY_REVISION}/gallery/Flow/flow.py" in resolved
    assert "/raw/main/" not in resolved


def test_gallery_file_read_rejects_oversized_content(tmp_path):
    source = tmp_path / "large.md"
    source.write_bytes(b"x" * 9)

    with pytest.raises(RuntimeError, match="exceeds 8 bytes"):
        mkdocs_hooks._read_gallery_file(source, 8)
