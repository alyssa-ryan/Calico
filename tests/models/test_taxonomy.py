from __future__ import annotations

from pathlib import Path

import pytest

from calico.errors import ConfigError
from calico.models.taxonomy import discover_taxonomies, load_taxonomy

BUILTIN = Path(__file__).parents[2] / "src" / "calico" / "models" / "taxonomies"


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_builtins_load() -> None:
    found = discover_taxonomies([BUILTIN])
    assert {"aesthetic", "mood", "place_type", "shot_kind", "surface"} <= set(found)
    for t in found.values():
        assert len(t.labels) >= 2
        assert len(t.prompts) == len(t.labels)


def test_the_general_aesthetic_tells_documents_from_weak_photos() -> None:
    aesthetic = discover_taxonomies([BUILTIN])["aesthetic"]
    assert aesthetic.labels == ("strong", "weak", "document")
    assert aesthetic.version == "4"


def test_the_general_aesthetic_names_what_it_missed_in_the_owners_look() -> None:
    """S5 (7 November 2026): General ranked reference shots of objects and phone snapshots
    above the owner's portfolio-worthy photos, and architectural geometry below them."""
    strong, weak, document = discover_taxonomies([BUILTIN])["aesthetic"].prompts
    assert "a photograph of architecture with strong geometry and symmetry" in strong
    assert "a casual smartphone snapshot" in weak
    assert "a reference photo of an object or a museum display case" in document


def test_the_general_aesthetic_counts_a_striking_portrait_as_strong() -> None:
    """Without it, photos with a person in them ranked below the rest (27 October 2026)."""
    strong = discover_taxonomies([BUILTIN])["aesthetic"].prompts[0]
    assert "a striking portrait of a person, beautifully lit" in strong


def test_scale_runs_from_farthest_to_nearest() -> None:
    """Its order is what makes a photo's expected label position a scale number."""
    scale = discover_taxonomies([BUILTIN])["scale"]
    assert scale.labels == ("aerial", "wide", "medium", "close", "macro")


def test_subject_says_whether_a_person_or_animal_is_in_the_photo() -> None:
    """Telling stories by roles reads ``living`` by name."""
    assert discover_taxonomies([BUILTIN])["subject"].labels == ("living", "none")


def test_load_preserves_order(tmp_path: Path) -> None:
    p = write(
        tmp_path / "x.toml",
        'description="d"\nversion="2"\n[labels.zed]\nprompts=["a"]\n'
        '[labels."big:small-one"]\nprompts=["b","c"]\n',
    )
    t = load_taxonomy(p)
    assert (t.name, t.version, t.description) == ("x", "2", "d")
    assert t.labels == ("zed", "big:small-one")
    assert t.prompts == (("a",), ("b", "c"))


@pytest.mark.parametrize(
    "body",
    [
        "[labels.a]\nprompts=['x']\n",  # one label
        "[labels.a]\nprompts=[]\n[labels.b]\nprompts=['x']\n",
        "[labels.a]\nprompts=['  ']\n[labels.b]\nprompts=['x']\n",
        "[labels.'Bad Id']\nprompts=['x']\n[labels.b]\nprompts=['x']\n",
        "[labels.a]\nprompts=['x']\n[labels.b]\nnope=1\n",
        '[labels."a\\n"]\nprompts=["x"]\n[labels.b]\nprompts=["x"]\n',  # trailing newline
        "[labels.a\n",
        "labels = 3\n",
    ],
)
def test_invalid(tmp_path: Path, body: str) -> None:
    p = write(tmp_path / "bad.toml", 'description="d"\nversion="1"\n' + body)
    with pytest.raises(ConfigError, match=r"bad\.toml"):
        load_taxonomy(p)


def test_discover_override_and_missing(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    body = '[labels.p]\nprompts=["x"]\n[labels.q]\nprompts=["y"]\n'
    write(a / "t.toml", 'description="first"\nversion="1"\n' + body)
    write(b / "t.toml", 'description="second"\nversion="1"\n' + body)
    found = discover_taxonomies([a, tmp_path / "missing", b])
    assert found["t"].description == "second"
