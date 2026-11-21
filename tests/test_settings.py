from __future__ import annotations

from pathlib import Path

import pytest

from calico.errors import ConfigError, NotInCollectionError
from calico.settings import CollectionConfig, Settings, load_settings, save_settings


def collection(
    name: str = "explore", root: str = "/archive/explore", **kw: object
) -> CollectionConfig:
    return CollectionConfig.model_validate({"name": name, "root": root, **kw})


def test_missing_settings_file_gives_defaults_rooted_at_home(calico_home: Path) -> None:
    settings = load_settings()
    assert settings.home == calico_home.resolve()
    assert settings.collections == ()
    assert settings.catalog_path == calico_home.resolve() / "catalog.sqlite"


def test_settings_round_trip_through_toml(calico_home: Path) -> None:
    original = Settings(
        home=calico_home, embedder="tiny", describe_embedder="tiny", view_embedder="tiny"
    ).with_collection(collection(location_pattern=r"^(?P<location>.+?) - "))
    save_settings(original)
    assert load_settings(calico_home) == original.model_copy(update={"home": calico_home.resolve()})


def test_location_rules_round_trip_through_toml(calico_home: Path) -> None:
    original = Settings(
        home=calico_home,
        location_aliases={"Cobrett Building": "Corbett Building"},
    ).with_collection(collection(location_ignore=[r"\([^)]*\)"]))
    save_settings(original)
    assert load_settings(calico_home) == original.model_copy(update={"home": calico_home.resolve()})


def test_location_aliases_may_not_chain(calico_home: Path) -> None:
    with pytest.raises(ValueError, match=r"Musuem.*Museum"):
        Settings(home=calico_home, location_aliases={"Musuem": "Museum", "museum": "Gallery"})


def test_only_chosen_settings_are_saved_so_better_defaults_still_arrive(
    calico_home: Path,
) -> None:
    settings = load_settings(calico_home).with_collection(collection())
    save_settings(settings)
    saved = (calico_home / "calico.toml").read_text(encoding="utf-8")
    assert "criteria_scorer" not in saved
    assert "embedder" not in saved
    assert 'name = "explore"' in saved
    chosen = load_settings(calico_home).model_copy(update={"embedder": "tiny"})
    save_settings(chosen)
    assert 'embedder = "tiny"' in (calico_home / "calico.toml").read_text(encoding="utf-8")
    assert load_settings(calico_home).embedder == "tiny"


def test_invalid_toml_is_a_config_error(calico_home: Path) -> None:
    calico_home.mkdir(parents=True)
    (calico_home / "calico.toml").write_text("embedder = ", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_settings()


def test_unknown_setting_is_a_config_error(calico_home: Path) -> None:
    calico_home.mkdir(parents=True)
    (calico_home / "calico.toml").write_text('embeder = "tiny"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="embeder"):
        load_settings()


def test_collection_root_must_be_absolute() -> None:
    with pytest.raises(ValueError, match="absolute"):
        collection(root="relative/path")


def test_location_pattern_needs_a_location_group() -> None:
    with pytest.raises(ValueError, match="named group"):
        collection(location_pattern=r"^(.+) - ")


def test_location_pattern_must_compile() -> None:
    with pytest.raises(ValueError, match="valid regex"):
        collection(location_pattern=r"(?P<location>")


def test_duplicate_collection_names_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Settings(home=tmp_path, collections=(collection(), collection(root="/elsewhere")))


def test_collection_for_picks_the_deepest_containing_root(tmp_path: Path) -> None:
    outer = collection("outer", "/archive")
    inner = collection("inner", "/archive/explore")
    settings = Settings(home=tmp_path, collections=(outer, inner))
    assert settings.collection_for(Path("/archive/explore/site/a.jpg")).name == "inner"
    assert settings.collection_for(Path("/archive/travel/a.jpg")).name == "outer"


def test_collection_for_outside_every_root_says_how_to_fix_it(tmp_path: Path) -> None:
    settings = Settings(home=tmp_path, collections=(collection(),))
    with pytest.raises(NotInCollectionError, match="calico collections add"):
        settings.collection_for(Path("/somewhere/else.jpg"))


def test_with_collection_replaces_by_name(tmp_path: Path) -> None:
    settings = Settings(home=tmp_path).with_collection(collection(root="/a"))
    settings = settings.with_collection(collection(root="/b"))
    assert [c.root for c in settings.collections] == [Path("/b")]


def test_standard_extractors_include_the_configured_embedders(tmp_path: Path) -> None:
    both = Settings(home=tmp_path, embedder="tiny")
    assert both.feature_extractors[-3:] == ("tiny", "siglip", "dinov2")
    same = Settings(home=tmp_path, embedder="tiny", describe_embedder="tiny", view_embedder="tiny")
    assert same.feature_extractors.count("tiny") == 1


def test_same_view_is_matched_by_dinov2_unless_chosen_otherwise(tmp_path: Path) -> None:
    """DINOv2 told same views from look-alikes better than CLIP (see calico eval same-view)."""
    assert Settings(home=tmp_path).view_embedder == "dinov2"
    chosen = Settings(home=tmp_path, embedder="tiny", view_embedder="palette")
    assert "palette" in chosen.feature_extractors


def test_describe_it_is_read_by_siglip_unless_chosen_otherwise(tmp_path: Path) -> None:
    """SigLIP ranked labelled searches better than CLIP and can tell a clear match."""
    assert Settings(home=tmp_path).describe_embedder == "siglip"


def test_collection_root_is_resolved_through_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    assert collection(root=str(link)).root == real.resolve()


def test_collection_root_that_does_not_exist_still_validates() -> None:
    assert collection(root="/not/mounted/yet").root == Path("/not/mounted/yet")


def test_collection_root_is_nfc_normalised() -> None:
    assert collection(root="/archive/Cafe\u0301").root == Path("/archive/Caf\u00e9")


def test_two_aliases_for_one_place_are_refused(calico_home: Path) -> None:
    """They differ only in capitals, so one would silently win."""
    with pytest.raises(ValueError, match=r"mill.*MILL"):
        Settings(home=calico_home, location_aliases={"mill": "Old Mill", "MILL": "New Mill"})
