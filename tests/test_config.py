"""Tests for ctd_processing.config."""

import pytest
from pydantic import ValidationError

from ctd_processing.config import (
    Settings,
    load_settings,
    merge_overrides,
    parse_overrides,
)


def test_parse_overrides_flat_key() -> None:
    """A flat key=value pair should parse to a single-level dict."""
    assert parse_overrides(["name=1"]) == {"name": 1}


def test_parse_overrides_nested_dotted_key() -> None:
    """Dotted keys should build nested dictionaries."""
    assert parse_overrides(["section.key=1"]) == {"section": {"key": 1}}


def test_parse_overrides_deeply_nested_dotted_key() -> None:
    """Multiple dots should build multiple levels of nesting."""
    assert parse_overrides(["a.b.c=1"]) == {"a": {"b": {"c": 1}}}


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("1", 1),
        ("1.5", 1.5),
        ("true", True),
        ("false", False),
        ('"text"', "text"),
        ("[1, 2, 3]", [1, 2, 3]),
    ],
)
def test_parse_overrides_type_coercion(
    raw_value: str, expected: object
) -> None:
    """Values should be parsed using TOML syntax, matching the config file."""
    assert parse_overrides([f"key={raw_value}"]) == {"key": expected}


def test_parse_overrides_missing_equals_raises() -> None:
    """A pair with no '=' should raise ValueError."""
    with pytest.raises(ValueError, match="expected key=value"):
        parse_overrides(["not-a-pair"])


def test_parse_overrides_invalid_value_raises() -> None:
    """A value that isn't valid TOML syntax should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid TOML value"):
        parse_overrides(["key=not valid toml"])


def test_merge_overrides_deep_merges_nested_dicts() -> None:
    """Overrides should merge into nested dicts rather than replacing them."""
    base = {"section": {"a": 1, "b": 2}}
    merged = merge_overrides(base, ["section.b=3"])
    assert merged == {"section": {"a": 1, "b": 3}}


def test_merge_overrides_does_not_mutate_input() -> None:
    """merge_overrides should return a new dict, leaving the input untouched."""
    base = {"section": {"a": 1}}
    merge_overrides(base, ["section.a=2"])
    assert base == {"section": {"a": 1}}


def test_load_settings_defaults_without_file() -> None:
    """With no config path or overrides, defaults are used."""
    assert load_settings() == Settings()


def test_load_settings_reads_toml_file(tmp_path) -> None:
    """A comment-only config file should load without error."""
    config_path = tmp_path / "config.toml"
    config_path.write_text("# just a comment\n", encoding="utf-8")
    assert load_settings(config_path) == Settings()


def test_load_settings_missing_file_raises(tmp_path) -> None:
    """A nonexistent config path should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "missing.toml")


def test_load_settings_rejects_unknown_key_from_file(tmp_path) -> None:
    """An unknown key in the config file should fail validation."""
    config_path = tmp_path / "config.toml"
    config_path.write_text("not_a_real_option = 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(config_path)


def test_load_settings_rejects_unknown_key_from_overrides() -> None:
    """An unknown key from --set should fail validation."""
    with pytest.raises(ValidationError):
        load_settings(set_=["not_a_real_option=1"])


def test_load_settings_propagates_malformed_override() -> None:
    """A malformed --set pair raises ValueError, not a validation error."""
    with pytest.raises(ValueError, match="expected key=value"):
        load_settings(set_=["not-a-pair"])
