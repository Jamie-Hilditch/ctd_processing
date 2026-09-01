"""Tests for ctd_processing.cli.bin."""

from pathlib import Path

import pytest

from ctd_processing.cli.bin import resolve_deployment_stems


def test_resolve_deployment_stems_explicit_targets_happy_path(
    tmp_path: Path,
) -> None:
    """Explicit targets are returned in the given order, unresolved."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    result = resolve_deployment_stems(tmp_path, ["b", "a"])

    assert result == ["b", "a"]


def test_resolve_deployment_stems_auto_discovers_sorted(
    tmp_path: Path,
) -> None:
    """Auto-discovery returns top-level subdirectory names sorted by name."""
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()

    result = resolve_deployment_stems(tmp_path, None)

    assert result == ["a", "b"]


def test_resolve_deployment_stems_auto_discovery_ignores_files(
    tmp_path: Path,
) -> None:
    """Auto-discovery only considers subdirectories, not plain files."""
    (tmp_path / "a").mkdir()
    (tmp_path / "notes.txt").write_text("", encoding="utf-8")

    result = resolve_deployment_stems(tmp_path, None)

    assert result == ["a"]


def test_resolve_deployment_stems_auto_discovery_empty_raises(
    tmp_path: Path,
) -> None:
    """A profiles_directory with no subdirectories raises ValueError."""
    (tmp_path / "notes.txt").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="No deployment subdirectories"):
        resolve_deployment_stems(tmp_path, None)


def test_resolve_deployment_stems_rejects_dotdot_traversal(
    tmp_path: Path,
) -> None:
    """A ../ target that escapes profiles_directory is rejected."""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (tmp_path / "outside").mkdir()

    with pytest.raises(ValueError, match="outside profiles_directory"):
        resolve_deployment_stems(profiles_dir, ["../outside"])


def test_resolve_deployment_stems_rejects_absolute_target_escape(
    tmp_path: Path,
) -> None:
    """An absolute-path target outside profiles_directory is rejected."""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    with pytest.raises(ValueError, match="outside profiles_directory"):
        resolve_deployment_stems(profiles_dir, [str(outside_dir)])


def test_resolve_deployment_stems_rejects_nonexistent_target(
    tmp_path: Path,
) -> None:
    """A target naming a subdirectory that doesn't exist is rejected."""
    with pytest.raises(ValueError, match="does not exist"):
        resolve_deployment_stems(tmp_path, ["missing"])


def test_resolve_deployment_stems_rejects_target_that_is_a_file(
    tmp_path: Path,
) -> None:
    """A target naming a file rather than a subdirectory is rejected."""
    (tmp_path / "deployment").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="does not exist"):
        resolve_deployment_stems(tmp_path, ["deployment"])


def test_resolve_deployment_stems_missing_profiles_directory_raises(
    tmp_path: Path,
) -> None:
    """A nonexistent profiles_directory raises ValueError."""
    missing = tmp_path / "does-not-exist"

    with pytest.raises(ValueError, match="does not exist"):
        resolve_deployment_stems(missing, None)


def test_resolve_deployment_stems_profiles_directory_is_a_file_raises(
    tmp_path: Path,
) -> None:
    """A profiles_directory pointing at a plain file raises ValueError."""
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        resolve_deployment_stems(file_path, None)
