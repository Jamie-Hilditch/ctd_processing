"""Tests for ctd_processing.cli.concatenate."""

from pathlib import Path

import pytest

from ctd_processing.cli.concatenate import resolve_binned_files


def _touch(path: Path) -> Path:
    path.write_text("", encoding="utf-8")
    return path


def test_resolve_binned_files_explicit_targets_happy_path(
    tmp_path: Path,
) -> None:
    """Explicit targets resolve in order under binned_directory."""
    _touch(tmp_path / "a.nc")
    _touch(tmp_path / "b.nc")

    result = resolve_binned_files(tmp_path, ["b", "a"], "nc")

    assert result == [
        (tmp_path / "b.nc").resolve(),
        (tmp_path / "a.nc").resolve(),
    ]


def test_resolve_binned_files_auto_discovers_sorted(tmp_path: Path) -> None:
    """Auto-discovery returns matching-extension files sorted by name."""
    _touch(tmp_path / "b.nc")
    _touch(tmp_path / "a.nc")

    result = resolve_binned_files(tmp_path, None, "nc")

    assert result == [
        (tmp_path / "a.nc").resolve(),
        (tmp_path / "b.nc").resolve(),
    ]


def test_resolve_binned_files_auto_discovery_ignores_wrong_extension(
    tmp_path: Path,
) -> None:
    """Auto-discovery only matches the given extension."""
    _touch(tmp_path / "a.nc")
    _touch(tmp_path / "b.txt")

    result = resolve_binned_files(tmp_path, None, "nc")

    assert result == [(tmp_path / "a.nc").resolve()]


def test_resolve_binned_files_auto_discovery_empty_raises(
    tmp_path: Path,
) -> None:
    """A binned_directory with no matching files raises ValueError."""
    _touch(tmp_path / "notes.txt")

    with pytest.raises(ValueError, match="No .nc files"):
        resolve_binned_files(tmp_path, None, "nc")


def test_resolve_binned_files_zarr_target_is_a_directory(
    tmp_path: Path,
) -> None:
    """A zarr target is a directory, not a plain file, and still resolves."""
    (tmp_path / "a.zarr").mkdir()

    result = resolve_binned_files(tmp_path, ["a"], "zarr")

    assert result == [(tmp_path / "a.zarr").resolve()]


def test_resolve_binned_files_rejects_dotdot_traversal(tmp_path: Path) -> None:
    """A ../ target that escapes binned_directory is rejected."""
    binned_dir = tmp_path / "binned"
    binned_dir.mkdir()
    _touch(tmp_path / "outside.nc")

    with pytest.raises(ValueError, match="outside binned_directory"):
        resolve_binned_files(binned_dir, ["../outside"], "nc")


def test_resolve_binned_files_rejects_absolute_target_escape(
    tmp_path: Path,
) -> None:
    """An absolute-path target outside binned_directory is rejected."""
    binned_dir = tmp_path / "binned"
    binned_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = _touch(outside_dir / "secret.nc")

    with pytest.raises(ValueError, match="outside binned_directory"):
        resolve_binned_files(
            binned_dir, [str(outside_file.with_suffix(""))], "nc"
        )


def test_resolve_binned_files_rejects_nonexistent_target(
    tmp_path: Path,
) -> None:
    """A target naming a file that doesn't exist is rejected."""
    with pytest.raises(ValueError, match="does not exist"):
        resolve_binned_files(tmp_path, ["missing"], "nc")


def test_resolve_binned_files_missing_binned_directory_raises(
    tmp_path: Path,
) -> None:
    """A nonexistent binned_directory raises ValueError."""
    missing = tmp_path / "does-not-exist"

    with pytest.raises(ValueError, match="does not exist"):
        resolve_binned_files(missing, None, "nc")


def test_resolve_binned_files_binned_directory_is_a_file_raises(
    tmp_path: Path,
) -> None:
    """A binned_directory pointing at a plain file raises ValueError."""
    file_path = _touch(tmp_path / "not_a_dir")

    with pytest.raises(ValueError, match="not a directory"):
        resolve_binned_files(file_path, None, "nc")
