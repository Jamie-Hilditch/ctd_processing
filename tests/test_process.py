"""Tests for ctd_processing.cli.process."""

from pathlib import Path

import pytest

from ctd_processing.cli.process import resolve_deployment_files


def _touch(path: Path) -> Path:
    path.write_text("", encoding="utf-8")
    return path


def test_resolve_deployment_files_explicit_targets_happy_path(
    tmp_path: Path,
) -> None:
    """Explicit targets are resolved in the given order under rsk_directory."""
    _touch(tmp_path / "a.rsk")
    _touch(tmp_path / "b.rsk")

    result = resolve_deployment_files(tmp_path, ["b.rsk", "a.rsk"])

    assert result == [
        (tmp_path / "b.rsk").resolve(),
        (tmp_path / "a.rsk").resolve(),
    ]


def test_resolve_deployment_files_auto_discovers_sorted(
    tmp_path: Path,
) -> None:
    """Auto-discovery returns top-level .rsk files sorted by name."""
    _touch(tmp_path / "b.rsk")
    _touch(tmp_path / "a.rsk")

    result = resolve_deployment_files(tmp_path, None)

    assert result == [
        (tmp_path / "a.rsk").resolve(),
        (tmp_path / "b.rsk").resolve(),
    ]


def test_resolve_deployment_files_auto_discovery_ignores_subdirectories(
    tmp_path: Path,
) -> None:
    """Auto-discovery does not recurse into subdirectories."""
    _touch(tmp_path / "a.rsk")
    nested = tmp_path / "sub"
    nested.mkdir()
    _touch(nested / "nested.rsk")

    result = resolve_deployment_files(tmp_path, None)

    assert result == [(tmp_path / "a.rsk").resolve()]


def test_resolve_deployment_files_auto_discovery_empty_raises(
    tmp_path: Path,
) -> None:
    """An rsk_directory with no .rsk files raises ValueError on discovery."""
    _touch(tmp_path / "notes.txt")

    with pytest.raises(ValueError, match="No .rsk files"):
        resolve_deployment_files(tmp_path, None)


def test_resolve_deployment_files_rejects_dotdot_traversal(
    tmp_path: Path,
) -> None:
    """A ../ target that escapes rsk_directory is rejected."""
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()
    _touch(tmp_path / "outside.rsk")

    with pytest.raises(ValueError, match="outside rsk_directory"):
        resolve_deployment_files(rsk_dir, ["../outside.rsk"])


def test_resolve_deployment_files_rejects_absolute_target_escape(
    tmp_path: Path,
) -> None:
    """An absolute-path target outside rsk_directory is rejected."""
    rsk_dir = tmp_path / "rsk"
    rsk_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = _touch(outside_dir / "secret.rsk")

    with pytest.raises(ValueError, match="outside rsk_directory"):
        resolve_deployment_files(rsk_dir, [str(outside_file)])


def test_resolve_deployment_files_rejects_wrong_extension(
    tmp_path: Path,
) -> None:
    """A target that isn't a .rsk file is rejected."""
    _touch(tmp_path / "notes.txt")

    with pytest.raises(ValueError, match=r"\.rsk"):
        resolve_deployment_files(tmp_path, ["notes.txt"])


def test_resolve_deployment_files_rejects_nonexistent_target(
    tmp_path: Path,
) -> None:
    """A target naming a file that doesn't exist is rejected."""
    with pytest.raises(ValueError, match="does not exist"):
        resolve_deployment_files(tmp_path, ["missing.rsk"])


def test_resolve_deployment_files_rejects_target_that_is_a_directory(
    tmp_path: Path,
) -> None:
    """A target naming a directory rather than a file is rejected."""
    (tmp_path / "deployment.rsk").mkdir()

    with pytest.raises(ValueError, match="does not exist"):
        resolve_deployment_files(tmp_path, ["deployment.rsk"])


def test_resolve_deployment_files_missing_rsk_directory_raises(
    tmp_path: Path,
) -> None:
    """A nonexistent rsk_directory raises ValueError."""
    missing = tmp_path / "does-not-exist"

    with pytest.raises(ValueError, match="does not exist"):
        resolve_deployment_files(missing, None)


def test_resolve_deployment_files_rsk_directory_is_a_file_raises(
    tmp_path: Path,
) -> None:
    """An rsk_directory pointing at a plain file raises ValueError."""
    file_path = _touch(tmp_path / "not_a_dir.rsk")

    with pytest.raises(ValueError, match="not a directory"):
        resolve_deployment_files(file_path, None)
