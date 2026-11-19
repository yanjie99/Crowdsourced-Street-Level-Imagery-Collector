"""The folder browser behind "Save in" (no network needed)."""
import os

import pytest

from collect.ui.server import ApiError, list_folders, make_folder


def test_lists_folders_only_and_goes_up(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "A").mkdir()
    (tmp_path / "file.txt").write_text("x")
    r = list_folders(str(tmp_path))
    assert r["folders"] == ["A", "b"]                 # folders only, sorted ignoring case
    assert r["parent"] == os.path.dirname(str(tmp_path)) and r["drives"]
    assert list_folders("")["path"] == ""             # empty path: the drive list


def test_make_folder_checks_the_name(tmp_path):
    r = make_folder(str(tmp_path), "Street imagery (2026)")
    assert os.path.isdir(tmp_path / "Street imagery (2026)") and r["path"].endswith("Street imagery (2026)")
    for bad in ("", "..", "a\\b", "a/b", "c:", "x."):
        with pytest.raises(ApiError):
            make_folder(str(tmp_path), bad)
    with pytest.raises(ApiError):                     # already there
        make_folder(str(tmp_path), "Street imagery (2026)")
