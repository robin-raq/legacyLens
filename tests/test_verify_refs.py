"""Tests for the reference verification script."""

import sys
sys.path.insert(0, ".")

import pytest
from scripts.verify_refs import verify_answer, scan_disk_files, RE_FILENAME, RE_LINE_REF


class TestFilenameRegex:
    """Tests for the Fortran filename regex pattern."""

    def test_matches_dot_f(self):
        assert RE_FILENAME.findall("see dgemm.f for details")

    def test_matches_dot_f90(self):
        assert RE_FILENAME.findall("check file.f90")

    def test_matches_dot_for(self):
        assert RE_FILENAME.findall("in legacy.for module")

    def test_no_match_on_plain_text(self):
        assert RE_FILENAME.findall("no file references here") == []

    def test_multiple_filenames(self):
        text = "Compare dgemm.f and saxpy.f implementations"
        matches = RE_FILENAME.findall(text)
        assert len(matches) == 2
        assert "dgemm.f" in matches
        assert "saxpy.f" in matches


class TestLineRefRegex:
    """Tests for line number reference patterns."""

    def test_line_number(self):
        assert RE_LINE_REF.findall("see line 42")

    def test_lines_range(self):
        assert RE_LINE_REF.findall("lines 10-20")

    def test_colon_format(self):
        """file:line format like dgemm.f:42."""
        assert RE_LINE_REF.findall("dgemm.f:42")

    def test_no_line_refs(self):
        assert RE_LINE_REF.findall("no references here") == []


class TestVerifyAnswer:
    """Tests for the verify_answer function."""

    def test_answer_with_expected_file(self):
        answer = "The DGEMM routine in dgemm.f performs matrix multiplication."
        result = verify_answer(answer, "dgemm.f", set())
        assert result["has_filename"] is True
        assert result["has_expected"] is True

    def test_answer_without_file_refs(self):
        answer = "DGEMM performs matrix multiplication."
        result = verify_answer(answer, "dgemm.f", set())
        assert result["has_filename"] is False
        assert result["has_expected"] is False

    def test_answer_with_line_refs(self):
        answer = "See dgemm.f line 42 for the implementation."
        result = verify_answer(answer, "dgemm.f", set())
        assert result["has_line_ref"] is True
        assert result["line_ref_count"] >= 1

    def test_answer_without_line_refs(self):
        answer = "See dgemm.f for the implementation."
        result = verify_answer(answer, "dgemm.f", set())
        assert result["has_line_ref"] is False

    def test_disk_validation_valid_file(self):
        disk = {"dgemm.f", "saxpy.f"}
        answer = "Found in dgemm.f at the start."
        result = verify_answer(answer, "dgemm.f", disk)
        assert result["valid_files"] == ["dgemm.f"]
        assert result["invalid_files"] == []

    def test_disk_validation_hallucinated_file(self):
        disk = {"dgemm.f", "saxpy.f"}
        answer = "Found in fake_routine.f which does not exist."
        result = verify_answer(answer, "dgemm.f", disk)
        assert "fake_routine.f" in result["invalid_files"]

    def test_case_insensitive_expected(self):
        answer = "See DGEMM.f for details."
        result = verify_answer(answer, "dgemm.f", set())
        assert result["has_expected"] is True

    def test_multiple_files_deduped(self):
        answer = "Both dgemm.f and dgemm.f are referenced, plus saxpy.f."
        result = verify_answer(answer, "dgemm.f", set())
        assert result["filenames"].count("dgemm.f") == 1  # deduped


class TestScanDiskFiles:
    """Tests for disk-based file scanning."""

    def test_finds_fortran_files(self, tmp_path):
        (tmp_path / "dgemm.f").write_text("content")
        (tmp_path / "saxpy.f90").write_text("content")
        (tmp_path / "readme.txt").write_text("content")  # not Fortran
        result = scan_disk_files(str(tmp_path))
        assert "dgemm.f" in result
        assert "saxpy.f90" in result
        assert "readme.txt" not in result

    def test_nonexistent_dir_returns_empty(self):
        result = scan_disk_files("/nonexistent/path")
        assert result == set()

    def test_returns_lowercase(self, tmp_path):
        (tmp_path / "DGEMM.F").write_text("content")
        result = scan_disk_files(str(tmp_path))
        assert "dgemm.f" in result
