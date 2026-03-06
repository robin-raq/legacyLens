"""Tests for ingestion throughput metrics (LOC counting, step timing)."""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

# We need to import from scripts/ which isn't a package, so patch sys.path
import sys
sys.path.insert(0, ".")
from scripts.ingest import count_loc, _step_time


class TestCountLoc:
    """Tests for the count_loc function."""

    def test_counts_lines_in_single_file(self, tmp_path):
        """Should count total and non-blank lines correctly."""
        f = tmp_path / "test.f"
        f.write_text("      SUBROUTINE FOO\n\n      END\n")
        total, non_blank = count_loc([f])
        assert total == 3
        assert non_blank == 2  # blank line excluded

    def test_counts_across_multiple_files(self, tmp_path):
        """Should sum LOC across all files."""
        f1 = tmp_path / "a.f"
        f1.write_text("line1\nline2\n")
        f2 = tmp_path / "b.f"
        f2.write_text("line1\nline2\nline3\n")
        total, non_blank = count_loc([f1, f2])
        assert total == 5
        assert non_blank == 5

    def test_empty_file_returns_zero(self, tmp_path):
        """Empty file should contribute 0 lines."""
        f = tmp_path / "empty.f"
        f.write_text("")
        total, non_blank = count_loc([f])
        assert total == 0
        assert non_blank == 0

    def test_blank_only_file(self, tmp_path):
        """File with only blank lines: total > 0 but non_blank == 0."""
        f = tmp_path / "blanks.f"
        f.write_text("\n\n\n")
        total, non_blank = count_loc([f])
        assert total == 3
        assert non_blank == 0

    def test_nonexistent_file_skipped(self, tmp_path):
        """Missing files should be silently skipped."""
        missing = tmp_path / "nope.f"
        total, non_blank = count_loc([missing])
        assert total == 0
        assert non_blank == 0

    def test_empty_file_list(self):
        """No files = zero LOC."""
        total, non_blank = count_loc([])
        assert total == 0
        assert non_blank == 0


class TestStepTime:
    """Tests for the _step_time helper."""

    def test_returns_current_time(self):
        """Should return a time value after the given t0."""
        t0 = time.time()
        result = _step_time("test", t0)
        assert result >= t0

    def test_prints_label(self, capsys):
        """Should print the label with elapsed time."""
        t0 = time.time()
        _step_time("scan", t0)
        captured = capsys.readouterr()
        assert "scan" in captured.out
        assert "s" in captured.out  # has time unit
