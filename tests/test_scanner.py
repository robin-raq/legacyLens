"""Tests for the Fortran file scanner."""

import pytest
from pathlib import Path
from app.ingestion.scanner import scan_fortran_files


class TestScanFortranFiles:
    """Tests for scan_fortran_files()."""

    def test_finds_all_fortran_files(self):
        """Should find all .f and .f90 files in blas_src."""
        files = scan_fortran_files("blas_src")
        assert len(files) >= 150, f"Expected 150+ files, got {len(files)}"

    def test_returns_path_objects(self):
        """Should return a list of Path objects."""
        files = scan_fortran_files("blas_src")
        assert all(isinstance(f, Path) for f in files)

    def test_only_fortran_extensions(self):
        """Should only include Fortran file extensions."""
        valid_extensions = {".f", ".f90", ".f95", ".for", ".fpp"}
        files = scan_fortran_files("blas_src")
        for f in files:
            assert f.suffix.lower() in valid_extensions, f"Unexpected extension: {f.suffix}"

    def test_files_exist(self):
        """Every returned path should point to a real file."""
        files = scan_fortran_files("blas_src")
        for f in files[:10]:  # Spot check first 10
            assert f.exists(), f"File does not exist: {f}"

    def test_empty_directory_returns_empty(self, tmp_path):
        """Should return empty list for a directory with no Fortran files."""
        files = scan_fortran_files(str(tmp_path))
        assert files == []

    def test_deterministic_output(self):
        """Two scans should return the same files in the same order."""
        files1 = scan_fortran_files("blas_src")
        files2 = scan_fortran_files("blas_src")
        assert files1 == files2
