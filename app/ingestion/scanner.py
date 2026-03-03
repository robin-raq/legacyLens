"""Scan BLAS source directory for Fortran files."""

import os
from pathlib import Path


def scan_fortran_files(source_dir: str) -> list[Path]:
    """Recursively find all Fortran source files."""
    extensions = {".f", ".f90", ".f95", ".for", ".fpp"}
    files = []
    for root, _, filenames in os.walk(source_dir):
        for fname in sorted(filenames):
            if Path(fname).suffix.lower() in extensions:
                files.append(Path(root) / fname)
    return files
