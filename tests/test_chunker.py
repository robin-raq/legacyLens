"""Tests for the Fortran chunker."""

import pytest
from pathlib import Path
from app.ingestion.chunker import (
    chunk_fortran_file,
    chunk_all_files,
    _detect_data_type,
    _detect_blas_level,
    _extract_purpose_block,
    _extract_param_summary,
)


# ── Helper: read a real BLAS file for testing ──


def _read_lines(filename: str) -> list[str]:
    """Read lines from a BLAS source file."""
    path = Path("blas_src/SRC") / filename
    return path.read_text(encoding="utf-8", errors="replace").split("\n")


# ── Data type detection ──


class TestDetectDataType:
    def test_double_real(self):
        assert _detect_data_type("DGEMM") == "double real"

    def test_single_real(self):
        assert _detect_data_type("SSWAP") == "single real"

    def test_single_complex(self):
        assert _detect_data_type("CAXPY") == "single complex"

    def test_double_complex(self):
        assert _detect_data_type("ZGEMV") == "double complex"

    def test_unknown_prefix(self):
        assert _detect_data_type("XERBLA") == "unknown"

    def test_empty_string(self):
        assert _detect_data_type("") == "unknown"


# ── BLAS level detection ──


class TestDetectBlasLevel:
    def test_level1_swap(self):
        assert _detect_blas_level("DSWAP") == "1"

    def test_level1_axpy(self):
        assert _detect_blas_level("SAXPY") == "1"

    def test_level1_dot(self):
        assert _detect_blas_level("CDOTC") == "1"

    def test_level2_gemv(self):
        assert _detect_blas_level("DGEMV") == "2"

    def test_level2_trsv(self):
        assert _detect_blas_level("ZTRSV") == "2"

    def test_level3_gemm(self):
        assert _detect_blas_level("DGEMM") == "3"

    def test_level3_trmm(self):
        assert _detect_blas_level("CTRMM") == "3"

    def test_unknown_name(self):
        assert _detect_blas_level("XERBLA") == "unknown"


# ── Purpose block extraction ──


class TestExtractPurposeBlock:
    def test_dgemm_has_purpose(self):
        lines = _read_lines("dgemm.f")
        purpose = _extract_purpose_block(lines)
        assert "matrix-matrix operations" in purpose.lower()

    def test_dswap_has_purpose(self):
        lines = _read_lines("dswap.f")
        purpose = _extract_purpose_block(lines)
        assert "interchange" in purpose.lower() or "swap" in purpose.lower()

    def test_purpose_strips_comment_prefixes(self):
        """Purpose text should not contain *> or * prefixes."""
        lines = _read_lines("dgemm.f")
        purpose = _extract_purpose_block(lines)
        assert "*>" not in purpose
        assert not any(line.startswith("*") for line in purpose.split("\n") if line.strip())

    def test_empty_file_returns_empty(self):
        purpose = _extract_purpose_block([])
        assert purpose == ""

    def test_file_without_purpose_returns_empty(self):
        lines = ["      SUBROUTINE FOO(X)", "      END SUBROUTINE"]
        purpose = _extract_purpose_block(lines)
        assert purpose == ""


# ── Parameter summary extraction ──


class TestExtractParamSummary:
    def test_dgemm_has_params(self):
        lines = _read_lines("dgemm.f")
        params = _extract_param_summary(lines)
        assert "TRANSA" in params
        assert "ALPHA" in params

    def test_param_count_capped(self):
        """Should not return more than 12 parameters."""
        lines = _read_lines("dgemm.f")
        params = _extract_param_summary(lines)
        param_lines = [l for l in params.split("\n") if l.strip()]
        assert len(param_lines) <= 12

    def test_empty_file_returns_empty(self):
        params = _extract_param_summary([])
        assert params == ""


# ── Full file chunking ──


class TestChunkFortranFile:
    def test_dgemm_produces_one_chunk(self):
        """DGEMM file should produce exactly one chunk."""
        chunks = chunk_fortran_file(Path("blas_src/SRC/dgemm.f"), "blas_src")
        assert len(chunks) == 1

    def test_chunk_has_correct_id(self):
        chunks = chunk_fortran_file(Path("blas_src/SRC/dgemm.f"), "blas_src")
        assert chunks[0].id == "SRC_dgemm.f::DGEMM"

    def test_chunk_has_metadata(self):
        chunks = chunk_fortran_file(Path("blas_src/SRC/dgemm.f"), "blas_src")
        m = chunks[0].metadata
        assert m.subroutine_name == "DGEMM"
        assert m.blas_level == "3"
        assert m.data_type == "double real"
        assert m.file_path == "SRC/dgemm.f"
        assert m.start_line > 0
        assert m.end_line > m.start_line

    def test_chunk_text_includes_purpose(self):
        """Chunk text should contain the Purpose description."""
        chunks = chunk_fortran_file(Path("blas_src/SRC/dgemm.f"), "blas_src")
        assert "Purpose:" in chunks[0].text
        assert "matrix-matrix operations" in chunks[0].text.lower()

    def test_chunk_text_includes_params(self):
        """Chunk text should contain parameter summaries."""
        chunks = chunk_fortran_file(Path("blas_src/SRC/dgemm.f"), "blas_src")
        assert "Parameters:" in chunks[0].text
        assert "TRANSA" in chunks[0].text

    def test_chunk_text_includes_source_code(self):
        """Chunk text should contain the actual Fortran source."""
        chunks = chunk_fortran_file(Path("blas_src/SRC/dgemm.f"), "blas_src")
        assert "SUBROUTINE DGEMM" in chunks[0].text

    def test_description_is_populated(self):
        """Metadata description should be non-empty for BLAS files."""
        chunks = chunk_fortran_file(Path("blas_src/SRC/dgemm.f"), "blas_src")
        assert len(chunks[0].metadata.description) > 10

    def test_nonexistent_file_returns_empty(self):
        chunks = chunk_fortran_file(Path("blas_src/SRC/nonexistent.f"), "blas_src")
        assert chunks == []


# ── Bulk chunking ──


class TestChunkAllFiles:
    def test_chunks_all_files(self):
        """Should produce at least one chunk per file."""
        from app.ingestion.scanner import scan_fortran_files

        files = scan_fortran_files("blas_src")
        chunks = chunk_all_files("blas_src", files)
        assert len(chunks) >= 150

    def test_all_chunks_have_purpose(self):
        """Every chunk should have a Purpose block in its text."""
        from app.ingestion.scanner import scan_fortran_files

        files = scan_fortran_files("blas_src")
        chunks = chunk_all_files("blas_src", files)
        with_purpose = [c for c in chunks if "Purpose:" in c.text]
        ratio = len(with_purpose) / len(chunks)
        assert ratio >= 0.95, f"Only {ratio:.0%} of chunks have Purpose blocks"

    def test_no_empty_chunks(self):
        """No chunk should have empty text."""
        from app.ingestion.scanner import scan_fortran_files

        files = scan_fortran_files("blas_src")
        chunks = chunk_all_files("blas_src", files)
        for c in chunks:
            assert len(c.text.strip()) > 0, f"Empty chunk: {c.id}"

    def test_unique_ids(self):
        """All chunk IDs should be unique."""
        from app.ingestion.scanner import scan_fortran_files

        files = scan_fortran_files("blas_src")
        chunks = chunk_all_files("blas_src", files)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"
