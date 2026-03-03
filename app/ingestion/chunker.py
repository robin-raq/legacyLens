"""Syntax-aware chunker for Fortran BLAS source code.

Splits on SUBROUTINE / FUNCTION / END boundaries using regex.
Falls back to fixed-size chunking for files that don't match.
"""

import re
from pathlib import Path
from app.models import CodeChunk, ChunkMetadata

# Matches start of a subroutine or function (Fortran 77 and 90 style)
_START_PATTERN = re.compile(
    r"^\s*(?:(?:DOUBLE\s+PRECISION|REAL|INTEGER|COMPLEX\*?\d*|LOGICAL|CHARACTER)\s+)?"
    r"(?:RECURSIVE\s+)?"
    r"(SUBROUTINE|FUNCTION)\s+(\w+)",
    re.IGNORECASE | re.MULTILINE,
)

# Matches END SUBROUTINE / END FUNCTION / plain END
_END_PATTERN = re.compile(
    r"^\s*END\s*(?:SUBROUTINE|FUNCTION)?\s*(\w*)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# BLAS naming convention for data type prefix
_DATA_TYPE_MAP = {
    "S": "single real",
    "D": "double real",
    "C": "single complex",
    "Z": "double complex",
}

# BLAS level detection from filename patterns
_BLAS_LEVELS = {
    "1": ["ROTG", "ROT", "ROTMG", "ROTM", "SWAP", "SCAL", "COPY", "AXPY",
           "DOT", "DOTU", "DOTC", "NRM2", "ASUM", "AMAX", "IAMAX"],
    "2": ["GEMV", "GBMV", "HEMV", "HBMV", "HPMV", "SYMV", "SBMV", "SPMV",
           "TRMV", "TBMV", "TPMV", "TRSV", "TBSV", "TPSV", "GER", "GERU",
           "GERC", "HER", "HPR", "HER2", "HPR2", "SYR", "SPR", "SYR2", "SPR2"],
    "3": ["GEMM", "SYMM", "HEMM", "SYRK", "HERK", "SYR2K", "HER2K",
           "TRMM", "TRSM"],
}


def _detect_data_type(name: str) -> str:
    """Detect data type from BLAS naming convention (first char)."""
    if name and name[0].upper() in _DATA_TYPE_MAP:
        return _DATA_TYPE_MAP[name[0].upper()]
    return "unknown"


def _detect_blas_level(name: str) -> str:
    """Detect BLAS level (1, 2, or 3) from subroutine name."""
    upper = name.upper()
    # Strip data type prefix (first char)
    suffix = upper[1:] if len(upper) > 1 else ""
    for level, ops in _BLAS_LEVELS.items():
        for op in ops:
            if suffix == op or suffix.startswith(op):
                return level
    return "unknown"


def _extract_header_comment(lines: list[str], start_idx: int) -> str:
    """Extract the header comment block (c, C, *, !) after subroutine declaration."""
    comments = []
    for i in range(start_idx + 1, min(start_idx + 50, len(lines))):
        line = lines[i]
        stripped = line.lstrip()
        if stripped and stripped[0] in ("c", "C", "*", "!"):
            # Remove comment character and leading spaces
            comment_text = stripped[1:].strip()
            if comment_text:
                comments.append(comment_text)
        elif stripped == "":
            continue
        else:
            break
    return " ".join(comments[:5])  # First 5 meaningful comment lines


def chunk_fortran_file(file_path: Path, source_dir: str) -> list[CodeChunk]:
    """Split a Fortran file into subroutine/function-level chunks."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    lines = content.split("\n")
    rel_path = str(file_path.relative_to(source_dir))
    chunks = []

    # Find all subroutine/function boundaries
    boundaries = []
    for i, line in enumerate(lines):
        start_match = _START_PATTERN.match(line)
        if start_match:
            kind = start_match.group(1).upper()
            name = start_match.group(2)
            boundaries.append({"start": i, "name": name, "kind": kind})

        end_match = _END_PATTERN.match(line)
        if end_match and boundaries and "end" not in boundaries[-1]:
            boundaries[-1]["end"] = i

    # If we found subroutine boundaries, use them
    if boundaries:
        for b in boundaries:
            start = b["start"]
            end = b.get("end", min(start + 200, len(lines) - 1))
            chunk_lines = lines[start : end + 1]
            chunk_text = "\n".join(chunk_lines)
            name = b["name"]

            description = _extract_header_comment(lines, start)
            data_type = _detect_data_type(name)
            blas_level = _detect_blas_level(name)

            # Prepend metadata to chunk text for better semantic matching
            metadata_prefix = f"BLAS {b['kind']} {name}"
            if blas_level != "unknown":
                metadata_prefix += f" (Level {blas_level}, {data_type})"
            if description:
                metadata_prefix += f": {description}"
            metadata_prefix += "\n\n"

            chunk_id = f"{rel_path}::{name}".replace("/", "_").replace("\\", "_")

            chunks.append(CodeChunk(
                id=chunk_id,
                text=metadata_prefix + chunk_text,
                metadata=ChunkMetadata(
                    file_path=rel_path,
                    start_line=start + 1,
                    end_line=end + 1,
                    subroutine_name=name,
                    blas_level=blas_level,
                    data_type=data_type,
                    description=description,
                    line_count=end - start + 1,
                ),
            ))
    else:
        # Fallback: whole file as one chunk (for files without clear boundaries)
        if len(lines) > 5:
            chunk_text = content
            chunk_id = f"{rel_path}::file".replace("/", "_").replace("\\", "_")
            chunks.append(CodeChunk(
                id=chunk_id,
                text=chunk_text,
                metadata=ChunkMetadata(
                    file_path=rel_path,
                    start_line=1,
                    end_line=len(lines),
                    subroutine_name=file_path.stem,
                    line_count=len(lines),
                ),
            ))

    return chunks


def chunk_all_files(source_dir: str, files: list[Path]) -> list[CodeChunk]:
    """Chunk all Fortran files."""
    all_chunks = []
    for f in files:
        all_chunks.extend(chunk_fortran_file(f, source_dir))
    return all_chunks
