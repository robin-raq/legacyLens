"""Syntax-aware chunker for Fortran BLAS source code.

Splits on SUBROUTINE / FUNCTION / END boundaries using regex.
Extracts rich metadata from BLAS file headers (Purpose, params).
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
    suffix = upper[1:] if len(upper) > 1 else ""
    for level, ops in _BLAS_LEVELS.items():
        for op in ops:
            if suffix == op or suffix.startswith(op):
                return level
    return "unknown"


def _extract_purpose_block(lines: list[str]) -> str:
    """Extract the Purpose section from the BLAS file header.

    BLAS files have a structured header above the SUBROUTINE line:
        *> \\par Purpose:
        *  =============
        *> \\verbatim
        *>
        *> DGEMM performs one of the matrix-matrix operations ...
        *> \\endverbatim

    We grab the text between \\verbatim and \\endverbatim after
    the Purpose marker, strip the *> comment prefixes, and return
    clean English text.
    """
    in_purpose = False
    in_verbatim = False
    purpose_lines = []

    for line in lines:
        stripped = line.strip()

        # Look for the Purpose marker
        if "\\par Purpose" in stripped or "Purpose:" in stripped:
            in_purpose = True
            continue

        if in_purpose:
            # Enter the verbatim block (the actual description text)
            if "\\verbatim" in stripped and "\\endverbatim" not in stripped:
                in_verbatim = True
                continue

            # Exit the verbatim block — we're done
            if "\\endverbatim" in stripped:
                break

            if in_verbatim:
                # Strip the *> or * comment prefix
                text = stripped
                if text.startswith("*>"):
                    text = text[2:].strip()
                elif text.startswith("*"):
                    text = text[1:].strip()
                purpose_lines.append(text)

    # Join and clean up (collapse multiple blank lines)
    result = "\n".join(purpose_lines).strip()
    # Collapse runs of blank lines into single blank line
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def _extract_param_summary(lines: list[str]) -> str:
    """Extract a brief summary of parameters from the file header.

    Returns a compact list like:
      TRANSA: CHARACTER*1 - form of op(A)
      M: INTEGER - number of rows

    We only grab the first line of each param description to keep it short.
    """
    params = []
    current_param = None
    in_verbatim = False

    for line in lines:
        stripped = line.strip()

        # Detect \param[in] NAME or \param[in,out] NAME
        param_match = re.search(r"\\param\[[\w,]+\]\s+(\w+)", stripped)
        if param_match:
            current_param = param_match.group(1)
            in_verbatim = False
            continue

        if current_param:
            if "\\verbatim" in stripped and "\\endverbatim" not in stripped:
                in_verbatim = True
                continue

            if "\\endverbatim" in stripped:
                current_param = None
                in_verbatim = False
                continue

            if in_verbatim:
                text = stripped
                if text.startswith("*>"):
                    text = text[2:].strip()
                elif text.startswith("*"):
                    text = text[1:].strip()

                # Grab the first meaningful line (usually "NAME is TYPE")
                if text and current_param and current_param not in [p.split(":")[0] for p in params]:
                    params.append(f"{current_param}: {text}")

        # Stop once we hit the actual code (subroutine declaration)
        if _START_PATTERN.match(line):
            break

    return "\n".join(params[:12])  # Cap at 12 params to avoid huge chunks


def _extract_header_comment(lines: list[str], start_idx: int) -> str:
    """Extract the header comment block (c, C, *, !) after subroutine declaration."""
    comments = []
    for i in range(start_idx + 1, min(start_idx + 50, len(lines))):
        line = lines[i]
        stripped = line.lstrip()
        if stripped and stripped[0] in ("c", "C", "*", "!"):
            comment_text = stripped[1:].strip()
            if comment_text:
                comments.append(comment_text)
        elif stripped == "":
            continue
        else:
            break
    return " ".join(comments[:5])


def chunk_fortran_file(file_path: Path, source_dir: str) -> list[CodeChunk]:
    """Split a Fortran file into subroutine/function-level chunks.

    Each chunk includes:
    - The Purpose description from the file header (if present)
    - A brief parameter summary
    - Rich metadata prefix for better embedding quality
    - The actual subroutine source code
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    lines = content.split("\n")
    rel_path = str(file_path.relative_to(source_dir))
    chunks = []

    # Extract documentation from the file header (above the SUBROUTINE line)
    purpose = _extract_purpose_block(lines)
    param_summary = _extract_param_summary(lines)

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

    if boundaries:
        for b in boundaries:
            start = b["start"]
            end = b.get("end", min(start + 200, len(lines) - 1))
            chunk_lines = lines[start : end + 1]
            chunk_text = "\n".join(chunk_lines)
            name = b["name"]

            # Use purpose block if available, fall back to inline header comment
            description = purpose if purpose else _extract_header_comment(lines, start)
            data_type = _detect_data_type(name)
            blas_level = _detect_blas_level(name)

            # Build a rich metadata prefix for the embedding
            metadata_prefix = f"BLAS {b['kind']} {name}"
            if blas_level != "unknown":
                metadata_prefix += f" (Level {blas_level}, {data_type})"
            metadata_prefix += "\n"

            if description:
                metadata_prefix += f"\nPurpose: {description}\n"

            if param_summary:
                metadata_prefix += f"\nParameters:\n{param_summary}\n"

            metadata_prefix += "\nSource Code:\n"

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
                    description=description[:500],  # Cap for metadata storage
                    line_count=end - start + 1,
                ),
            ))
    else:
        # Fallback: whole file as one chunk
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
