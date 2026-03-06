"""Ingest BLAS codebase: scan -> chunk -> embed -> upsert to Pinecone.

Reports LOC count and per-step timing for throughput verification.
"""

import sys
import time
sys.path.insert(0, ".")

from pathlib import Path

from app.ingestion.scanner import scan_fortran_files
from app.ingestion.chunker import chunk_all_files
from app.embeddings.openai_embed import embed_texts
from app.vectordb.pinecone_client import upsert_chunks, get_index_stats

SOURCE_DIR = "blas_src"


def count_loc(files: list[Path]) -> tuple[int, int]:
    """Count total and non-blank lines of code across all files.

    Returns:
        (total_lines, non_blank_lines)
    """
    total = 0
    non_blank = 0
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            total += len(lines)
            non_blank += sum(1 for line in lines if line.strip())
        except OSError:
            pass
    return total, non_blank


def _step_time(label: str, t0: float) -> float:
    """Print elapsed time for a step and return the current time."""
    now = time.time()
    print(f"  ⏱  {now - t0:.1f}s — {label}")
    return now


def main():
    pipeline_start = time.time()

    # Step 1: Scan
    print("Step 1/4 — Scanning Fortran files...")
    files = scan_fortran_files(SOURCE_DIR)
    t = _step_time("scan", pipeline_start)
    print(f"  Found {len(files)} files")

    # LOC count
    total_loc, non_blank_loc = count_loc(files)
    print(f"  Total LOC: {total_loc:,}  (non-blank: {non_blank_loc:,})")

    # Step 2: Chunk
    print("\nStep 2/4 — Chunking files (syntax-aware)...")
    chunks = chunk_all_files(SOURCE_DIR, files)
    t = _step_time("chunk", t)
    print(f"  Generated {len(chunks)} chunks")

    if not chunks:
        print("ERROR: No chunks generated. Check source directory.")
        return

    # Show sample
    sample = chunks[0]
    print(f"\n  Sample chunk: {sample.id}")
    print(f"  Subroutine: {sample.metadata.subroutine_name}")
    print(f"  File: {sample.metadata.file_path}:{sample.metadata.start_line}-{sample.metadata.end_line}")
    print(f"  Lines: {sample.metadata.line_count}")
    print(f"  Text preview: {sample.text[:150]}...")
    print()

    # Step 3: Embed
    print("Step 3/4 — Generating embeddings (text-embedding-3-small)...")
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    t = _step_time("embed", t)
    print(f"  Generated {len(embeddings)} embeddings ({len(embeddings[0])} dimensions)")

    # Step 4: Upsert
    print("\nStep 4/4 — Upserting to Pinecone...")
    count = upsert_chunks(chunks, embeddings)
    t = _step_time("upsert", t)
    print(f"  Upserted {count} vectors")

    # ── Summary ──
    elapsed = time.time() - pipeline_start
    loc_per_sec = total_loc / elapsed if elapsed > 0 else 0
    print("\n" + "=" * 60)
    print("  Ingestion Summary")
    print("=" * 60)
    print(f"  Files:      {len(files)}")
    print(f"  LOC:        {total_loc:,}  (non-blank: {non_blank_loc:,})")
    print(f"  Chunks:     {len(chunks)}")
    print(f"  Vectors:    {count}")
    print(f"  Elapsed:    {elapsed:.1f}s")
    print(f"  Throughput: {loc_per_sec:,.0f} LOC/s")
    print("=" * 60)

    # Wait a moment for Pinecone to index
    print("\nWaiting 10s for Pinecone to index...")
    time.sleep(10)
    stats = get_index_stats()
    print(f"Index stats: {stats}")


if __name__ == "__main__":
    main()
