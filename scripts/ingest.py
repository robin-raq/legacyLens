"""Ingest BLAS codebase: scan -> chunk -> embed -> upsert to Pinecone."""

import sys
import time
sys.path.insert(0, ".")

from app.ingestion.scanner import scan_fortran_files
from app.ingestion.chunker import chunk_all_files
from app.embeddings.openai_embed import embed_texts
from app.vectordb.pinecone_client import upsert_chunks, get_index_stats

SOURCE_DIR = "blas_src"


def main():
    start = time.time()

    # Step 1: Scan
    print("Scanning Fortran files...")
    files = scan_fortran_files(SOURCE_DIR)
    print(f"  Found {len(files)} files")

    # Step 2: Chunk
    print("Chunking files (syntax-aware)...")
    chunks = chunk_all_files(SOURCE_DIR, files)
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
    print("Generating embeddings (text-embedding-3-small)...")
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    print(f"  Generated {len(embeddings)} embeddings ({len(embeddings[0])} dimensions)")

    # Step 4: Upsert
    print("Upserting to Pinecone...")
    count = upsert_chunks(chunks, embeddings)
    print(f"  Upserted {count} vectors")

    # Stats
    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")

    # Wait a moment for Pinecone to index
    print("Waiting 10s for Pinecone to index...")
    time.sleep(10)
    stats = get_index_stats()
    print(f"Index stats: {stats}")


if __name__ == "__main__":
    main()
