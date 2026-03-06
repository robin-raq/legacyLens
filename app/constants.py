"""Shared BLAS constants used across ingestion and retrieval."""

# BLAS naming convention: first character indicates data type
DATA_TYPE_MAP = {
    "S": "single real",
    "D": "double real",
    "C": "single complex",
    "Z": "double complex",
}

DATA_TYPE_PREFIXES = list(DATA_TYPE_MAP.keys())

# BLAS operations grouped by level
BLAS_LEVELS = {
    "1": [
        "ROTG", "ROT", "ROTMG", "ROTM", "SWAP", "SCAL", "COPY", "AXPY",
        "DOT", "DOTU", "DOTC", "NRM2", "ASUM", "AMAX", "IAMAX", "AXPBY",
    ],
    "2": [
        "GEMV", "GBMV", "HEMV", "HBMV", "HPMV", "SYMV", "SBMV", "SPMV",
        "TRMV", "TBMV", "TPMV", "TRSV", "TBSV", "TPSV", "GER", "GERU",
        "GERC", "HER", "HPR", "HER2", "HPR2", "SYR", "SPR", "SYR2", "SPR2",
    ],
    "3": [
        "GEMM", "SYMM", "HEMM", "SYRK", "HERK", "SYR2K", "HER2K",
        "TRMM", "TRSM", "GEMMTR",
    ],
}

# Flat list of all BLAS operations (without prefix)
ALL_BLAS_OPS = [op for ops in BLAS_LEVELS.values() for op in ops]

# Utility routines (not prefixed by data type)
UTILITY_NAMES = {"XERBLA", "XERBLA_ARRAY", "LSAME"}

# Full set of known BLAS routine names (prefix + operation)
KNOWN_BLAS_NAMES: set[str] = set()
for _pfx in DATA_TYPE_PREFIXES:
    for _op in ALL_BLAS_OPS:
        KNOWN_BLAS_NAMES.add(_pfx + _op)
KNOWN_BLAS_NAMES.update(UTILITY_NAMES)
