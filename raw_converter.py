"""
raw_converter.py — NG Data Conversion Pipeline
================================================
Converts a raw binary volume file into an OME-Zarr multiresolution store
suitable for viewing in Neuroglancer.

Auto-discovers the paired .nhdr NRRD header file (same stem, same directory)
to extract volume shape, dtype, and encoding — no hardcoded values needed.

Usage (CLI):
    python3 raw_converter.py --input /path/to/file.raw --output /path/to/output.zarr

    # All volume params are read from the .nhdr automatically.
    # You can override any of them via CLI flags:
    python3 raw_converter.py --input file.raw --output out.zarr --shape 706,20000,20000

Arguments:
    --input   Path to the raw binary input file (.raw)
    --output  Path (directory) for the output OME-Zarr store
    --shape   Override Z,Y,X shape  (default: parsed from .nhdr)
    --dtype   Override NumPy dtype  (default: parsed from .nhdr)
    --chunks  Dask chunk sizes Z,Y,X (default: auto-computed from shape)
    --levels  Number of pyramid downsampling levels (default: 4)

Exit codes:
    0 — success
    1 — missing file, parse error, or conversion error
"""

import argparse
import sys
import os
import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import dask.array as da
import zarr
from ome_zarr.writer import write_image
from ome_zarr.scale import Scaler

# ---------------------------------------------------------------------------
# Logging — stdout so FastAPI's subprocess capture picks it up cleanly
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("raw_converter")

# ---------------------------------------------------------------------------
# NRRD dtype string → NumPy dtype mapping
# Covers all types defined in the NRRD spec:
#   http://teem.sourceforge.net/nrrd/format.html
# ---------------------------------------------------------------------------
NRRD_DTYPE_MAP: Dict[str, str] = {
    # Unsigned integers
    "unsigned char":       "uint8",
    "uchar":               "uint8",
    "uint8":               "uint8",
    "uint8_t":             "uint8",
    "unsigned short":      "uint16",
    "ushort":              "uint16",
    "uint16":              "uint16",
    "uint16_t":            "uint16",
    "unsigned int":        "uint32",
    "uint":                "uint32",
    "uint32":              "uint32",
    "uint32_t":            "uint32",
    "unsigned long":       "uint64",
    "unsigned long long":  "uint64",
    "uint64":              "uint64",
    "uint64_t":            "uint64",
    # Signed integers
    "signed char":         "int8",
    "int8":                "int8",
    "int8_t":              "int8",
    "short":               "int16",
    "int16":               "int16",
    "int16_t":             "int16",
    "int":                 "int32",
    "int32":               "int32",
    "int32_t":             "int32",
    "long":                "int64",
    "long long":           "int64",
    "int64":               "int64",
    "int64_t":             "int64",
    # Floating point
    "float":               "float32",
    "double":              "float64",
}


# ---------------------------------------------------------------------------
# NRRD / NHDR Header Parser
# ---------------------------------------------------------------------------

def parse_nhdr(nhdr_path: str) -> Dict[str, Any]:
    """
    Parse a detached NRRD header file (.nhdr) and return a dict with:

        {
            "shape":    (Z, Y, X)  — NumPy / C-order (reversed from NRRD X,Y,Z),
            "dtype":    np.dtype,
            "encoding": str,       — e.g. "raw", "gzip"
            "data_file": str,      — relative or absolute path to the .raw file
            "dimension": int,
            "raw_sizes": (X, Y, Z) — original NRRD order for reference
        }

    Raises:
        FileNotFoundError: if nhdr_path does not exist.
        ValueError: if required fields (type, sizes) are missing or unparseable.
    """
    if not os.path.isfile(nhdr_path):
        raise FileNotFoundError(f"NHDR file not found: {nhdr_path}")

    fields: Dict[str, str] = {}

    with open(nhdr_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()

            # Skip blank lines and pure comment lines
            if not line or line.startswith("#"):
                continue

            # NRRD magic line (e.g. "NRRD0004") — skip
            if line.upper().startswith("NRRD"):
                continue

            # Key: value  (split on first colon only)
            if ":" in line:
                key, _, value = line.partition(":")
                fields[key.strip().lower()] = value.strip()

    # ------------------------------------------------------------------ dtype
    raw_type = fields.get("type", "")
    dtype_str = NRRD_DTYPE_MAP.get(raw_type.lower())
    if dtype_str is None:
        raise ValueError(
            f"Unknown NRRD type '{raw_type}' in {nhdr_path}. "
            f"Supported types: {list(NRRD_DTYPE_MAP.keys())}"
        )
    dtype = np.dtype(dtype_str)

    # ------------------------------------------------------------------ sizes
    raw_sizes_str = fields.get("sizes", "")
    if not raw_sizes_str:
        raise ValueError(f"'sizes' field missing from {nhdr_path}")

    try:
        # NRRD stores sizes as X Y Z (fastest→slowest axis)
        raw_sizes = tuple(int(s) for s in raw_sizes_str.split())
    except ValueError:
        raise ValueError(f"Cannot parse 'sizes' field '{raw_sizes_str}' in {nhdr_path}")

    dimension = int(fields.get("dimension", len(raw_sizes)))
    if len(raw_sizes) != dimension:
        raise ValueError(
            f"'dimension' ({dimension}) does not match number of sizes "
            f"({len(raw_sizes)}) in {nhdr_path}"
        )

    # Convert NRRD (X, Y, Z) → NumPy C-order (Z, Y, X)
    shape: Tuple[int, ...] = tuple(reversed(raw_sizes))

    # ----------------------------------------------------------- data file
    data_file = fields.get("data file", "") or fields.get("datafile", "")

    encoding = fields.get("encoding", "raw").lower()

    return {
        "shape":     shape,
        "dtype":     dtype,
        "encoding":  encoding,
        "data_file": data_file,
        "dimension": dimension,
        "raw_sizes": raw_sizes,
    }


def auto_chunks(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Compute sensible Dask chunk sizes for a 3-D volume.

    Targets ~64–128 slices in Z and 2 048-pixel tiles in Y/X,
    capped at the actual dimension size to avoid empty chunks.

    Args:
        shape: (Z, Y, X) volume dimensions.

    Returns:
        (chunk_z, chunk_y, chunk_x)
    """
    z, y, x = shape
    chunk_z = min(16, z)
    chunk_y = min(2048, y)
    chunk_x = min(2048, x)
    return (chunk_z, chunk_y, chunk_x)


# ---------------------------------------------------------------------------
# Conversion Core
# ---------------------------------------------------------------------------

def convert(
    raw_file: str,
    output_zarr: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    chunks: Tuple[int, ...],
    max_layer: int,
) -> None:
    """
    Memory-map a raw binary volume, wrap it in Dask, and write an
    OME-Zarr multiresolution pyramid.

    Args:
        raw_file:    Path to the input .raw binary file.
        output_zarr: Destination path for the .zarr store.
        shape:       (Z, Y, X) volume dimensions.
        dtype:       NumPy dtype.
        chunks:      (Z, Y, X) Dask chunk sizes.
        max_layer:   Number of downsampled pyramid levels.
    """
    if not os.path.isfile(raw_file):
        logger.error("Input file not found: %s", raw_file)
        sys.exit(1)

    # Step 1 — Memory-map (no RAM copy)
    logger.info(
        "Memory-mapping raw file: %s  |  shape=%s  dtype=%s",
        raw_file, shape, dtype,
    )
    mapped_data = np.memmap(raw_file, dtype=dtype, mode='r', shape=shape)

    # Step 2 — Dask array (lazy, chunked)
    logger.info("Wrapping in Dask array with chunks=%s", chunks)
    dask_array = da.from_array(mapped_data, chunks=chunks)

    # Step 3 — Zarr store
    logger.info("Creating OME-Zarr store at: %s", output_zarr)
    store = zarr.storage.LocalStore(output_zarr)
    root = zarr.group(store=store, overwrite=True)

    # Step 4 — Multiresolution pyramid
    scaler = Scaler(max_layer=max_layer, method='nearest')
    logger.info(
        "Writing OME-Zarr pyramid (%d downsampling levels) — "
        "this may take a while for large volumes…",
        max_layer,
    )
    write_image(
        image=dask_array,
        group=root,
        axes="zyx",
        scaler=scaler,
    )

    logger.info("✅ Conversion complete → %s", output_zarr)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a raw binary volume to an OME-Zarr multiresolution store. "
            "Volume parameters are auto-read from the paired .nhdr file; "
            "CLI flags override any parsed values."
        )
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Path to the input .raw binary file.")
    parser.add_argument("--output", "-o", required=True,
                        help="Destination path for the output .zarr directory.")
    parser.add_argument("--shape",  default=None,
                        help="Override shape as Z,Y,X (e.g. 706,20000,20000). "
                             "Auto-detected from .nhdr if omitted.")
    parser.add_argument("--dtype",  default=None,
                        help="Override NumPy dtype (e.g. uint8). "
                             "Auto-detected from .nhdr if omitted.")
    parser.add_argument("--chunks", default=None,
                        help="Dask chunk sizes as Z,Y,X. Auto-computed if omitted.")
    parser.add_argument("--levels", type=int, default=4,
                        help="Number of OME-Zarr pyramid levels (default: 4).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_file: str  = os.path.abspath(args.input)
    output_zarr: str = os.path.abspath(args.output)

    # ------------------------------------------------------------------
    # Auto-discover and parse the .nhdr file
    # (same directory, same stem, .nhdr extension)
    # ------------------------------------------------------------------
    stem     = os.path.splitext(raw_file)[0]
    nhdr_path = stem + ".nhdr"

    nhdr_meta: Dict[str, Any] = {}

    if os.path.isfile(nhdr_path):
        logger.info("Found NHDR header: %s — parsing…", nhdr_path)
        try:
            nhdr_meta = parse_nhdr(nhdr_path)
            logger.info(
                "NHDR parsed  →  raw sizes (X,Y,Z)=%s  |  numpy shape (Z,Y,X)=%s  |  dtype=%s  |  encoding=%s",
                nhdr_meta["raw_sizes"],
                nhdr_meta["shape"],
                nhdr_meta["dtype"],
                nhdr_meta["encoding"],
            )
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("Could not parse NHDR (%s). Will use CLI/default values.", exc)
    else:
        logger.warning(
            "No .nhdr file found at %s — falling back to CLI / default values.",
            nhdr_path,
        )

    # ------------------------------------------------------------------
    # Resolve final shape — priority: CLI flag > NHDR > error
    # ------------------------------------------------------------------
    if args.shape:
        try:
            shape = tuple(int(x) for x in args.shape.split(","))
            assert len(shape) == 3
        except (ValueError, AssertionError):
            logger.error("--shape must be 3 comma-separated integers, e.g. 706,20000,20000")
            sys.exit(1)
        logger.info("Shape overridden via CLI: %s", shape)
    elif nhdr_meta.get("shape"):
        shape = nhdr_meta["shape"]
    else:
        logger.error(
            "Cannot determine volume shape: no .nhdr file found and --shape not provided."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Resolve dtype — priority: CLI flag > NHDR > error
    # ------------------------------------------------------------------
    if args.dtype:
        dtype = np.dtype(args.dtype)
        logger.info("Dtype overridden via CLI: %s", dtype)
    elif nhdr_meta.get("dtype") is not None:
        dtype = nhdr_meta["dtype"]
    else:
        logger.error(
            "Cannot determine dtype: no .nhdr file found and --dtype not provided."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Resolve chunks — priority: CLI flag > auto-compute from shape
    # ------------------------------------------------------------------
    if args.chunks:
        try:
            chunks = tuple(int(x) for x in args.chunks.split(","))
            assert len(chunks) == 3
        except (ValueError, AssertionError):
            logger.error("--chunks must be 3 comma-separated integers, e.g. 16,2000,2000")
            sys.exit(1)
        logger.info("Chunks overridden via CLI: %s", chunks)
    else:
        chunks = auto_chunks(shape)
        logger.info("Chunks auto-computed from shape: %s", chunks)

    # ------------------------------------------------------------------
    # Run conversion
    # ------------------------------------------------------------------
    convert(
        raw_file=raw_file,
        output_zarr=output_zarr,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        max_layer=args.levels,
    )


if __name__ == "__main__":
    main()