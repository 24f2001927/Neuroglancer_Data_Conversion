"""
FastAPI Orchestration & Monitoring Backend
==========================================
This application acts as the central control plane for a file conversion pipeline.

It communicates with:
  - Raw Data Server:       http://172.20.23.241:10228/
  - Converted Data Server: http://172.20.23.241:10229/
  - Local conversion script (subprocess)

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import httpx
from bs4 import BeautifulSoup
from fastapi import BackgroundTasks, FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pipeline_api")

# ---------------------------------------------------------------------------
# External Server Configuration
# ⚠️  MODIFY THESE if your server addresses or paths change.
# ---------------------------------------------------------------------------
RAW_SERVER_URL = "http://172.20.23.241:10228/"
CONVERTED_SERVER_URL = "http://172.20.23.241:10229/"

# ---------------------------------------------------------------------------
# Conversion Script Configuration — raw_converter.py
# ⚠️  Adjust RAW_FILES_DIR and OUTPUT_ZARR_DIR to match your environment.
#     Volume shape, dtype and chunks are now read AUTOMATICALLY from the
#     paired .nhdr file — no manual config needed.
# ---------------------------------------------------------------------------

# Absolute path to the conversion script (same directory as main.py)
CONVERSION_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "raw_converter.py"
)

# Directory where raw .raw / .nhdr file pairs are stored locally.
# ⚠️  Update this to your actual raw data directory.
RAW_FILES_DIR = "/home/tahmeed/nvIndexViewer/"

# Directory where converted .zarr stores will be written.
# ⚠️  Update this to your desired output location.
OUTPUT_ZARR_DIR = "/home/tahmeed/Dataneuroglancer/converted"

# Default number of OME-Zarr downsampling pyramid levels.
# Can be overridden per-request in the future.
PYRAMID_LEVELS = 4

# Request timeout for fetching remote directory listings (seconds)
HTTP_TIMEOUT = 10.0

# ---------------------------------------------------------------------------
# NRRD dtype string → NumPy dtype  (mirrors raw_converter.py for API use)
# ---------------------------------------------------------------------------
_NRRD_DTYPE_MAP: Dict[str, str] = {
    "unsigned char": "uint8",
    "uchar": "uint8",
    "uint8": "uint8",
    "unsigned short": "uint16",
    "ushort": "uint16",
    "uint16": "uint16",
    "unsigned int": "uint32",
    "uint": "uint32",
    "uint32": "uint32",
    "unsigned long long": "uint64",
    "uint64": "uint64",
    "signed char": "int8",
    "int8": "int8",
    "short": "int16",
    "int16": "int16",
    "int": "int32",
    "int32": "int32",
    "long long": "int64",
    "int64": "int64",
    "float": "float32",
    "double": "float64",
}

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NG Data Conversion Pipeline API",
    description=(
        "Orchestration and monitoring backend for the NG file conversion pipeline. "
        "Provides endpoints to list raw/converted files, check conversion status, "
        "and trigger background conversions."
    ),
    version="1.0.0",
    contact={"name": "Pipeline Ops"},
)

# ---------------------------------------------------------------------------
# CORS Middleware
# ⚠️  Restrict `allow_origins` to specific domains in production.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (index.html, style.css) from the project directory
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=_PROJECT_DIR), name="static")

# ---------------------------------------------------------------------------
# Pydantic Response Models
# ---------------------------------------------------------------------------


class RawFilesResponse(BaseModel):
    """Response model for GET /api/files/raw"""

    raw_files: List[str] = Field(
        ..., description="List of filenames on the raw data server."
    )


class ConvertedFilesResponse(BaseModel):
    """Response model for GET /api/files/converted"""

    converted_files: List[str] = Field(
        ..., description="List of filenames on the converted data server."
    )


class FileStatusEntry(BaseModel):
    """A single file's conversion status entry."""

    filename: str = Field(..., description="The raw filename.")
    status: str = Field(
        ...,
        description="'converted' if a matching file exists on the converted server, otherwise 'pending_conversion'.",
    )
    converted_url: Optional[str] = Field(
        None,
        description="Direct URL to the converted file (only present when status is 'converted').",
    )


class NhdrMetadata(BaseModel):
    """Parsed metadata from a .nhdr NRRD header file."""

    filename: str = Field(
        ..., description="The .raw filename this metadata belongs to."
    )
    nhdr_file: str = Field(
        ..., description="Absolute path to the .nhdr file that was parsed."
    )
    raw_sizes: List[int] = Field(
        ..., description="Original NRRD sizes order: [X, Y, Z]."
    )
    numpy_shape: List[int] = Field(..., description="NumPy / C-order shape: [Z, Y, X].")
    dtype: str = Field(..., description="NumPy dtype string (e.g. 'uint8').")
    encoding: str = Field(..., description="NRRD encoding field (e.g. 'raw', 'gzip').")
    dimension: int = Field(..., description="Number of spatial dimensions.")
    data_file: Optional[str] = Field(
        None, description="'data file' field from the header if present."
    )


class ConversionTriggerResponse(BaseModel):
    """Response model for POST /api/convert/{filename}"""

    message: str
    status: str
    filename: str
    nhdr_found: bool = Field(
        ..., description="Whether a paired .nhdr file was found and will be used."
    )
    input_path: str = Field(
        ..., description="Absolute path to the .raw file being converted."
    )
    output_path: str = Field(
        ..., description="Absolute path where the .zarr store will be written."
    )


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _parse_directory_listing(html: str) -> List[str]:
    """
    Parse a standard HTTP directory listing page and return a list of filenames.

    Works with:
      - Python's built-in http.server / SimpleHTTPServer
      - Nginx autoindex
      - Apache mod_autoindex

    ⚠️  If your server returns a non-standard listing page, update the parsing
        logic below. The key selector is the <a> tag whose href does NOT start
        with '?' (query strings used for sorting) and is NOT the parent link '..'.

    Args:
        html: Raw HTML string of the directory listing page.

    Returns:
        List of filenames (strings), excluding navigation/sorting links.
    """
    soup = BeautifulSoup(html, "html.parser")
    filenames: List[str] = []

    for anchor in soup.find_all("a", href=True):
        href: str = anchor["href"]

        # Skip sorting/query links (e.g. ?C=N&O=D), parent directory links,
        # and any absolute URLs that point elsewhere.
        if (
            href.startswith("?")
            or href == "/"
            or href == "../"
            or href.startswith("http")
        ):
            continue

        # Strip any trailing slash (directory entries) — keep if you want dirs too.
        # ⚠️  Remove the `if not href.endswith("/")` guard if you want to list
        #     sub-directories as well.
        if not href.endswith("/"):
            filenames.append(href)

    return filenames


async def _fetch_file_list(server_url: str) -> List[str]:
    """
    Asynchronously fetch and parse the directory listing from `server_url`.

    Raises:
        HTTPException(502): If the remote server is unreachable or returns an error.
    """
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(server_url, follow_redirects=True)
            response.raise_for_status()
    except httpx.RequestError as exc:
        logger.error("Network error fetching %s: %s", server_url, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach file server at {server_url}. Network error: {exc}",
        )
    except httpx.HTTPStatusError as exc:
        logger.error("HTTP error from %s: %s", server_url, exc.response.status_code)
        raise HTTPException(
            status_code=502,
            detail=(
                f"File server at {server_url} returned HTTP {exc.response.status_code}."
            ),
        )

    return _parse_directory_listing(response.text)


# ---------------------------------------------------------------------------
# NHDR / NRRD Header Parser  (used by API endpoints and background tasks)
# ---------------------------------------------------------------------------


def _parse_nhdr_file(nhdr_path: str) -> Dict[str, Any]:
    """
    Parse a detached NRRD header (.nhdr) and return a metadata dict:

        {
            "raw_sizes":   (X, Y, Z)  — original NRRD order,
            "numpy_shape": (Z, Y, X)  — C-order for NumPy,
            "dtype":       str,        — NumPy dtype string,
            "encoding":    str,
            "data_file":   str,
            "dimension":   int,
        }

    Raises:
        FileNotFoundError: nhdr_path does not exist.
        ValueError: Required fields missing or dtype unknown.
    """
    if not os.path.isfile(nhdr_path):
        raise FileNotFoundError(f"NHDR not found: {nhdr_path}")

    fields: Dict[str, str] = {}
    with open(nhdr_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.upper().startswith("NRRD"):
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                fields[key.strip().lower()] = value.strip()

    # dtype
    raw_type = fields.get("type", "")
    dtype_str = _NRRD_DTYPE_MAP.get(raw_type.lower())
    if dtype_str is None:
        raise ValueError(f"Unknown NRRD type '{raw_type}' in {nhdr_path}")

    # sizes  — NRRD stores as X Y Z
    sizes_str = fields.get("sizes", "")
    if not sizes_str:
        raise ValueError(f"'sizes' field missing from {nhdr_path}")
    raw_sizes: Tuple[int, ...] = tuple(int(s) for s in sizes_str.split())
    dimension = int(fields.get("dimension", len(raw_sizes)))

    # Reverse to NumPy C-order: (Z, Y, X)
    numpy_shape: Tuple[int, ...] = tuple(reversed(raw_sizes))

    return {
        "raw_sizes": raw_sizes,
        "numpy_shape": numpy_shape,
        "dtype": dtype_str,
        "encoding": fields.get("encoding", "raw").lower(),
        "data_file": fields.get("data file", "") or fields.get("datafile", ""),
        "dimension": dimension,
    }


# ---------------------------------------------------------------------------
# Background Task: Conversion Worker
# ---------------------------------------------------------------------------


def _run_conversion(filename: str) -> None:
    """
    Background task that invokes raw_converter.py via subprocess.

    - Input  path: RAW_FILES_DIR / filename
    - Output path: OUTPUT_ZARR_DIR / <stem>.zarr
    - Volume params (shape, dtype, chunks) are read from the paired .nhdr
      file by raw_converter.py automatically — no hardcoded values here.

    Runs in FastAPI's thread-pool so it never blocks the async event loop.
    stdout/stderr from raw_converter.py is forwarded to the pipeline_api logger.
    """
    input_path = os.path.join(RAW_FILES_DIR, filename)
    stem = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_ZARR_DIR, f"{stem}.zarr")
    nhdr_path = os.path.join(RAW_FILES_DIR, f"{stem}.nhdr")

    nhdr_found = os.path.isfile(nhdr_path)
    logger.info(
        "[CONVERSION] ▶  Starting: %s\n"
        "  input   → %s\n"
        "  output  → %s\n"
        "  nhdr    → %s (%s)",
        filename,
        input_path,
        output_path,
        nhdr_path,
        "found ✅"
        if nhdr_found
        else "NOT FOUND ⚠️  — raw_converter will require --shape/--dtype",
    )

    os.makedirs(OUTPUT_ZARR_DIR, exist_ok=True)

    try:
        # raw_converter.py auto-discovers and parses the .nhdr file;
        # we only need to pass --input, --output and --levels.
        cmd = [
            "python3",
            CONVERSION_SCRIPT,
            "--input",
            input_path,
            "--output",
            output_path,
            "--levels",
            str(PYRAMID_LEVELS),
        ]

        logger.info("[CONVERSION] Command: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # ⚠️  Increase for very large volumes
            check=False,
        )

        if result.returncode == 0:
            logger.info(
                "[CONVERSION] ✅ Completed: %s\n  stdout: %s",
                filename,
                result.stdout.strip() or "(no output)",
            )
        else:
            logger.error(
                "[CONVERSION] ❌ Failed: %s (exit %d)\n  stdout: %s\n  stderr: %s",
                filename,
                result.returncode,
                result.stdout.strip() or "(none)",
                result.stderr.strip() or "(none)",
            )

    except subprocess.TimeoutExpired:
        logger.error("[CONVERSION] ⏰ Timed out: %s", filename)
    except FileNotFoundError:
        logger.error(
            "[CONVERSION] ❌ Script not found: '%s'. Check CONVERSION_SCRIPT in main.py.",
            CONVERSION_SCRIPT,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("[CONVERSION] ❌ Unexpected error for %s: %s", filename, exc)


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------


@app.get(
    "/api/files/raw",
    response_model=RawFilesResponse,
    summary="List raw (unconverted) files",
    tags=["Files"],
)
async def list_raw_files() -> RawFilesResponse:
    """
    Fetches the directory listing from the **Raw Data Server** and returns
    a list of filenames available for conversion.

    - **Returns**: `{"raw_files": ["file1.dat", ...]}`
    - **Errors**: `502` if the raw server is unreachable.
    """
    files = await _fetch_file_list(RAW_SERVER_URL)
    logger.info("Fetched %d raw file(s) from %s", len(files), RAW_SERVER_URL)
    return RawFilesResponse(raw_files=files)


@app.get(
    "/api/files/converted",
    response_model=ConvertedFilesResponse,
    summary="List converted files",
    tags=["Files"],
)
async def list_converted_files() -> ConvertedFilesResponse:
    """
    Fetches the directory listing from the **Converted Data Server** and
    returns a list of already-converted filenames.

    - **Returns**: `{"converted_files": ["file1_converted.txt", ...]}`
    - **Errors**: `502` if the converted server is unreachable.
    """
    files = await _fetch_file_list(CONVERTED_SERVER_URL)
    logger.info(
        "Fetched %d converted file(s) from %s", len(files), CONVERTED_SERVER_URL
    )
    return ConvertedFilesResponse(converted_files=files)


@app.get(
    "/api/files/status",
    response_model=List[FileStatusEntry],
    summary="Cross-server file status overview",
    tags=["Files"],
)
async def file_status() -> List[FileStatusEntry]:
    """
    Combines listings from both servers and returns the conversion status of
    every raw file.

    Each entry reports whether a raw file has a corresponding entry on the
    converted server.

    ⚠️  The matching logic currently checks for an **exact filename match**
        between raw and converted servers. If your conversion script renames
        files (e.g., appending `_converted`), update the `is_converted` check
        below to use your actual naming convention.

    - **Returns**: Array of `{filename, status, converted_url?}` objects.
    - **Errors**: `502` if either server is unreachable.
    """
    # Fetch both listings concurrently
    import asyncio

    raw_files, converted_files = await asyncio.gather(
        _fetch_file_list(RAW_SERVER_URL),
        _fetch_file_list(CONVERTED_SERVER_URL),
    )

    # Build a set for O(1) lookup
    converted_set = set(converted_files)

    results: List[FileStatusEntry] = []
    for filename in raw_files:
        # ⚠️  UPDATE this condition if your converted filenames differ from raw
        #     filenames (e.g., is_converted = f"{filename}.converted" in converted_set).
        is_converted = filename in converted_set

        results.append(
            FileStatusEntry(
                filename=filename,
                status="converted" if is_converted else "pending_conversion",
                converted_url=(
                    f"{CONVERTED_SERVER_URL.rstrip('/')}/{filename}"
                    if is_converted
                    else None
                ),
            )
        )

    converted_count = sum(1 for r in results if r.status == "converted")
    pending_count = len(results) - converted_count
    logger.info(
        "Status overview: %d total | %d converted | %d pending",
        len(results),
        converted_count,
        pending_count,
    )

    return results


@app.get(
    "/api/files/raw/{filename}",
    response_model=NhdrMetadata,
    summary="Read NHDR metadata for a raw file",
    tags=["Files"],
)
async def get_file_metadata(
    filename: str = Path(
        ...,
        description="The .raw filename whose paired .nhdr will be parsed.",
        examples=["142_2dwarp_img_4mpp_new.raw"],
    ),
) -> NhdrMetadata:
    """
    Reads and returns the parsed NRRD metadata from the `.nhdr` file paired
    with the given `.raw` filename (same directory, same stem).

    Useful for verifying what shape/dtype will be used before triggering a
    conversion.

    - **Returns**: `NhdrMetadata` object with parsed volume parameters.
    - **Errors**: `404` if no `.nhdr` file is found; `422` if it cannot be parsed.
    """
    stem = os.path.splitext(filename)[0]
    nhdr_path = os.path.join(RAW_FILES_DIR, f"{stem}.nhdr")

    if not os.path.isfile(nhdr_path):
        raise HTTPException(
            status_code=404,
            detail=f"No .nhdr file found for '{filename}' at {nhdr_path}",
        )

    try:
        meta = _parse_nhdr_file(nhdr_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    logger.info(
        "Metadata served for %s: shape=%s dtype=%s",
        filename,
        meta["numpy_shape"],
        meta["dtype"],
    )

    return NhdrMetadata(
        filename=filename,
        nhdr_file=nhdr_path,
        raw_sizes=list(meta["raw_sizes"]),
        numpy_shape=list(meta["numpy_shape"]),
        dtype=meta["dtype"],
        encoding=meta["encoding"],
        dimension=meta["dimension"],
        data_file=meta["data_file"] or None,
    )


@app.post(
    "/api/convert/{filename}",
    response_model=ConversionTriggerResponse,
    status_code=202,
    summary="Trigger conversion for a specific file",
    tags=["Conversion"],
)
async def trigger_conversion(
    background_tasks: BackgroundTasks,
    filename: str = Path(
        ...,
        description="The exact .raw filename from the raw server to convert.",
        examples=["142_2dwarp_img_4mpp_new.raw"],
    ),
) -> ConversionTriggerResponse:
    """
    Enqueues a **background conversion job** for the specified `.raw` filename.

    The endpoint returns immediately (`202 Accepted`) while `raw_converter.py`
    runs in the background. Volume parameters (shape, dtype, chunks) are
    auto-read from the paired `.nhdr` file — no manual config required.

    - **Path param**: `filename` — must be a `.raw` file on the raw server.
    - **Returns**: Acknowledgement + resolved input/output paths.

    > **Tip**: Call `GET /api/files/raw/{filename}` first to verify the
    > NHDR is readable and the volume parameters look correct.
    """
    stem = os.path.splitext(filename)[0]
    input_path = os.path.join(RAW_FILES_DIR, filename)
    output_path = os.path.join(OUTPUT_ZARR_DIR, f"{stem}.zarr")
    nhdr_path = os.path.join(RAW_FILES_DIR, f"{stem}.nhdr")
    nhdr_found = os.path.isfile(nhdr_path)

    if not nhdr_found:
        logger.warning(
            "No .nhdr found for '%s' at %s — conversion may fail without shape/dtype.",
            filename,
            nhdr_path,
        )

    logger.info("Conversion requested for: %s  (nhdr_found=%s)", filename, nhdr_found)

    # Enqueue background task — returns immediately
    background_tasks.add_task(_run_conversion, filename)

    return ConversionTriggerResponse(
        message=f"Conversion started for {filename}",
        status="processing",
        filename=filename,
        nhdr_found=nhdr_found,
        input_path=input_path,
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# Health / Root Endpoints
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root():
    """Serve the skeuomorphic HTML control-panel homepage."""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.isfile(html_path):
        return FileResponse(html_path, media_type="text/html")
    return {
        "service": "NG Data Conversion Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Health"], summary="Health check")
async def health_check():
    """Simple liveness probe. Returns 200 OK when the service is running."""
    return {"status": "healthy"}
