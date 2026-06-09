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
# ---------------------------------------------------------------------------

# Absolute path to the conversion script (same directory as main.py)
CONVERSION_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "raw_converter.py"
)

# Directory where converted .zarr stores will be written locally.
# ⚠️  Update this to your desired output path on this machine.
OUTPUT_ZARR_DIR = "/apps/workspace/Data_NG_Converted"

# Default number of OME-Zarr downsampling pyramid levels.
PYRAMID_LEVELS = 4

# Request timeout for fetching remote directory listings (seconds).
# Note: .raw file downloads use a separate streaming client with no timeout.
HTTP_TIMEOUT = 10.0

# ---------------------------------------------------------------------------
# NRRD dtype string → NumPy dtype  (mirrors raw_converter.py for API use)
# ---------------------------------------------------------------------------
_NRRD_DTYPE_MAP: Dict[str, str] = {
    "unsigned char": "uint8",   "uchar": "uint8",       "uint8": "uint8",
    "unsigned short": "uint16", "ushort": "uint16",     "uint16": "uint16",
    "unsigned int": "uint32",   "uint": "uint32",        "uint32": "uint32",
    "unsigned long long": "uint64",                      "uint64": "uint64",
    "signed char": "int8",      "int8": "int8",
    "short": "int16",           "int16": "int16",
    "int": "int32",             "int32": "int32",
    "long long": "int64",       "int64": "int64",
    "float": "float32",         "double": "float64",
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
    raw_files: List[str] = Field(..., description="List of filenames on the raw data server.")


class ConvertedFilesResponse(BaseModel):
    """Response model for GET /api/files/converted"""
    converted_files: List[str] = Field(..., description="List of filenames on the converted data server.")


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
    filename:    str            = Field(..., description="The .raw filename this metadata belongs to.")
    nhdr_file:   str            = Field(..., description="Absolute path to the .nhdr file that was parsed.")
    raw_sizes:   List[int]      = Field(..., description="Original NRRD sizes order: [X, Y, Z].")
    numpy_shape: List[int]      = Field(..., description="NumPy / C-order shape: [Z, Y, X].")
    dtype:       str            = Field(..., description="NumPy dtype string (e.g. 'uint8').")
    encoding:    str            = Field(..., description="NRRD encoding field (e.g. 'raw', 'gzip').")
    dimension:   int            = Field(..., description="Number of spatial dimensions.")
    data_file:   Optional[str]  = Field(None, description="'data file' field from the header if present.")


class ConversionTriggerResponse(BaseModel):
    """Response model for POST /api/convert/{filename}"""
    message:    str
    status:     str
    filename:   str
    nhdr_found: bool = Field(..., description="Whether a paired .nhdr file was found and will be used.")
    input_path: str  = Field(..., description="Absolute path to the .raw file being converted.")
    output_path: str = Field(..., description="Absolute path where the .zarr store will be written.")


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _parse_directory_listing(html: str) -> List[str]:
    """
    Parse a standard HTTP directory listing page and return a list of filenames
    AND directory names (e.g. .zarr folders on the converted server).

    Works with:
      - Python's built-in http.server / SimpleHTTPServer
      - Nginx autoindex
      - Apache mod_autoindex

    Both files and directories are returned. Directory hrefs ending in '/'
    are included with the trailing slash stripped so that .zarr folder names
    appear cleanly (e.g. 'volume.zarr' not 'volume.zarr/').

    Args:
        html: Raw HTML string of the directory listing page.

    Returns:
        List of entry names (files and directories), excluding navigation links.
    """
    soup = BeautifulSoup(html, "html.parser")
    entries: List[str] = []

    for anchor in soup.find_all("a", href=True):
        href: str = anchor["href"]

        # Skip sorting/query links (e.g. ?C=N&O=D), root link, parent dir link,
        # and absolute URLs that point elsewhere.
        if (
            href.startswith("?")
            or href == "/"
            or href in ("../", "..")
            or href.startswith("http")
        ):
            continue

        # Strip trailing slash from directory entries so 'volume.zarr/' → 'volume.zarr'
        name = href.rstrip("/")
        if name:  # guard against empty strings
            entries.append(name)

    return entries


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
                f"File server at {server_url} returned HTTP "
                f"{exc.response.status_code}."
            ),
        )

    return _parse_directory_listing(response.text)


# ---------------------------------------------------------------------------
# NHDR / NRRD Header Parser  (used by API endpoints and background tasks)
# ---------------------------------------------------------------------------


def _parse_nhdr_content(text: str, source: str = "<string>") -> Dict[str, Any]:
    """
    Parse NHDR field text (already loaded into memory) and return a metadata dict:

        {
            "raw_sizes":   (X, Y, Z)  — original NRRD order,
            "numpy_shape": (Z, Y, X)  — C-order for NumPy,
            "dtype":       str,
            "encoding":    str,
            "data_file":   str,
            "dimension":   int,
        }

    Args:
        text:   Full text content of the .nhdr file.
        source: Label used in error messages (e.g. the URL or path).

    Raises:
        ValueError: Required fields missing or dtype unrecognised.
    """
    fields: Dict[str, str] = {}
    for line in text.splitlines():
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
        raise ValueError(f"Unknown NRRD type '{raw_type}' in {source}")

    # sizes — NRRD stores as X Y Z (fastest-axis first)
    sizes_str = fields.get("sizes", "")
    if not sizes_str:
        raise ValueError(f"'sizes' field missing from {source}")
    raw_sizes: Tuple[int, ...] = tuple(int(s) for s in sizes_str.split())
    dimension = int(fields.get("dimension", len(raw_sizes)))

    # Reverse to NumPy C-order: (Z, Y, X)
    numpy_shape: Tuple[int, ...] = tuple(reversed(raw_sizes))

    return {
        "raw_sizes":   raw_sizes,
        "numpy_shape": numpy_shape,
        "dtype":       dtype_str,
        "encoding":    fields.get("encoding", "raw").lower(),
        "data_file":   fields.get("data file", "") or fields.get("datafile", ""),
        "dimension":   dimension,
    }


def _parse_nhdr_file(nhdr_path: str) -> Dict[str, Any]:
    """Convenience wrapper: read a local .nhdr file then call _parse_nhdr_content."""
    if not os.path.isfile(nhdr_path):
        raise FileNotFoundError(f"NHDR not found: {nhdr_path}")
    with open(nhdr_path, "r", encoding="utf-8", errors="replace") as fh:
        return _parse_nhdr_content(fh.read(), source=nhdr_path)


# ---------------------------------------------------------------------------
# Background Task: Conversion Worker
# ---------------------------------------------------------------------------


def _run_conversion(filename: str) -> None:
    """
    Background task that:
      1. Downloads the .nhdr sidecar from the raw HTTP server (small, fast)
      2. Streams the .raw binary from the raw HTTP server to a temp directory
      3. Runs raw_converter.py on the temp files
      4. Cleans up the temp directory (raw + nhdr) after conversion
         — the output .zarr in OUTPUT_ZARR_DIR is kept.

    Runs in FastAPI's thread-pool so it never blocks the async event loop.
    """
    import shutil
    import tempfile

    stem        = os.path.splitext(filename)[0]
    raw_url     = f"{RAW_SERVER_URL.rstrip('/')}/{filename}"
    nhdr_url    = f"{RAW_SERVER_URL.rstrip('/')}/{stem}.nhdr"
    output_path = os.path.join(OUTPUT_ZARR_DIR, f"{stem}.zarr")

    logger.info(
        "[CONVERSION] ▶  Starting: %s\n"
        "  source  → %s\n"
        "  output  → %s",
        filename, raw_url, output_path,
    )

    tmp_dir = tempfile.mkdtemp(prefix="ng_convert_")
    tmp_raw  = os.path.join(tmp_dir, filename)
    tmp_nhdr = os.path.join(tmp_dir, f"{stem}.nhdr")

    try:
        # ── Step 1: Download .nhdr (small header file) ──────────────────────
        logger.info("[CONVERSION] Downloading NHDR: %s", nhdr_url)
        with httpx.Client(timeout=30.0) as client:
            r = client.get(nhdr_url)
            if r.status_code == 200:
                with open(tmp_nhdr, "wb") as f:
                    f.write(r.content)
                logger.info("[CONVERSION] NHDR saved → %s", tmp_nhdr)
            else:
                logger.warning(
                    "[CONVERSION] NHDR not found on server (HTTP %d) "
                    "— raw_converter will exit unless --shape/--dtype are passed.",
                    r.status_code,
                )

        # ── Step 2: Stream .raw file to temp dir ────────────────────────────
        logger.info("[CONVERSION] Streaming .raw from server: %s", raw_url)
        with httpx.Client(timeout=None) as client:       # no timeout — large files
            with client.stream("GET", raw_url) as r:
                r.raise_for_status()
                content_length = int(r.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 8 * 1024 * 1024             # 8 MB chunks
                with open(tmp_raw, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if content_length:
                            pct = downloaded / content_length * 100
                            logger.info(
                                "[CONVERSION] Download %.1f%%  (%d / %d bytes)",
                                pct, downloaded, content_length,
                            )
        logger.info("[CONVERSION] .raw saved → %s  (%d bytes)", tmp_raw, downloaded)

        # ── Step 3: Run conversion ───────────────────────────────────────────
        # raw_converter.py auto-discovers the .nhdr next to the .raw in tmp_dir
        os.makedirs(OUTPUT_ZARR_DIR, exist_ok=True)
        cmd = [
            "python3", CONVERSION_SCRIPT,
            "--input",  tmp_raw,
            "--output", output_path,
            "--levels", str(PYRAMID_LEVELS),
        ]
        logger.info("[CONVERSION] Command: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,       # 2 h — increase for very large volumes
            check=False,
        )

        if result.returncode == 0:
            logger.info(
                "[CONVERSION] ✅ Completed: %s → %s\n  stdout: %s",
                filename, output_path,
                result.stdout.strip() or "(no output)",
            )
        else:
            logger.error(
                "[CONVERSION] ❌ Failed: %s (exit %d)\n  stdout: %s\n  stderr: %s",
                filename, result.returncode,
                result.stdout.strip() or "(none)",
                result.stderr.strip() or "(none)",
            )

    except httpx.RequestError as exc:
        logger.error("[CONVERSION] ❌ Network error downloading %s: %s", filename, exc)
    except subprocess.TimeoutExpired:
        logger.error("[CONVERSION] ⏰ Timed out converting: %s", filename)
    except FileNotFoundError:
        logger.error(
            "[CONVERSION] ❌ Script not found: '%s'. Check CONVERSION_SCRIPT in main.py.",
            CONVERSION_SCRIPT,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("[CONVERSION] ❌ Unexpected error for %s: %s", filename, exc)
    finally:
        # ── Step 4: Clean up temp dir (raw + nhdr) — keep the .zarr output ──
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("[CONVERSION] Temp dir cleaned up: %s", tmp_dir)


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
    # Fetch both listings concurrently (includes folders like .zarr dirs)
    import asyncio
    raw_files, converted_files = await asyncio.gather(
        _fetch_file_list(RAW_SERVER_URL),
        _fetch_file_list(CONVERTED_SERVER_URL),
    )

    # Build a stem→entry map for the converted server so we can match
    # 'file.raw' against 'file.zarr', 'file.raw', or any other converted name
    # that shares the same stem (filename without extension).
    converted_stem_map: Dict[str, str] = {}
    for entry in converted_files:
        stem = os.path.splitext(entry)[0]   # 'volume.zarr' → 'volume'
        converted_stem_map[stem] = entry    # last one wins if duplicates

    results: List[FileStatusEntry] = []
    for filename in raw_files:
        raw_stem = os.path.splitext(filename)[0]   # 'file.raw' → 'file'

        # Match by exact full name first, then by stem (handles .raw ↔ .zarr)
        if filename in set(converted_files):
            matched_entry = filename
            is_converted = True
        elif raw_stem in converted_stem_map:
            matched_entry = converted_stem_map[raw_stem]
            is_converted = True
        else:
            matched_entry = None
            is_converted = False

        results.append(
            FileStatusEntry(
                filename=filename,
                status="converted" if is_converted else "pending_conversion",
                converted_url=(
                    f"{CONVERTED_SERVER_URL.rstrip('/')}/{matched_entry}"
                    if is_converted
                    else None
                ),
            )
        )

    converted_count = sum(1 for r in results if r.status == "converted")
    pending_count = len(results) - converted_count
    logger.info(
        "Status overview: %d total | %d converted | %d pending",
        len(results), converted_count, pending_count,
    )

    return results


@app.get(
    "/api/files/metadata/{filename}",
    response_model=NhdrMetadata,
    summary="Read NHDR metadata for a raw file",
    tags=["Files"],
)
async def get_file_metadata(
    filename: str = Path(
        ...,
        description="The .raw filename whose paired .nhdr will be fetched from the raw server.",
        examples=["142_2dwarp_img_4mpp_new.raw"],
    ),
) -> NhdrMetadata:
    """
    Fetches the `.nhdr` sidecar file from the **Raw Data Server** and returns
    its parsed volume metadata (shape, dtype, encoding, etc.).

    No local filesystem access required — reads directly from the HTTP server.

    - **Returns**: `NhdrMetadata` with parsed volume parameters.
    - **Errors**: `404` if no `.nhdr` exists on the server; `422` if malformed.
    """
    stem     = os.path.splitext(filename)[0]
    nhdr_url = f"{RAW_SERVER_URL.rstrip('/')}/{stem}.nhdr"

    # Fetch .nhdr text from the raw server
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.get(nhdr_url)
            if r.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"No .nhdr found for '{filename}' on raw server at {nhdr_url}",
                )
            r.raise_for_status()
            nhdr_text = r.text
    except HTTPException:
        raise
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot reach raw server to fetch NHDR: {exc}",
        )

    # Parse in-memory (no temp file needed for the small header)
    try:
        meta = _parse_nhdr_content(nhdr_text, source=nhdr_url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    logger.info(
        "Metadata served for %s: shape=%s dtype=%s (source: %s)",
        filename, meta["numpy_shape"], meta["dtype"], nhdr_url,
    )

    return NhdrMetadata(
        filename    = filename,
        nhdr_file   = nhdr_url,          # URL not a local path
        raw_sizes   = list(meta["raw_sizes"]),
        numpy_shape = list(meta["numpy_shape"]),
        dtype       = meta["dtype"],
        encoding    = meta["encoding"],
        dimension   = meta["dimension"],
        data_file   = meta["data_file"] or None,
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

    The endpoint returns immediately (`202 Accepted`). The background task will:
    1. Download the `.nhdr` sidecar from the raw server
    2. Stream the `.raw` file from the raw server into a temp directory
    3. Run `raw_converter.py` (shape/dtype auto-read from `.nhdr`)
    4. Write the output `.zarr` to `OUTPUT_ZARR_DIR`
    5. Delete the temp files

    - **Path param**: `filename` — must be a `.raw` file listed on the raw server.
    - **Returns**: Acknowledgement + source URL + output path.
    """
    stem       = os.path.splitext(filename)[0]
    raw_url    = f"{RAW_SERVER_URL.rstrip('/')}/{filename}"
    nhdr_url   = f"{RAW_SERVER_URL.rstrip('/')}/{stem}.nhdr"
    output_path = os.path.join(OUTPUT_ZARR_DIR, f"{stem}.zarr")

    # Quick HEAD check to see if the .nhdr exists on the server
    nhdr_found = False
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.head(nhdr_url)
            nhdr_found = r.status_code == 200
    except httpx.RequestError:
        pass  # server may not support HEAD — conversion will find out at download time

    if not nhdr_found:
        logger.warning(
            "NHDR not found on server for '%s' (%s) — conversion may fail.",
            filename, nhdr_url,
        )

    logger.info(
        "Conversion queued: %s  nhdr_found=%s  output→%s",
        filename, nhdr_found, output_path,
    )

    background_tasks.add_task(_run_conversion, filename)

    return ConversionTriggerResponse(
        message     = f"Conversion started for {filename}",
        status      = "processing",
        filename    = filename,
        nhdr_found  = nhdr_found,
        input_path  = raw_url,       # URL on the raw server (not a local path)
        output_path = output_path,
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
