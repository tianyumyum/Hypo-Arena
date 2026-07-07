"""Assemble the Runner input for the Context Forge: text prompt + (optional) source file.

Large PDFs (≥ inline cap) are split via Ghostscript into roughly equal page-range chunks
and passed as multiple input_file parts. Falls back to halving and finally 300 DPI image
downsampling if a chunk still exceeds the cap.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from basics import SourceRecord
from basics.paths import source_dir

logger = logging.getLogger("hypo.construction.source")

_MAX_INLINE_BYTES = 14_500_000


def _resolve_source_file(record: SourceRecord) -> Path | None:
    """Resolve the SourceRecord.file to an absolute path under benchmark/<domain>/source/."""
    if not record.file:
        return None
    return source_dir(record.domain) / record.file


def _encode_part(*, data: bytes, filename: str, mime: str) -> dict[str, str]:
    """Build one input_file content part from raw bytes."""
    encoded = base64.b64encode(data).decode("utf-8")
    return {
        "type": "input_file",
        "filename": filename,
        "file_data": f"data:{mime};base64,{encoded}",
    }


def _encode_file(path: Path) -> list[dict[str, str]]:
    """Encode a source file as one or more input_file parts; auto-splits oversize PDFs."""
    mime = mimetypes.guess_type(path.name)[0] or "application/pdf"
    size = path.stat().st_size
    if size < _MAX_INLINE_BYTES:
        return [_encode_part(data=path.read_bytes(), filename=path.name, mime=mime)]
    if path.suffix.lower() != ".pdf":
        raise NotImplementedError(
            f"Source file {path} ({size} bytes) exceeds inline cap "
            f"{_MAX_INLINE_BYTES}; only PDFs support auto-chunking."
        )
    return _split_pdf(path, mime=mime)


def _split_pdf(path: Path, *, mime: str) -> list[dict[str, str]]:
    """Split an oversize PDF into page-range chunks each under the inline cap."""
    total_pages = _gs_page_count(path)
    file_size = path.stat().st_size
    pages_per_chunk = max(1, int(total_pages * _MAX_INLINE_BYTES / file_size))
    logger.info(
        "pdf.split path=%s size=%d pages=%d pages_per_chunk=%d",
        path.name, file_size, total_pages, pages_per_chunk,
    )

    parts: list[dict[str, str]] = []
    start = 1
    while start <= total_pages:
        end = min(start + pages_per_chunk - 1, total_pages)
        chunk_bytes = _gs_extract_pages(path, start, end)
        # Halve the range until under the threshold (still original-resolution pages).
        while len(chunk_bytes) >= _MAX_INLINE_BYTES and end > start:
            end = start + (end - start) // 2
            chunk_bytes = _gs_extract_pages(path, start, end)
        # Last resort: downsample images on the same range.
        if len(chunk_bytes) >= _MAX_INLINE_BYTES:
            chunk_bytes = _gs_extract_pages(path, start, end, downsample=300)
            if len(chunk_bytes) >= _MAX_INLINE_BYTES:
                raise RuntimeError(
                    f"PDF chunk {path.name} pages {start}-{end} still exceeds "
                    f"{_MAX_INLINE_BYTES} bytes after 300 DPI downsample "
                    f"({len(chunk_bytes)} bytes)."
                )
        parts.append(_encode_part(
            data=chunk_bytes,
            filename=f"{path.stem}_p{start}-{end}.pdf",
            mime=mime,
        ))
        start = end + 1

    logger.info("pdf.split path=%s chunks=%d", path.name, len(parts))
    return parts


def _gs_page_count(path: Path) -> int:
    """Count PDF pages via Ghostscript (no Python deps)."""
    ps_path = str(path).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    result = subprocess.run(
        ["gs", "-q", "-dNODISPLAY", "-dNOSAFER",
         "-c", f"({ps_path}) (r) file runpdfbegin pdfpagecount = quit"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gs page count failed for {path}: {result.stderr.strip()}")
    return int(result.stdout.strip())


def _gs_extract_pages(path: Path, first: int, last: int, *, downsample: int = 0) -> bytes:
    """Extract pages [first..last] as a new PDF; optional image downsample for size reduction."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "gs", "-sDEVICE=pdfwrite", "-dNOPAUSE", "-dBATCH", "-dQUIET",
            f"-dFirstPage={first}", f"-dLastPage={last}",
        ]
        if downsample:
            cmd += [
                "-dDownsampleColorImages=true", f"-dColorImageResolution={downsample}",
                "-dDownsampleGrayImages=true", f"-dGrayImageResolution={downsample}",
                "-dDownsampleMonoImages=true", f"-dMonoImageResolution={downsample}",
            ]
        cmd += [f"-sOutputFile={tmp_path}", str(path)]
        subprocess.run(cmd, capture_output=True, timeout=120, check=True)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def assemble_forge_input(text: str, record: SourceRecord) -> str | list[dict[str, Any]]:
    """Wrap a text prompt and (when present) the source file into one user message."""
    file_path = _resolve_source_file(record)
    if file_path is None or not file_path.exists():
        return text
    content: list[dict[str, Any]] = list(_encode_file(file_path))
    content.append({"type": "input_text", "text": text})
    return [{"role": "user", "content": content}]
