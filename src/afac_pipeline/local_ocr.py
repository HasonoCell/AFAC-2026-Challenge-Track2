"""Cross-platform local OCR backends for guarded matrix reconstruction."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .images import ImageSlice, make_grid_slices
from .vision import (
    VisionMatrixResult,
    VisionObservation,
    reconstruct_numeric_matrix_from_observations,
)


class LocalOCRError(RuntimeError):
    pass


_RAPID_OCR_LOCAL = threading.local()
_RAPIDOCR_MAX_CANDIDATES = 1000


def run_local_numeric_matrix_ocr(
    *,
    backend: str,
    slices: list[ImageSlice],
    cache_dir: Path,
    workers: int = 4,
    refine_saturated: bool = False,
    max_refine_depth: int = 1,
) -> VisionMatrixResult | None:
    """OCR grid slices locally and apply the guarded matrix reconstructor."""

    if backend == "rapidocr":
        return run_rapidocr_numeric_matrix_ocr(
            slices=slices,
            cache_dir=cache_dir,
            workers=workers,
            refine_saturated=refine_saturated,
            max_refine_depth=max_refine_depth,
        )
    if backend == "vision":
        # Kept as an explicit diagnostic backend; frozen presets use RapidOCR.
        from .vision import run_vision_numeric_matrix_ocr

        return run_vision_numeric_matrix_ocr(
            slices=slices,
            cache_dir=cache_dir,
            workers=workers,
        )
    if backend == "off":
        return None
    raise ValueError(f"unknown local OCR backend {backend!r}")


def run_rapidocr_numeric_matrix_ocr(
    *,
    slices: list[ImageSlice],
    cache_dir: Path,
    workers: int = 4,
    refine_saturated: bool = False,
    max_refine_depth: int = 1,
) -> VisionMatrixResult | None:
    """Run PP-OCRv4 through ONNX Runtime and reconstruct a numeric matrix."""

    if not slices:
        return None
    tile_cache_dir = cache_dir / "rapidocr-v4" / "tiles"
    tile_cache_dir.mkdir(parents=True, exist_ok=True)
    _ensure_root_tile_manifest(tile_cache_dir, slices)

    def read_slice(
        image_slice: ImageSlice,
        *,
        depth: int = 0,
    ) -> list[VisionObservation]:
        stem = Path(image_slice.file_name).stem
        image_path = tile_cache_dir / f"{stem}.jpg"
        tsv_path = tile_cache_dir / f"{stem}.tsv"
        if not tsv_path.exists():
            if not image_path.exists():
                image_path.write_bytes(image_slice.image_bytes)
            result = _rapidocr_engine()(str(image_path))
            lines = result[0] if isinstance(result, tuple) else result
            tsv_path.write_text(
                _serialize_rapidocr_lines(lines or []),
                encoding="utf-8",
            )
        parsed = _parse_rapidocr_tsv(
            tsv_path.read_text(encoding="utf-8"),
            image_slice,
        )
        if (
            refine_saturated
            and depth < max_refine_depth
            and len(parsed) >= _RAPIDOCR_MAX_CANDIDATES
        ):
            refined: list[VisionObservation] = []
            for child in _split_image_slice(image_slice):
                refined.extend(read_slice(child, depth=depth + 1))
            return refined
        return parsed

    observations: list[VisionObservation] = []
    worker_count = min(max(1, workers), len(slices))
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(read_slice, image_slice) for image_slice in slices]
            for future in as_completed(futures):
                observations.extend(future.result())
    except (ImportError, ModuleNotFoundError) as exc:
        raise LocalOCRError(
            "RapidOCR is unavailable; install the rapidocr-onnxruntime dependency"
        ) from exc
    except Exception as exc:
        raise LocalOCRError(str(exc)) from exc
    return reconstruct_numeric_matrix_from_observations(observations)


def _ensure_root_tile_manifest(
    tile_cache_dir: Path,
    slices: list[ImageSlice],
) -> None:
    """Bind a RapidOCR cache to the exact root-tile geometry and bytes.

    Tile filenames only contain row/column identifiers.  Reusing their TSV
    contents after changing a content-box crop shifts every OCR coordinate
    while looking like a valid cache hit.  A small root manifest makes that
    mismatch explicit; child tiles are deterministically derived from roots.
    """

    manifest_path = tile_cache_dir.parent / "root_tile_manifest.json"
    payload = {
        "version": 1,
        "tiles": [
            {
                "file_name": image_slice.file_name,
                "x0": image_slice.x0,
                "x1": image_slice.x1,
                "y0": image_slice.y0,
                "y1": image_slice.y1,
                "width": image_slice.width,
                "height": image_slice.height,
                "sha256": hashlib.sha256(image_slice.image_bytes).hexdigest(),
            }
            for image_slice in slices
        ],
    }
    if manifest_path.exists():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalOCRError(f"invalid RapidOCR tile manifest: {manifest_path}") from exc
        if cached != payload:
            raise LocalOCRError(
                "RapidOCR tile cache geometry does not match the requested slices"
            )
        return
    if any(tile_cache_dir.glob("*.tsv")):
        raise LocalOCRError(
            "RapidOCR tile cache has no geometry manifest; rerun OCR into a new cache"
        )
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _split_image_slice(image_slice: ImageSlice) -> list[ImageSlice]:
    """Split one encoded tile into 2x2 children with source-page coordinates."""

    local_children = make_grid_slices(
        file_name=image_slice.file_name,
        image_bytes=image_slice.image_bytes,
        slice_width=max(1, (image_slice.width + 1) // 2),
        slice_height=max(1, (image_slice.height + 1) // 2),
        x_overlap=0,
        y_overlap=0,
    )
    x_scale = (image_slice.x1 - image_slice.x0) / max(1, image_slice.width)
    y_scale = (image_slice.y1 - image_slice.y0) / max(1, image_slice.height)
    children: list[ImageSlice] = []
    for child in local_children:
        children.append(
            ImageSlice(
                file_name=child.file_name,
                image_bytes=child.image_bytes,
                x0=round(image_slice.x0 + child.x0 * x_scale),
                x1=round(image_slice.x0 + child.x1 * x_scale),
                y0=round(image_slice.y0 + child.y0 * y_scale),
                y1=round(image_slice.y0 + child.y1 * y_scale),
                width=child.width,
                height=child.height,
                row=child.row,
                col=child.col,
                rows=child.rows,
                cols=child.cols,
            )
        )
    return children


def _rapidocr_engine() -> Any:
    engine = getattr(_RAPID_OCR_LOCAL, "engine", None)
    if engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except (ImportError, ModuleNotFoundError) as exc:
            raise LocalOCRError(
                "RapidOCR is unavailable; install rapidocr-onnxruntime"
            ) from exc
        # One ONNX thread per worker prevents nested oversubscription when a
        # document is processed as dozens of independent image tiles.
        engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
        _RAPID_OCR_LOCAL.engine = engine
    return engine


def _serialize_rapidocr_lines(lines: list[Any]) -> str:
    serialized: list[str] = []
    for line in lines:
        if not isinstance(line, (list, tuple)) or len(line) < 3:
            continue
        quad, text, confidence = line[:3]
        try:
            xs = [float(point[0]) for point in quad]
            ys = [float(point[1]) for point in quad]
            score = float(confidence)
        except (TypeError, ValueError, IndexError):
            continue
        if not xs or not ys:
            continue
        clean_text = str(text).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        serialized.append(
            "\t".join(
                (
                    f"{(x0 + x1) / 2:.6f}",
                    f"{(y0 + y1) / 2:.6f}",
                    f"{x1 - x0:.6f}",
                    f"{y1 - y0:.6f}",
                    f"{score:.8f}",
                    clean_text,
                )
            )
        )
    return "\n".join(serialized)


def _parse_rapidocr_tsv(
    text: str,
    image_slice: ImageSlice,
) -> list[VisionObservation]:
    observations: list[VisionObservation] = []
    local_width = max(1, image_slice.width)
    local_height = max(1, image_slice.height)
    global_width = image_slice.x1 - image_slice.x0
    global_height = image_slice.y1 - image_slice.y0
    x_scale = global_width / local_width
    y_scale = global_height / local_height
    for line in text.splitlines():
        fields = line.split("\t", 5)
        if len(fields) != 6:
            continue
        try:
            x, y, width, height, confidence = map(float, fields[:5])
        except ValueError:
            continue
        observations.append(
            VisionObservation(
                x=image_slice.x0 + x * x_scale,
                y=image_slice.y0 + y * y_scale,
                width=width * x_scale,
                height=height * y_scale,
                confidence=confidence,
                text=fields[5].strip(),
            )
        )
    return observations
