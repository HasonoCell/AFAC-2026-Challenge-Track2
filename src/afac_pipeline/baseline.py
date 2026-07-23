"""Low-call-count baseline submission generation."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from .api import (
    FinixDocClient,
    FinixDocError,
    html_table_structure_issue,
    normalize_markdown_payload,
)
from .datasets import ImageRecord
from .images import (
    DEFAULT_CROP_ANCHORS,
    DocumentProfile,
    ImageSlice,
    make_anchor_crops,
    make_adaptive_content_grid_slices,
    make_content_grid_slices,
    make_grid_slices,
    make_vertical_slices,
    profile_image,
)
from .local_ocr import (
    LocalOCRError,
    run_local_numeric_matrix_ocr,
    run_rapidocr_observations,
)
from .merge import merge_sliced_markdown
from .pipeline import write_errors, write_submission
from .submission import _read_submission_rows
from .tables import html_tables_to_markdown, parse_table, retain_complete_pipe_table_rows
from .vision import render_observations_in_reading_order


BASELINE_CACHE_SCHEMA_VERSION = 12
_MAX_HTML_CELLS_PER_ROW = 256


@dataclass(frozen=True)
class BaselineConfig:
    output_csv: Path
    cache_dir: Path
    offset: int = 0
    limit: int | None = None
    crop_sizes: tuple[int, ...] = (800, 600, 500, 400)
    anchors: tuple[str, ...] = DEFAULT_CROP_ANCHORS
    jpeg_quality: int = 95
    sleep_seconds: float = 0.0
    resume: bool = True
    resume_repair_tiles: bool = True
    retries: int = 0
    retry_sleep_seconds: float = 60.0
    min_chars: int = 20
    table_repair_min_chars: int = 600
    table_repair_min_chars_per_content_pixel: float = 0.0
    table_repair_min_gain: int = 300
    table_repair_rows: int = 4
    table_repair_cols: int = 4
    table_repair_target_tile_width: int = 0
    table_repair_target_tile_height: int = 0
    # Very tall, narrow tables are harmed by splitting the content box into
    # several narrow columns: each tile may contain too little structure and
    # the vision model can repeat the full table.  A ratio-based one-column
    # route keeps this generic and independent of document names.
    table_repair_vertical_aspect_threshold: float = 1.9
    table_repair_overlap: int = 120
    table_repair_content_threshold: int = 245
    table_repair_content_scale: float = 0.04
    table_repair_content_padding: int = 200
    table_repair_min_content_pixels: int = 1000
    table_repair_min_content_ratio: float = 0.001
    table_repair_min_text_pixels: int = 0
    table_repair_header_context_height: int = 0
    table_repair_left_context_width: int = 0
    table_repair_snap_boundaries: bool = False
    table_repair_snap_x_boundaries: bool = False
    table_repair_snap_y_boundaries: bool = False
    table_repair_min_success_parts: int = 4
    table_repair_min_success_ratio: float = 0.0
    table_repair_max_calls: int = 24
    table_repair_max_failed_parts: int = 0
    table_repair_max_failed_ratio: float = 0.0
    table_repair_max_identical_parts: int = 3
    table_repair_identical_min_chars: int = 1000
    table_repair_workers: int = 1
    table_local_ocr_backend: str = "off"
    table_local_ocr_min_pixels: int = 100_000_000
    table_local_ocr_max_pixels: int = 0
    table_local_ocr_trigger_max_chars: int = 0
    table_local_ocr_workers: int = 4
    table_local_ocr_refine_saturated: bool = False
    table_local_ocr_max_refine_depth: int = 1
    table_local_ocr_max_output_bytes: int = 0
    table_anchor_max_candidates: int = 1
    table_anchor_max_attempts: int = 0
    table_mode: str = 'coverage'
    table_target_tile_width: int = 2800
    table_target_tile_height: int = 4200
    table_max_rows: int = 8
    table_max_cols: int = 10
    table_overlap_ratio: float = 0.05
    table_min_overlap: int = 80
    table_max_blocks: int = 32
    table_min_score: float = 45.0
    table_max_duplicate_line_ratio: float = 0.30
    table_hybrid_min_content_ratio: float = 0.50
    table_refine_max_depth: int = 1
    table_refine_rows: int = 2
    table_refine_cols: int = 2
    table_fragment_max_blocks: int = 1
    table_fragment_refine_cols: int = 2
    table_workers: int = 1
    long_aspect_threshold: float = 0.12
    long_slice_height: int = 12000
    long_slice_overlap: int = 400
    # A successful long-page merge can still be suspiciously sparse.  When
    # enabled by a frozen preset, rerun such pages once with smaller slices;
    # the rule is based on visible content height, never on file identity.
    long_low_confidence_char_density: float = 0.0
    long_fallback_slice_height: int = 0
    long_fallback_overlap: int = 0
    # Extremely sparse remote output on a narrow prose page is a coverage
    # failure, not a table-repair problem.  Keep local prose OCR opt-in and
    # heavily bounded: it has no structural guesses and is used only when it
    # materially restores missing text.
    long_local_ocr_backend: str = "off"
    long_local_ocr_min_pixels: int = 0
    long_local_ocr_max_width: int = 0
    long_local_ocr_trigger_char_density: float = 0.0
    long_local_ocr_slice_height: int = 2_000
    long_local_ocr_overlap: int = 80
    long_local_ocr_workers: int = 4
    long_local_ocr_min_char_density: float = 0.0
    long_local_ocr_min_gain: int = 0
    long_min_chars: int = 20
    long_min_success_ratio: float = 1.0
    long_max_failed_parts: int = 0
    on_error: str = "raise"
    errors_csv: Path | None = None
    submission_file_names: tuple[str, ...] | None = None
    missing_markdown: str = "<table></table>\n"
    # B-list has repeatedly accepted the pipe representation and rejected
    # submissions containing HTML table tags.  Keep the raw OCR cache intact,
    # but make the final pipeline boundary explicit and testable.
    table_output_format: str = "html"


@dataclass(frozen=True)
class BaselineStats:
    total_discovered: int
    processed: int
    cache_hits: int
    api_calls: int
    fallbacks: int
    template_missing: int
    output_csv: Path


@dataclass(frozen=True)
class CachedLocalMatrixRepairStats:
    """Results of an offline local-matrix re-evaluation from raw OCR cache."""

    scanned: int
    cached_records: int
    selected: int
    selected_file_names: tuple[str, ...]
    output_csv: Path


@dataclass(frozen=True)
class TableCandidateScore:
    score: float
    reason: str
    table_count: int
    row_count: int
    cell_count: int
    duplicate_line_ratio: float
    truncated: bool
    balanced_html: bool


class _RouteFailure(FinixDocError):
    def __init__(self, message: str, *, calls: int) -> None:
        super().__init__(message)
        self.calls = calls


class _RepeatedRepairOutput(_RouteFailure):
    """Fatal repair-grid failure caused by repeated large tile responses."""


class _PartialLongResult(_RouteFailure):
    """Usable partial output that must not become a completed record cache."""

    def __init__(self, markdown: str, *, calls: int, message: str) -> None:
        super().__init__(message, calls=calls)
        self.markdown = markdown


def run_baseline_submission(
    *,
    records: Iterable[ImageRecord],
    client: FinixDocClient,
    config: BaselineConfig,
) -> BaselineStats:
    if config.table_local_ocr_backend not in {"off", "rapidocr", "vision"}:
        raise ValueError(
            "table_local_ocr_backend must be one of: off, rapidocr, vision"
        )
    if config.table_local_ocr_min_pixels < 0:
        raise ValueError("table_local_ocr_min_pixels must be >= 0")
    if config.table_local_ocr_max_pixels < 0:
        raise ValueError("table_local_ocr_max_pixels must be >= 0")
    if (
        config.table_local_ocr_max_pixels
        and config.table_local_ocr_max_pixels < config.table_local_ocr_min_pixels
    ):
        raise ValueError(
            "table_local_ocr_max_pixels must be >= table_local_ocr_min_pixels"
        )
    if config.table_local_ocr_trigger_max_chars < 0:
        raise ValueError("table_local_ocr_trigger_max_chars must be >= 0")
    if config.table_local_ocr_workers < 1:
        raise ValueError("table_local_ocr_workers must be >= 1")
    if config.table_local_ocr_max_refine_depth < 0:
        raise ValueError("table_local_ocr_max_refine_depth must be >= 0")
    if config.table_local_ocr_max_output_bytes < 0:
        raise ValueError("table_local_ocr_max_output_bytes must be >= 0")
    if config.long_low_confidence_char_density < 0:
        raise ValueError("long_low_confidence_char_density must be >= 0")
    if config.long_fallback_slice_height < 0:
        raise ValueError("long_fallback_slice_height must be >= 0")
    if config.long_fallback_overlap < 0:
        raise ValueError("long_fallback_overlap must be >= 0")
    if config.long_local_ocr_backend not in {"off", "rapidocr"}:
        raise ValueError("long_local_ocr_backend must be one of: off, rapidocr")
    if config.long_local_ocr_min_pixels < 0:
        raise ValueError("long_local_ocr_min_pixels must be >= 0")
    if config.long_local_ocr_max_width < 0:
        raise ValueError("long_local_ocr_max_width must be >= 0")
    if config.long_local_ocr_trigger_char_density < 0:
        raise ValueError("long_local_ocr_trigger_char_density must be >= 0")
    if config.long_local_ocr_slice_height <= 0:
        raise ValueError("long_local_ocr_slice_height must be positive")
    if config.long_local_ocr_overlap < 0:
        raise ValueError("long_local_ocr_overlap must be >= 0")
    if config.long_local_ocr_workers < 1:
        raise ValueError("long_local_ocr_workers must be >= 1")
    if config.long_local_ocr_min_char_density < 0:
        raise ValueError("long_local_ocr_min_char_density must be >= 0")
    if config.long_local_ocr_min_gain < 0:
        raise ValueError("long_local_ocr_min_gain must be >= 0")
    if config.table_output_format not in {"html", "markdown"}:
        raise ValueError("table_output_format must be one of: html, markdown")
    all_records = list(records)
    records_after_offset = all_records[config.offset :]
    selected_records = records_after_offset[: config.limit] if config.limit else records_after_offset

    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if config.errors_csv:
        config.errors_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_by_name: dict[str, str] = {}
    errors: list[dict[str, str]] = []
    cache_hits = 0
    api_calls = 0
    fallbacks = 0

    for index, record in enumerate(selected_records, start=1):
        strategy = _record_strategy(record, config)
        strategy_cache_dir = config.cache_dir / _baseline_cache_namespace(config, strategy)
        strategy_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = _cache_path(strategy_cache_dir, record.file_name)
        markdown: str

        try:
            cache_hit = False
            if config.resume and cache_path.exists():
                cached_text = cache_path.read_text(encoding="utf-8")
                markdown = normalize_markdown_payload(cached_text)
                if strategy == "table":
                    issue = _table_candidate_issue(markdown, config)
                else:
                    issue = _candidate_issue(
                        markdown,
                        config,
                        min_chars=_candidate_min_chars(strategy, config),
                    )
                if issue:
                    print(
                        f"[{index}/{len(selected_records)}] invalid baseline cache "
                        f"{record.file_name} ({strategy}): {issue}"
                    )
                    cache_path.unlink()
                else:
                    if markdown != cached_text:
                        cache_path.write_text(markdown, encoding="utf-8")
                    cache_hits += 1
                    cache_hit = True
                    print(
                        f"[{index}/{len(selected_records)}] baseline cache hit "
                        f"{record.file_name} ({strategy})"
                    )

            if not cache_hit:
                try:
                    markdown, calls = _call_record_baseline(
                        client,
                        record,
                        config,
                        strategy=strategy,
                    )
                except _PartialLongResult as partial:
                    markdown = partial.markdown
                    calls = partial.calls
                    print(
                        f"[{index}/{len(selected_records)}] partial long result "
                        f"is output but not record-cached: {record.file_name}"
                    )
                else:
                    cache_path.write_text(markdown, encoding="utf-8")
                api_calls += calls
                if config.sleep_seconds > 0:
                    time.sleep(config.sleep_seconds)
        except Exception as exc:
            error = {
                "file_name": record.file_name,
                "source": record.source,
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(error)
            print(
                f"[{index}/{len(selected_records)}] BASELINE ERROR "
                f"{record.file_name} ({strategy}): {error['error']}"
            )
            if config.on_error == "raise":
                _write_errors_if_needed(config.errors_csv, errors)
                raise
            if config.on_error == "placeholder":
                markdown = _placeholder_markdown(record)
                fallbacks += 1
            else:
                raise ValueError(f"Unsupported baseline on_error mode: {config.on_error}")

        rows_by_name[record.file_name] = _serialize_output_tables(markdown, config)

    rows, template_missing = _build_output_rows(
        rows_by_name=rows_by_name,
        selected_records=selected_records,
        submission_file_names=config.submission_file_names,
        missing_markdown=config.missing_markdown,
    )
    write_submission(config.output_csv, rows)
    _write_errors_if_needed(config.errors_csv, errors)
    return BaselineStats(
        total_discovered=len(all_records),
        processed=len(rows),
        cache_hits=cache_hits,
        api_calls=api_calls,
        fallbacks=fallbacks + template_missing,
        template_missing=template_missing,
        output_csv=config.output_csv,
    )


def rebuild_cached_local_matrix_repairs(
    *,
    records: Iterable[ImageRecord],
    base_csv: Path,
    output_csv: Path,
    local_cache_root: Path,
    config: BaselineConfig,
) -> CachedLocalMatrixRepairStats:
    """Re-evaluate cached local OCR matrices without remote API calls.

    Local matrix reconstruction evolves independently from the expensive OCR
    pass.  A final submission cache must not force all source tiles to be
    re-requested merely because the guarded coordinate reconstructor improved.
    This helper consumes only complete, already-cached local tile TSV files
    and writes a *partial* CSV suitable for the normal checked overlay path.
    """

    if config.table_local_ocr_backend == "off":
        raise ValueError("cached local matrix rebuild requires a local OCR backend")
    if not local_cache_root.exists():
        raise ValueError(f"local OCR cache root does not exist: {local_cache_root}")

    base = _read_submission_rows(base_csv)
    selected_rows: list[dict[str, str]] = []
    scanned = 0
    cached_records = 0
    for record in records:
        scanned += 1
        if record.file_name not in base:
            continue
        if (
            config.table_local_ocr_trigger_max_chars
            and len(base[record.file_name]) > config.table_local_ocr_trigger_max_chars
        ):
            continue
        profile = _profile_record(record, config)
        pixels = profile.width * profile.height
        if pixels < config.table_local_ocr_min_pixels:
            continue
        if config.table_local_ocr_max_pixels and pixels > config.table_local_ocr_max_pixels:
            continue
        image_bytes = record.read_bytes()
        repair_rows, repair_cols = _table_repair_grid(image_bytes, config)
        slices = make_content_grid_slices(
            file_name=record.file_name,
            image_bytes=image_bytes,
            rows=repair_rows,
            cols=repair_cols,
            threshold=config.table_repair_content_threshold,
            sample_scale=config.table_repair_content_scale,
            padding=config.table_repair_content_padding,
            x_overlap=0,
            y_overlap=0,
            header_context_height=0,
            left_context_width=0,
            jpeg_quality=config.jpeg_quality,
        )
        record_cache_dir = local_cache_root / Path(record.file_name).stem
        tile_cache_dir = record_cache_dir / "rapidocr-v4" / "tiles"
        if not tile_cache_dir.exists() or not all(
            (tile_cache_dir / f"{Path(image_slice.file_name).stem}.tsv").exists()
            for image_slice in slices
        ):
            continue
        cached_records += 1
        try:
            result = run_local_numeric_matrix_ocr(
                backend=config.table_local_ocr_backend,
                slices=slices,
                cache_dir=record_cache_dir,
                workers=config.table_local_ocr_workers,
                refine_saturated=config.table_local_ocr_refine_saturated,
                max_refine_depth=config.table_local_ocr_max_refine_depth,
            )
        except (OSError, LocalOCRError) as exc:
            print(
                f"  cached local matrix rebuild unavailable {record.file_name}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if result is None:
            continue
        candidate_markdown = _local_matrix_candidate_markdown(result.markdown, config)
        issue = _table_candidate_issue(candidate_markdown, config)
        if issue is not None:
            print(
                f"  cached local matrix rebuild rejected {record.file_name}: {issue}"
            )
            continue
        base_markdown = base[record.file_name]
        if not _repair_is_better(
            base_markdown,
            candidate_markdown,
            config.table_repair_min_gain,
            max_duplicate_line_ratio=config.table_max_duplicate_line_ratio,
        ):
            continue
        selected_rows.append(
            {
                "file_name": record.file_name,
                "ground_truth": candidate_markdown,
            }
        )
        print(
            f"  selected cached local matrix rebuild {record.file_name}: "
            f"rows={result.rows} cols={result.cols} coverage={result.coverage:.3f} "
            f"chars={len(candidate_markdown)}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_submission(output_csv, selected_rows)
    return CachedLocalMatrixRepairStats(
        scanned=scanned,
        cached_records=cached_records,
        selected=len(selected_rows),
        selected_file_names=tuple(row["file_name"] for row in selected_rows),
        output_csv=output_csv,
    )


def _call_record_baseline(
    client: FinixDocClient,
    record: ImageRecord,
    config: BaselineConfig,
    *,
    strategy: str,
) -> tuple[str, int]:
    if strategy == "long":
        return _call_long_record(client, record, config)
    return _call_table_record(client, record, config)


def _should_use_table_coverage(record: ImageRecord, config: BaselineConfig) -> bool:
    """Choose tiled coverage only for dense table pages in hybrid mode."""
    if config.table_hybrid_min_content_ratio < 0:
        return True
    profile = profile_image(
        image_bytes=record.read_bytes(),
        threshold=config.table_repair_content_threshold,
        sample_scale=config.table_repair_content_scale,
    )
    return profile.content_ratio >= config.table_hybrid_min_content_ratio


def _call_table_record(
    client: FinixDocClient,
    record: ImageRecord,
    config: BaselineConfig,
) -> tuple[str, int]:
    if config.table_mode not in {'coverage', 'anchor', 'hybrid'}:
        raise ValueError('table_mode must be one of: coverage, anchor, hybrid')
    fallback_calls = 0
    use_coverage = config.table_mode == 'coverage' or (
        config.table_mode == 'hybrid' and _should_use_table_coverage(record, config)
    )
    if use_coverage:
        try:
            return _call_table_coverage_record(client, record, config)
        except _RouteFailure as exc:
            fallback_calls += exc.calls
            print(
                f'  coverage table route failed {record.file_name}: '
                f'{type(exc).__name__}: {exc}'
            )
            print(f'  falling back to anchor route {record.file_name}')
        except Exception as exc:
            print(
                f'  coverage table route failed {record.file_name}: '
                f'{type(exc).__name__}: {exc}'
            )
            print(f'  falling back to anchor route {record.file_name}')
    elif config.table_mode == 'hybrid':
        print(
            f'  hybrid table route selected anchor for {record.file_name}: '
            f'content ratio below {config.table_hybrid_min_content_ratio:.3f}'
        )
    crops = make_anchor_crops(
        file_name=record.file_name,
        image_bytes=record.read_bytes(),
        crop_sizes=config.crop_sizes,
        anchors=config.anchors,
        jpeg_quality=config.jpeg_quality,
    )
    print(
        f"baseline API {record.file_name}: trying up to {len(crops)} crops "
        f"(sizes={','.join(map(str, config.crop_sizes))})"
    )

    errors: list[str] = []
    calls = 0
    repair_attempted = False
    anchor_candidates: list[tuple[ImageSlice, str]] = []
    for attempt_index, crop in enumerate(crops, start=1):
        if (
            config.table_anchor_max_attempts > 0
            and attempt_index > config.table_anchor_max_attempts
        ):
            break
        try:
            calls += 1
            markdown = _call_with_retries(
                client,
                crop,
                config,
            )
            issue = _table_candidate_issue(markdown, config)
            if issue:
                if (
                    not repair_attempted
                    and config.table_repair_min_chars > 0
                    and _should_attempt_table_repair(record, markdown, config)
                ):
                    repair_attempted = True
                    try:
                        if config.sleep_seconds > 0:
                            time.sleep(config.sleep_seconds)
                        repaired_markdown, repair_calls = _call_table_content_grid_record(
                            client,
                            record,
                            config,
                            previous_markdown=markdown,
                        )
                        return repaired_markdown, fallback_calls + calls + repair_calls
                    except Exception as repair_exc:
                        errors.append(
                            "content grid repair after invalid crop: "
                            f"{type(repair_exc).__name__}: {repair_exc}"
                        )
                        print(
                            f"  content grid repair failed {record.file_name}: "
                            f"{type(repair_exc).__name__}: {repair_exc}"
                        )
                raise FinixDocError(issue)
            anchor_candidates.append((crop, markdown))
            if len(anchor_candidates) < config.table_anchor_max_candidates:
                continue
            break
        except Exception as exc:
            errors.append(f"{crop.file_name}: {type(exc).__name__}: {exc}")
            print(f"  crop failed {crop.file_name}: {type(exc).__name__}: {exc}")
            if config.sleep_seconds > 0:
                time.sleep(config.sleep_seconds)

    if anchor_candidates:
        selected_crop, markdown = max(
            anchor_candidates,
            key=lambda item: _score_table_candidate(item[1]).score,
        )
        print(
            f"  selected {selected_crop.file_name} "
            f"x={selected_crop.x0}:{selected_crop.x1} y={selected_crop.y0}:{selected_crop.y1} "
            f"chars={len(markdown)} candidates={len(anchor_candidates)}"
        )
        repaired = _maybe_repair_short_table(
            client=client,
            record=record,
            config=config,
            markdown=markdown,
        )
        if repaired is not None:
            repaired_markdown, repair_calls = repaired
            return repaired_markdown, fallback_calls + calls + repair_calls
        return markdown, fallback_calls + calls

    if config.table_repair_min_chars > 0 and not repair_attempted:
        try:
            repaired_markdown, repair_calls = _call_table_content_grid_record(
                client,
                record,
                config,
                previous_markdown="",
            )
            return repaired_markdown, fallback_calls + calls + repair_calls
        except Exception as exc:
            errors.append(f"content grid repair: {type(exc).__name__}: {exc}")
            print(f"  content grid repair failed {record.file_name}: {type(exc).__name__}: {exc}")

    raise FinixDocError(
        "all baseline crop candidates failed; last errors: "
        + " | ".join(errors[-3:])
    )


def _maybe_repair_short_table(
    *,
    client: FinixDocClient,
    record: ImageRecord,
    config: BaselineConfig,
    markdown: str,
) -> tuple[str, int] | None:
    # The guarded local route is a geometry-based alternative, not merely a
    # retry for short remote output.  Run it independently on eligible huge
    # pages so a superficially long but incomplete anchor cannot suppress it.
    local_repair = _maybe_local_matrix_repair(
        record=record,
        config=config,
        previous_markdown=markdown,
    )
    if local_repair is not None:
        return local_repair, 0

    if config.table_repair_min_chars <= 0:
        return None
    if not _should_attempt_table_repair(record, markdown, config):
        return None

    try:
        if config.sleep_seconds > 0:
            time.sleep(config.sleep_seconds)
        repaired_markdown, calls = _call_table_content_grid_record(
            client,
            record,
            config,
            previous_markdown=markdown,
        )
    except _RouteFailure as exc:
        print(
            f"  content grid repair failed {record.file_name}: "
            f"{type(exc).__name__}: {exc}"
        )
        return markdown, exc.calls
    except Exception as exc:
        print(
            f"  content grid repair failed {record.file_name}: "
            f"{type(exc).__name__}: {exc}"
        )
        return None
    repaired_issue = _table_candidate_issue(repaired_markdown, config)
    if repaired_issue is not None:
        print(
            f"  kept crop result after content grid repair "
            f"because repaired result is invalid: {repaired_issue}"
        )
        return markdown, calls
    if not _repair_is_better(
        markdown,
        repaired_markdown,
        config.table_repair_min_gain,
        max_duplicate_line_ratio=config.table_max_duplicate_line_ratio,
    ):
        print(
            f"  kept crop result after content grid repair "
            f"old_chars={len(markdown)} new_chars={len(repaired_markdown)}"
        )
        return markdown, calls
    return repaired_markdown, calls


def _maybe_local_matrix_repair(
    *,
    record: ImageRecord,
    config: BaselineConfig,
    previous_markdown: str,
) -> str | None:
    """Try a guarded local numeric-matrix route before remote fan-out."""

    if config.table_local_ocr_backend == "off":
        return None
    profile = _profile_record(record, config)
    pixels = profile.width * profile.height
    if pixels < config.table_local_ocr_min_pixels:
        return None
    if config.table_local_ocr_max_pixels and pixels > config.table_local_ocr_max_pixels:
        return None
    if (
        config.table_local_ocr_trigger_max_chars
        and len(previous_markdown) > config.table_local_ocr_trigger_max_chars
    ):
        return None
    image_bytes = record.read_bytes()
    repair_rows, repair_cols = _table_repair_grid(image_bytes, config)
    slices = make_content_grid_slices(
        file_name=record.file_name,
        image_bytes=image_bytes,
        rows=repair_rows,
        cols=repair_cols,
        threshold=config.table_repair_content_threshold,
        sample_scale=config.table_repair_content_scale,
        padding=config.table_repair_content_padding,
        x_overlap=0,
        y_overlap=0,
        header_context_height=0,
        left_context_width=0,
        jpeg_quality=config.jpeg_quality,
    )
    cache_dir = (
        config.cache_dir
        / _baseline_cache_namespace(config, f"local_{config.table_local_ocr_backend}_matrix")
        / Path(record.file_name).stem
    )
    try:
        result = run_local_numeric_matrix_ocr(
            backend=config.table_local_ocr_backend,
            slices=slices,
            cache_dir=cache_dir,
            workers=config.table_local_ocr_workers,
            refine_saturated=config.table_local_ocr_refine_saturated,
            max_refine_depth=config.table_local_ocr_max_refine_depth,
        )
    except (OSError, LocalOCRError) as exc:
        print(
            f"  local {config.table_local_ocr_backend} matrix route unavailable "
            f"{record.file_name}: "
            f"{type(exc).__name__}: {exc}"
        )
        return None
    if result is None:
        print(
            f"  local {config.table_local_ocr_backend} matrix route not applicable "
            f"{record.file_name}"
        )
        return None
    candidate_markdown = _local_matrix_candidate_markdown(result.markdown, config)
    issue = _table_candidate_issue(candidate_markdown, config)
    if issue is not None:
        print(
            f"  local {config.table_local_ocr_backend} matrix route rejected "
            f"{record.file_name}: {issue}"
        )
        return None
    if not _repair_is_better(
        previous_markdown,
        candidate_markdown,
        config.table_repair_min_gain,
        max_duplicate_line_ratio=config.table_max_duplicate_line_ratio,
    ):
        return None
    print(
        f"  selected local {config.table_local_ocr_backend} matrix repair "
        f"{record.file_name}: "
        f"tables={result.table_count} rows={result.rows} cols={result.cols} "
        f"coverage={result.coverage:.3f} chars={len(candidate_markdown)}"
    )
    return candidate_markdown


def _call_table_content_grid_record(
    client: FinixDocClient,
    record: ImageRecord,
    config: BaselineConfig,
    *,
    previous_markdown: str,
) -> tuple[str, int]:
    image_bytes = record.read_bytes()
    repair_rows, repair_cols = _table_repair_grid(image_bytes, config)
    slices = make_content_grid_slices(
        file_name=record.file_name,
        image_bytes=image_bytes,
        rows=repair_rows,
        cols=repair_cols,
        threshold=config.table_repair_content_threshold,
        sample_scale=config.table_repair_content_scale,
        padding=config.table_repair_content_padding,
        x_overlap=config.table_repair_overlap,
        y_overlap=config.table_repair_overlap,
        header_context_height=config.table_repair_header_context_height,
        left_context_width=config.table_repair_left_context_width,
        snap_boundaries=config.table_repair_snap_boundaries,
        snap_x_boundaries=config.table_repair_snap_x_boundaries,
        snap_y_boundaries=config.table_repair_snap_y_boundaries,
        jpeg_quality=config.jpeg_quality,
    )
    print(
        f"  trying content grid repair {record.file_name}: "
        f"{repair_rows}x{repair_cols} "
        f"threshold={config.table_repair_content_threshold} "
        f"header_ctx={config.table_repair_header_context_height} "
        f"left_ctx={config.table_repair_left_context_width}"
    )

    tile_cache_dir = (
        config.cache_dir
        / _baseline_cache_namespace(config, "repair_tiles")
        / Path(record.file_name).stem
    )
    tile_cache_dir.mkdir(parents=True, exist_ok=True)

    if config.table_repair_workers > 1:
        return _call_table_content_grid_record_parallel(
            client,
            slices,
            config,
            tile_cache_dir=tile_cache_dir,
            previous_markdown=previous_markdown,
            record_file_name=record.file_name,
        )

    parts: list[str] = []
    errors: list[str] = []
    calls = 0
    attempted_parts = 0
    failed_parts = 0
    budget_exhausted = False
    success_parts = 0
    eligible_parts = [
        not _should_skip_low_content_slice(image_slice, config)
        for image_slice in slices
    ]
    required_success_parts = max(
        config.table_repair_min_success_parts,
        math.ceil(sum(eligible_parts) * config.table_repair_min_success_ratio),
    )
    failed_part_budget = _table_repair_failed_part_budget(config, eligible_parts)
    identical_part_counts: dict[str, int] = {}
    for index, image_slice in enumerate(slices):
        if _should_skip_low_content_slice(image_slice, config):
            print(
                f"    skipped low-content {image_slice.file_name} "
                f"row={image_slice.row}/{image_slice.rows} "
                f"col={image_slice.col}/{image_slice.cols} "
                f"content_pixels={image_slice.content_pixels} "
                f"content_ratio={image_slice.content_ratio:.6f}"
            )
            parts.append("")
            continue
        tile_cache_path = _cache_path(tile_cache_dir, image_slice.file_name)
        if config.resume_repair_tiles and tile_cache_path.exists():
            cached_markdown = normalize_markdown_payload(
                tile_cache_path.read_text(encoding="utf-8")
            )
            issue = _table_candidate_issue(
                cached_markdown,
                config,
                min_chars=10,
            )
            if issue is None:
                print(
                    f"    repair cache hit {image_slice.file_name} "
                    f"chars={len(cached_markdown)}"
                )
                parts.append(cached_markdown)
                success_parts += 1
                _raise_if_repair_part_repeats(
                    cached_markdown,
                    config,
                    identical_part_counts,
                    calls=calls,
                )
                continue
            tile_cache_path.unlink()
        top_level_budget_exhausted = (
            config.table_repair_max_calls > 0
            and attempted_parts >= config.table_repair_max_calls
        )
        if top_level_budget_exhausted:
            budget_exhausted = True
            errors.append("table repair call budget exhausted")
            print(
                f"    skipped {image_slice.file_name}: table repair call budget "
                f"exhausted ({attempted_parts}/{config.table_repair_max_calls} "
                "top-level tiles)"
            )
            parts.append("")
        else:
            attempted = False
            try:
                attempted = True
                attempted_parts += 1
                markdown, slice_calls = _call_table_repair_slice(
                    client,
                    image_slice,
                    config,
                    remaining_calls=_table_repair_slice_call_limit(config),
                    cache_dir=tile_cache_dir,
                )
                calls += slice_calls
                tile_cache_path.write_text(markdown, encoding="utf-8")
                print(
                    f"    repaired {image_slice.file_name} "
                    f"row={image_slice.row}/{image_slice.rows} "
                    f"col={image_slice.col}/{image_slice.cols} "
                    f"chars={len(markdown)}"
                )
                parts.append(markdown)
                success_parts += 1
                _raise_if_repair_part_repeats(
                    markdown,
                    config,
                    identical_part_counts,
                    calls=calls,
                )
            except _RepeatedRepairOutput:
                raise
            except _RouteFailure as exc:
                calls += exc.calls
                failed_parts += 1
                errors.append(f"{image_slice.file_name}: {type(exc).__name__}: {exc}")
                print(f"    grid slice failed {image_slice.file_name}: {type(exc).__name__}: {exc}")
                parts.append("")
            except Exception as exc:
                failed_parts += 1
                errors.append(f"{image_slice.file_name}: {type(exc).__name__}: {exc}")
                print(f"    grid slice failed {image_slice.file_name}: {type(exc).__name__}: {exc}")
                parts.append("")
            finally:
                if attempted and config.sleep_seconds > 0:
                    time.sleep(config.sleep_seconds)

        if failed_part_budget is not None and failed_parts >= failed_part_budget:
            raise _RouteFailure(
                "content grid repair exceeded failed tile budget "
                f"({failed_parts}/{failed_part_budget})",
                calls=calls,
            )

        remaining_eligible_parts = sum(eligible_parts[index + 1:])
        if success_parts + remaining_eligible_parts < required_success_parts:
            raise _RouteFailure(
                "content grid repair can no longer reach its minimum usable parts "
                f"({success_parts}+{remaining_eligible_parts} < "
                f"{required_success_parts})",
                calls=calls,
            )

    if budget_exhausted:
        raise _RouteFailure(
            "content grid repair exceeded table repair call budget "
            f"({attempted_parts}/{config.table_repair_max_calls} top-level tiles)",
            calls=calls,
        )

    if success_parts < required_success_parts:
        raise _RouteFailure(
            "content grid repair produced too few usable parts "
            f"({success_parts}/{len(slices)}, required={required_success_parts}); last errors: " + " | ".join(errors[-3:]),
            calls=calls,
        )

    return _finish_table_grid_repair(
        slices=slices,
        parts=parts,
        config=config,
        previous_markdown=previous_markdown,
        calls=calls,
        success_parts=success_parts,
        record_file_name=record.file_name,
    )


def _call_table_content_grid_record_parallel(
    client: FinixDocClient,
    slices: list[ImageSlice],
    config: BaselineConfig,
    *,
    tile_cache_dir: Path,
    previous_markdown: str,
    record_file_name: str,
) -> tuple[str, int]:
    """Read independent repair tiles concurrently without changing merge order.

    ``table_repair_max_calls`` bounds the number of top-level grid tiles.  Each
    worker receives a separate, deterministic recursive budget derived from the
    configured refinement tree, so concurrency cannot make a tile exceed its
    own worst-case call count.
    """

    if config.table_repair_workers <= 1:
        raise ValueError("parallel table repair requires at least two workers")

    parts = [""] * len(slices)
    errors: list[str] = []
    calls = 0
    failed_parts = 0
    success_parts = 0
    identical_part_counts: dict[str, int] = {}
    eligible_parts = [
        not _should_skip_low_content_slice(image_slice, config)
        for image_slice in slices
    ]
    required_success_parts = max(
        config.table_repair_min_success_parts,
        math.ceil(sum(eligible_parts) * config.table_repair_min_success_ratio),
    )
    failed_part_budget = _table_repair_failed_part_budget(config, eligible_parts)
    pending: list[tuple[int, ImageSlice, Path]] = []

    for index, image_slice in enumerate(slices):
        if not eligible_parts[index]:
            print(
                f"    skipped low-content {image_slice.file_name} "
                f"row={image_slice.row}/{image_slice.rows} "
                f"col={image_slice.col}/{image_slice.cols} "
                f"content_pixels={image_slice.content_pixels} "
                f"content_ratio={image_slice.content_ratio:.6f}"
            )
            continue
        tile_cache_path = _cache_path(tile_cache_dir, image_slice.file_name)
        if config.resume_repair_tiles and tile_cache_path.exists():
            cached_markdown = normalize_markdown_payload(
                tile_cache_path.read_text(encoding="utf-8")
            )
            issue = _table_candidate_issue(cached_markdown, config, min_chars=10)
            if issue is None:
                print(
                    f"    repair cache hit {image_slice.file_name} "
                    f"chars={len(cached_markdown)}"
                )
                parts[index] = cached_markdown
                success_parts += 1
                _raise_if_repair_part_repeats(
                    cached_markdown,
                    config,
                    identical_part_counts,
                    calls=calls,
                )
                continue
            tile_cache_path.unlink()
        pending.append((index, image_slice, tile_cache_path))

    if config.table_repair_max_calls > 0 and len(pending) > config.table_repair_max_calls:
        raise _RouteFailure(
            "parallel content grid repair exceeds table repair call budget "
            f"({len(pending)}/{config.table_repair_max_calls})",
            calls=0,
        )

    def call_slice(
        item: tuple[int, ImageSlice, Path],
    ) -> tuple[int, ImageSlice, Path, str, int, Exception | None]:
        index, image_slice, tile_cache_path = item
        try:
            markdown, slice_calls = _call_table_repair_slice(
                client,
                image_slice,
                config,
                remaining_calls=_table_repair_slice_call_limit(config),
                cache_dir=tile_cache_dir,
            )
            return index, image_slice, tile_cache_path, markdown, slice_calls, None
        except Exception as exc:
            return (
                index,
                image_slice,
                tile_cache_path,
                "",
                exc.calls if isinstance(exc, _RouteFailure) else 0,
                exc,
            )
        finally:
            if config.sleep_seconds > 0:
                time.sleep(config.sleep_seconds)

    workers = min(config.table_repair_workers, max(1, len(pending)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(call_slice, item) for item in pending]
        for future in as_completed(futures):
            (
                index,
                image_slice,
                tile_cache_path,
                markdown,
                slice_calls,
                error,
            ) = future.result()
            calls += slice_calls
            if error is not None:
                failed_parts += 1
                errors.append(
                    f"{image_slice.file_name}: {type(error).__name__}: {error}"
                )
                print(
                    f"    grid slice failed {image_slice.file_name}: "
                    f"{type(error).__name__}: {error}"
                )
                continue
            # Persist as soon as a worker completes.  A later network failure
            # or interruption can then resume from this exact region.
            tile_cache_path.write_text(markdown, encoding="utf-8")
            parts[index] = markdown
            success_parts += 1
            print(
                f"    repaired {image_slice.file_name} "
                f"row={image_slice.row}/{image_slice.rows} "
                f"col={image_slice.col}/{image_slice.cols} chars={len(markdown)}"
            )
            _raise_if_repair_part_repeats(
                markdown,
                config,
                identical_part_counts,
                calls=calls,
            )

    if failed_part_budget is not None and failed_parts >= failed_part_budget:
        raise _RouteFailure(
            "content grid repair exceeded failed tile budget "
            f"({failed_parts}/{failed_part_budget})",
            calls=calls,
        )
    if success_parts < required_success_parts:
        raise _RouteFailure(
            "content grid repair produced too few usable parts "
            f"({success_parts}/{len(slices)}, required={required_success_parts}); "
            "last errors: " + " | ".join(errors[-3:]),
            calls=calls,
        )
    return _finish_table_grid_repair(
        slices=slices,
        parts=parts,
        config=config,
        previous_markdown=previous_markdown,
        calls=calls,
        success_parts=success_parts,
        record_file_name=record_file_name,
    )


def _table_repair_slice_call_limit(config: BaselineConfig) -> int:
    """Return the worst-case calls for one bounded refinement tree."""

    depth = max(0, config.table_refine_max_depth)
    child_count = max(1, config.table_refine_rows) * max(
        1,
        config.table_refine_cols,
    )
    total = 0
    nodes = 1
    for _ in range(depth + 1):
        total += nodes
        nodes *= child_count
    return total


def _finish_table_grid_repair(
    *,
    slices: list[ImageSlice],
    parts: list[str],
    config: BaselineConfig,
    previous_markdown: str,
    calls: int,
    success_parts: int,
    record_file_name: str,
) -> tuple[str, int]:
    repaired = merge_sliced_markdown(slices, parts)
    issue = _candidate_issue(repaired, config, min_chars=config.min_chars)
    if issue:
        raise _RouteFailure(
            f"content grid repair result is invalid: {issue}",
            calls=calls,
        )
    if not _repair_is_better(
        previous_markdown,
        repaired,
        config.table_repair_min_gain,
        max_duplicate_line_ratio=config.table_max_duplicate_line_ratio,
    ):
        raise _RouteFailure(
            "content grid repair did not improve output enough "
            f"(old_chars={len(previous_markdown.strip())}, "
            f"new_chars={len(repaired.strip())})",
            calls=calls,
        )
    print(
        f"  selected content grid repair {record_file_name} "
        f"success_parts={success_parts}/{len(slices)} chars={len(repaired)}"
    )
    return repaired, calls


def _call_table_repair_slice(
    client: FinixDocClient,
    image_slice: ImageSlice,
    config: BaselineConfig,
    *,
    depth: int = 0,
    remaining_calls: int | None = None,
    cache_dir: Path | None = None,
) -> tuple[str, int]:
    """Read a repair tile, subdividing only when Finix truncates it.

    Retrying the same oversized tile repeatedly does not recover omitted table
    cells.  A single bounded 2x2 refinement retains the parent tile's reading
    position while giving the service a smaller visual context.
    """
    if remaining_calls is not None and remaining_calls <= 0:
        raise _RouteFailure("table repair call budget exhausted", calls=0)

    refine_reason = "oversized"
    force_refine = _should_preemptively_refine_table_tile(
        image_slice,
        config,
        depth=depth,
    )
    calls = 0
    if not force_refine:
        calls = 1
        try:
            markdown = _call_with_retries(
                client,
                image_slice,
                config,
                allow_unclosed_fence=True,
                balance_html_tables=True,
                retry_truncation=False,
            )
            issue = _table_candidate_issue(markdown, config, min_chars=10)
            if issue:
                raise FinixDocError(issue)
            return markdown, calls
        except Exception as exc:
            if (
                not _is_refinable_table_tile_failure(exc)
                or depth >= config.table_refine_max_depth
            ):
                raise _RouteFailure(str(exc), calls=calls) from exc
            refine_reason = "truncated"

    child_slices = make_grid_slices(
        file_name=image_slice.file_name,
        image_bytes=image_slice.image_bytes,
        slice_width=max(1, math.ceil(image_slice.width / config.table_refine_cols)),
        slice_height=max(1, math.ceil(image_slice.height / config.table_refine_rows)),
        x_overlap=0,
        y_overlap=0,
        jpeg_quality=config.jpeg_quality,
    )
    if len(child_slices) <= 1:
        raise _RouteFailure('truncated repair tile cannot be subdivided', calls=calls)

    print(
        f"    refining {refine_reason} repair tile {image_slice.file_name}: "
        f"depth={depth + 1} as {child_slices[0].rows}x{child_slices[0].cols}"
    )
    child_parts: list[str] = []
    for child_slice in child_slices:
        child_cache_path = (
            _cache_path(cache_dir, child_slice.file_name)
            if cache_dir is not None
            else None
        )
        if (
            child_cache_path is not None
            and config.resume_repair_tiles
            and child_cache_path.exists()
        ):
            cached_markdown = normalize_markdown_payload(
                child_cache_path.read_text(encoding="utf-8")
            )
            issue = _table_candidate_issue(cached_markdown, config, min_chars=10)
            if issue is None:
                print(
                    f"      refinement cache hit {child_slice.file_name} "
                    f"chars={len(cached_markdown)}"
                )
                child_parts.append(cached_markdown)
                continue
            child_cache_path.unlink()
        try:
            child_markdown, child_calls = _call_table_repair_slice(
                client,
                child_slice,
                config,
                depth=depth + 1,
                remaining_calls=(
                    remaining_calls - calls if remaining_calls is not None else None
                ),
                cache_dir=cache_dir,
            )
        except _RouteFailure as child_exc:
            calls += child_exc.calls
            raise _RouteFailure(str(child_exc), calls=calls) from child_exc
        calls += child_calls
        child_parts.append(child_markdown)
        if child_cache_path is not None:
            child_cache_path.write_text(child_markdown, encoding="utf-8")

    merged = merge_sliced_markdown(child_slices, child_parts)
    issue = _table_candidate_issue(merged, config, min_chars=10)
    if issue:
        raise _RouteFailure(f'refined repair tile is invalid: {issue}', calls=calls)
    return merged, calls


def _should_preemptively_refine_table_tile(
    image_slice: ImageSlice,
    config: BaselineConfig,
    *,
    depth: int,
) -> bool:
    """Avoid known-oversized parent calls when a bounded child route exists.

    The adaptive grid may be compressed to fit the top-level call budget.  On
    extreme pages that can leave a tile several times larger than the target
    OCR context.  Calling that parent first commonly yields truncation or a
    disconnected response, so refine it directly when either axis exceeds two
    target tiles.
    """

    # Only skip the call-budget-compressed top-level parent.  At deeper levels
    # try the child first and fan out again solely when its response proves it
    # is still truncated.
    if depth != 0 or depth >= config.table_refine_max_depth:
        return False
    target_width = config.table_repair_target_tile_width
    target_height = config.table_repair_target_tile_height
    too_wide = target_width > 0 and image_slice.width > target_width * 2
    too_tall = target_height > 0 and image_slice.height > target_height * 2
    return too_wide or too_tall


# Coverage-first path for large financial tables. It tries to cover the full
# detected content box before falling back to the old anchor-crop baseline.
def _call_table_coverage_record(
    client: FinixDocClient,
    record: ImageRecord,
    config: BaselineConfig,
) -> tuple[str, int]:
    image_bytes = record.read_bytes()
    slices = make_adaptive_content_grid_slices(
        file_name=record.file_name,
        image_bytes=image_bytes,
        target_tile_width=config.table_target_tile_width,
        target_tile_height=config.table_target_tile_height,
        max_rows=config.table_max_rows,
        max_cols=config.table_max_cols,
        threshold=config.table_repair_content_threshold,
        sample_scale=config.table_repair_content_scale,
        padding=config.table_repair_content_padding,
        overlap_ratio=config.table_overlap_ratio,
        min_overlap=config.table_min_overlap,
        header_context_height=config.table_repair_header_context_height,
        left_context_width=config.table_repair_left_context_width,
        snap_boundaries=config.table_repair_snap_boundaries,
        snap_x_boundaries=config.table_repair_snap_x_boundaries,
        snap_y_boundaries=config.table_repair_snap_y_boundaries,
        jpeg_quality=config.jpeg_quality,
    )
    print(
        f'coverage table API {record.file_name}: '
        f'{slices[0].rows}x{slices[0].cols} tiles '
        f'target={config.table_target_tile_width}x{config.table_target_tile_height} '
        f'overlap_ratio={config.table_overlap_ratio:.3f}'
    )
    tile_cache_dir = (
        config.cache_dir
        / _baseline_cache_namespace(config, "coverage_tiles")
        / Path(record.file_name).stem
    )
    tile_cache_dir.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    errors: list[str] = []
    calls = 0
    def call_slice(index: int, image_slice: ImageSlice) -> tuple[int, str, int, str | None]:
        if _should_skip_low_content_slice(image_slice, config):
            return index, '', 0, 'skipped'
        tile_cache_path = _cache_path(tile_cache_dir, image_slice.file_name)
        if config.resume and tile_cache_path.exists():
            cached_markdown = normalize_markdown_payload(
                tile_cache_path.read_text(encoding="utf-8")
            )
            if not _candidate_issue(cached_markdown, config, min_chars=10):
                return index, cached_markdown, 0, None
            tile_cache_path.unlink()
        try:
            markdown, tile_calls = _call_coverage_tile(
                client,
                image_slice,
                config,
            )
            tile_cache_path.write_text(markdown, encoding="utf-8")
            return index, markdown, tile_calls, None
        except _RouteFailure as exc:
            return index, '', exc.calls, f'{type(exc).__name__}: {exc}'
        except Exception as exc:
            return index, '', 0, f'{type(exc).__name__}: {exc}'

    results: list[tuple[int, str, int, str | None]] = []
    workers = max(1, config.table_workers)
    if workers == 1:
        results = [call_slice(index, image_slice) for index, image_slice in enumerate(slices)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(call_slice, index, image_slice)
                for index, image_slice in enumerate(slices)
            ]
            results = [future.result() for future in as_completed(futures)]

    parts = [''] * len(slices)
    for index, markdown, tile_calls, error in results:
        image_slice = slices[index]
        calls += tile_calls
        parts[index] = markdown
        if error == 'skipped':
            print(
                f'    skipped low-content {image_slice.file_name} '
                f'row={image_slice.row}/{image_slice.rows} '
                f'col={image_slice.col}/{image_slice.cols} '
                f'content_pixels={image_slice.content_pixels} '
                f'content_ratio={image_slice.content_ratio:.6f}'
            )
        elif error:
            errors.append(f'{image_slice.file_name}: {error}')
            print(f'    coverage tile failed {image_slice.file_name}: {error}')
        else:
            print(
                f'    coverage tile {image_slice.file_name} '
                f'row={image_slice.row}/{image_slice.rows} '
                f'col={image_slice.col}/{image_slice.cols} chars={len(markdown)}'
            )

    success_parts = sum(bool(part.strip()) for part in parts)
    required_success_parts = _required_coverage_success_parts(slices, config)
    if success_parts < required_success_parts:
        raise _RouteFailure(
            'coverage table route produced too few usable parts '
            f'({success_parts}/{len(slices)}, required={required_success_parts}); '
            'last errors: ' + ' | '.join(errors[-3:]),
            calls=calls,
        )

    merged = merge_sliced_markdown(slices, parts)
    issue = _coverage_candidate_issue(merged, config)
    if issue:
        raise _RouteFailure(f'coverage table result is invalid: {issue}', calls=calls)
    candidate_score = _score_table_candidate(merged, parts=parts, slices=slices)
    if candidate_score.score < config.table_min_score:
        raise _RouteFailure(
            'coverage table score is too low '
            f'({candidate_score.score:.1f} < {config.table_min_score:.1f}: '
            f'{candidate_score.reason})',
            calls=calls,
        )
    print(
        f'  selected coverage table {record.file_name} score={candidate_score.score:.1f} '
        f'rows={candidate_score.row_count} cells={candidate_score.cell_count} '
        f'success_parts={success_parts}/{len(slices)} chars={len(merged)}'
    )
    return merged, calls


def _call_coverage_tile(
    client: FinixDocClient,
    image_slice: ImageSlice,
    config: BaselineConfig,
    *,
    depth: int = 0,
) -> tuple[str, int]:
    """Call one coverage tile, recursively refining only truncated responses."""

    calls = 1
    try:
        markdown = _call_with_retries(
            client,
            image_slice,
            config,
            retry_truncation=False,
        )
    except Exception as exc:
        if config.sleep_seconds > 0:
            time.sleep(config.sleep_seconds)
        if not _is_refinable_table_tile_failure(exc) or depth >= config.table_refine_max_depth:
            raise _RouteFailure(str(exc), calls=calls) from exc
        return _refine_coverage_tile(
            client,
            image_slice,
            config,
            depth=depth,
            rows=config.table_refine_rows,
            cols=config.table_refine_cols,
            reason='truncated',
            initial_calls=calls,
        )

    if config.sleep_seconds > 0:
        time.sleep(config.sleep_seconds)
    issue = _candidate_issue(markdown, config, min_chars=10)
    if issue:
        raise _RouteFailure(issue, calls=calls)
    table_blocks = markdown.lower().count('<table')
    if table_blocks > config.table_fragment_max_blocks and depth < config.table_refine_max_depth:
        return _refine_coverage_tile(
            client,
            image_slice,
            config,
            depth=depth,
            rows=1,
            cols=config.table_fragment_refine_cols,
            reason=f'fragmented ({table_blocks} HTML tables)',
            initial_calls=calls,
        )
    return markdown, calls


def _refine_coverage_tile(
    client: FinixDocClient,
    image_slice: ImageSlice,
    config: BaselineConfig,
    *,
    depth: int,
    rows: int,
    cols: int,
    reason: str,
    initial_calls: int,
) -> tuple[str, int]:
    if rows <= 0 or cols <= 0:
        raise ValueError('table refine rows and cols must be positive')
    child_slices = make_grid_slices(
        file_name=image_slice.file_name,
        image_bytes=image_slice.image_bytes,
        slice_width=max(1, math.ceil(image_slice.width / cols)),
        slice_height=max(1, math.ceil(image_slice.height / rows)),
        x_overlap=0,
        y_overlap=0,
        jpeg_quality=config.jpeg_quality,
    )
    if len(child_slices) <= 1:
        raise _RouteFailure(reason, calls=initial_calls)

    print(
        f'    refining {reason} tile {image_slice.file_name}: '
        f'depth={depth + 1} as {child_slices[0].rows}x{child_slices[0].cols}'
    )
    calls = initial_calls
    parts: list[str] = []
    for child in child_slices:
        try:
            child_markdown, child_calls = _call_coverage_tile(
                client,
                child,
                config,
                depth=depth + 1,
            )
        except _RouteFailure as child_exc:
            calls += child_exc.calls
            raise _RouteFailure(str(child_exc), calls=calls) from child_exc
        calls += child_calls
        parts.append(child_markdown)
    merged = merge_sliced_markdown(child_slices, parts)
    issue = _candidate_issue(merged, config, min_chars=10)
    if issue:
        raise _RouteFailure(f'refined tile is invalid: {issue}', calls=calls)
    return merged, calls


def _is_truncation_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "appears truncated" in message or "timed out" in message


def _is_refinable_table_tile_failure(exc: Exception) -> bool:
    """Return true only when the response itself proves the tile was oversized.

    A network timeout can occur for any tile. Treating it as a size signal caused
    recursive fan-out during transient service failures, so those failures stay
    local to the affected tile instead.
    """
    message = str(exc).lower()
    response_truncation_markers = (
        "appears truncated",
        "incomplete unstructured table text",
        "unclosed <table>",
        "unclosed <tr>",
        "unclosed <td>",
        "unclosed <th>",
        " while <",
    )
    return any(marker in message for marker in response_truncation_markers)


def _required_coverage_success_parts(
    slices: list[ImageSlice],
    config: BaselineConfig,
) -> int:
    expected_parts = sum(
        not _should_skip_low_content_slice(image_slice, config)
        for image_slice in slices
    )
    if expected_parts <= 0:
        return 1
    if expected_parts <= config.table_repair_min_success_parts:
        return expected_parts
    return max(config.table_repair_min_success_parts, math.ceil(expected_parts * 0.75))


def _should_skip_low_content_slice(image_slice: ImageSlice, config: BaselineConfig) -> bool:
    if (
        config.table_repair_min_text_pixels > 0
        and image_slice.text_pixels is not None
        and image_slice.text_pixels < config.table_repair_min_text_pixels
    ):
        return True

    min_pixels = config.table_repair_min_content_pixels
    min_ratio = config.table_repair_min_content_ratio
    if min_pixels <= 0 and min_ratio <= 0:
        return False
    if image_slice.content_pixels is None or image_slice.content_ratio is None:
        return False

    pixels_low = min_pixels > 0 and image_slice.content_pixels < min_pixels
    ratio_low = min_ratio > 0 and image_slice.content_ratio < min_ratio
    if min_pixels > 0 and min_ratio > 0:
        return pixels_low and ratio_low
    return pixels_low or ratio_low


def _call_long_record(
    client: FinixDocClient,
    record: ImageRecord,
    config: BaselineConfig,
) -> tuple[str, int]:
    slices = make_vertical_slices(
        file_name=record.file_name,
        image_bytes=record.read_bytes(),
        slice_height=config.long_slice_height,
        overlap=config.long_slice_overlap,
        jpeg_quality=config.jpeg_quality,
    )
    print(
        f"baseline API {record.file_name}: trying full-page slices "
        f"as {len(slices)} slices "
        f"(height={config.long_slice_height}, overlap={config.long_slice_overlap})"
    )

    tile_cache_dir = (
        config.cache_dir
        / _baseline_cache_namespace(config, "long_tiles")
        / Path(record.file_name).stem
    )
    tile_cache_dir.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    errors: list[str] = []
    calls = 0
    for slice_index, image_slice in enumerate(slices, start=1):
        tile_cache_path = _cache_path(tile_cache_dir, image_slice.file_name)
        called_api = False
        try:
            if tile_cache_path.exists():
                try:
                    markdown = normalize_markdown_payload(
                        tile_cache_path.read_text(encoding="utf-8")
                    )
                except FinixDocError as cache_error:
                    print(
                        f"  invalid long-slice cache {image_slice.file_name}: "
                        f"{cache_error}"
                    )
                    tile_cache_path.unlink()
                else:
                    issue = _candidate_issue(
                        markdown,
                        config,
                        min_chars=config.long_min_chars,
                    )
                    if issue is None:
                        print(
                            f"  long-slice cache hit {image_slice.file_name} "
                            f"chars={len(markdown)}"
                        )
                        parts.append(markdown)
                        continue
                    print(
                        f"  invalid long-slice cache {image_slice.file_name}: "
                        f"{issue}"
                    )
                    tile_cache_path.unlink()

            calls += 1
            called_api = True
            markdown = _call_with_retries(client, image_slice, config)
            issue = _candidate_issue(
                markdown,
                config,
                min_chars=config.long_min_chars,
            )
            if issue:
                raise FinixDocError(issue)
            tile_cache_path.write_text(markdown, encoding="utf-8")
            print(
                f"  selected {image_slice.file_name} "
                f"row={image_slice.row}/{image_slice.rows} "
                f"chars={len(markdown)}"
            )
            parts.append(markdown)
        except Exception as exc:
            errors.append(f"{image_slice.file_name}: {type(exc).__name__}: {exc}")
            print(f"  slice failed {image_slice.file_name}: {type(exc).__name__}: {exc}")
            parts.append("")
        finally:
            if (
                called_api
                and config.sleep_seconds > 0
                and slice_index < len(slices)
            ):
                time.sleep(config.sleep_seconds)

    if not 0 <= config.long_min_success_ratio <= 1:
        raise ValueError("long_min_success_ratio must be between 0 and 1")
    if config.long_max_failed_parts < 0:
        raise ValueError("long_max_failed_parts must be >= 0")
    success_parts = sum(bool(part.strip()) for part in parts)
    required_success_parts = math.ceil(len(slices) * config.long_min_success_ratio)
    if (
        len(errors) > config.long_max_failed_parts
        or success_parts < required_success_parts
    ):
        fallback = _try_long_slice_fallback(
            client=client,
            record=record,
            config=config,
            calls=calls,
        )
        if fallback is not None:
            return fallback
        raise _RouteFailure(
            "long-page slice candidates failed "
            f"(success={success_parts}/{len(slices)}, "
            f"failed={len(errors)}/{config.long_max_failed_parts}, "
            f"required={required_success_parts}); last errors: "
            + " | ".join(errors[-3:]),
            calls=calls,
        )

    if errors:
        merged = merge_sliced_markdown(slices, parts)
        fallback = _try_long_slice_fallback(
            client=client,
            record=record,
            config=config,
            calls=calls,
        )
        if fallback is not None:
            return fallback
        print(
            f"  accepted partial long document {record.file_name}: "
            f"success_parts={success_parts}/{len(slices)} failed={len(errors)}"
        )
        raise _PartialLongResult(
            merged,
            calls=calls,
            message="accepted partial long document",
        )

    merged = merge_sliced_markdown(slices, parts)
    fallback = _try_long_slice_fallback(
        client=client,
        record=record,
        config=config,
        calls=calls,
        markdown=merged,
    )
    if fallback is not None:
        merged, fallback_calls = fallback
        calls = fallback_calls
    local_repair = _maybe_local_long_text_repair(
        record=record,
        config=config,
        previous_markdown=merged,
    )
    if local_repair is not None:
        return local_repair, calls
    return merged, calls


def _try_long_slice_fallback(
    *,
    client: FinixDocClient,
    record: ImageRecord,
    config: BaselineConfig,
    calls: int,
    markdown: str | None = None,
) -> tuple[str, int] | None:
    """Retry a sparse long page once with a smaller, frozen slice geometry."""

    if (
        config.long_low_confidence_char_density <= 0
        or config.long_fallback_slice_height <= 0
        or config.long_fallback_slice_height >= config.long_slice_height
    ):
        return None
    if markdown is not None:
        profile = profile_image(
            image_bytes=record.read_bytes(),
            threshold=config.table_repair_content_threshold,
            sample_scale=config.table_repair_content_scale,
            padding=config.table_repair_content_padding,
        )
        density = len(markdown.strip()) / max(1, profile.content_height)
        if density >= config.long_low_confidence_char_density:
            return None
        print(
            f"  low-density long output {record.file_name}: "
            f"{density:.3f} < {config.long_low_confidence_char_density:.3f}; "
            f"retrying with height={config.long_fallback_slice_height}"
        )
    else:
        print(
            f"  long route failed {record.file_name}; "
            f"retrying with height={config.long_fallback_slice_height}"
        )

    fallback_config = replace(
        config,
        long_slice_height=config.long_fallback_slice_height,
        long_slice_overlap=(
            config.long_fallback_overlap
            if config.long_fallback_overlap > 0
            else config.long_slice_overlap
        ),
        long_low_confidence_char_density=0.0,
        long_fallback_slice_height=0,
        long_fallback_overlap=0,
    )
    try:
        fallback_markdown, fallback_calls = _call_long_record(
            client,
            record,
            fallback_config,
        )
    except (_RouteFailure, _PartialLongResult) as exc:
        print(
            f"  smaller long fallback rejected {record.file_name}: "
            f"{type(exc).__name__}: {exc}"
        )
        return None
    return fallback_markdown, calls + fallback_calls


def _maybe_local_long_text_repair(
    *,
    record: ImageRecord,
    config: BaselineConfig,
    previous_markdown: str,
) -> str | None:
    """Restore prose coverage only when the remote long route is implausibly sparse.

    The local renderer deliberately emits text lines rather than guessing
    Markdown hierarchy or tables.  It is therefore reserved for narrow,
    very-large documents where the remote output density proves that whole
    regions are missing and where the local pass adds substantial content.
    """

    if config.long_local_ocr_backend != "rapidocr":
        return None
    profile = _profile_record(record, config)
    pixels = profile.width * profile.height
    if pixels < config.long_local_ocr_min_pixels:
        return None
    if (
        config.long_local_ocr_max_width
        and profile.width > config.long_local_ocr_max_width
    ):
        return None
    remote_density = len(previous_markdown.strip()) / max(1, profile.content_height)
    if remote_density >= config.long_local_ocr_trigger_char_density:
        return None

    image_bytes = record.read_bytes()
    slices = make_vertical_slices(
        file_name=record.file_name,
        image_bytes=image_bytes,
        slice_height=config.long_local_ocr_slice_height,
        overlap=config.long_local_ocr_overlap,
        jpeg_quality=config.jpeg_quality,
    )
    cache_dir = (
        config.cache_dir
        / _baseline_cache_namespace(config, "local_rapidocr_text")
        / Path(record.file_name).stem
    )
    try:
        observations = run_rapidocr_observations(
            slices=slices,
            cache_dir=cache_dir,
            workers=config.long_local_ocr_workers,
        )
    except (OSError, LocalOCRError) as exc:
        print(
            f"  local rapidocr text route unavailable {record.file_name}: "
            f"{type(exc).__name__}: {exc}"
        )
        return None
    rendered = render_observations_in_reading_order(observations)
    local_density = len(rendered.strip()) / max(1, profile.content_height)
    if local_density < config.long_local_ocr_min_char_density:
        print(
            f"  local rapidocr text route rejected {record.file_name}: "
            f"density={local_density:.3f}"
        )
        return None
    if len(rendered.strip()) < len(previous_markdown.strip()) + config.long_local_ocr_min_gain:
        return None
    if _duplicate_line_ratio(rendered) > config.table_max_duplicate_line_ratio:
        return None
    print(
        f"  selected local rapidocr text repair {record.file_name}: "
        f"remote_density={remote_density:.3f} local_density={local_density:.3f} "
        f"chars={len(rendered)}"
    )
    return rendered


def _call_with_retries(
    client: FinixDocClient,
    crop: ImageSlice,
    config: BaselineConfig,
    *,
    allow_unclosed_fence: bool = False,
    balance_html_tables: bool = False,
    retry_truncation: bool = True,
) -> str:
    attempt = 0
    while True:
        try:
            if allow_unclosed_fence or balance_html_tables:
                return client.call_with_file(
                    crop.file_name,
                    crop.image_bytes,
                    allow_unclosed_fence=allow_unclosed_fence,
                    balance_html_tables=balance_html_tables,
                )
            return client.call_with_file(crop.file_name, crop.image_bytes)
        except Exception as exc:
            attempt += 1
            if not retry_truncation and _is_truncation_error(exc):
                raise
            if attempt > config.retries:
                raise
            delay_seconds = config.retry_sleep_seconds
            if "HTTP 429" in str(exc):
                delay_seconds *= 2 ** (attempt - 1)
            print(
                f"    retry {attempt}/{config.retries} after "
                f"{delay_seconds:.0f}s: {crop.file_name} "
                f"({type(exc).__name__}: {exc})"
            )
            time.sleep(delay_seconds)


def _placeholder_markdown(record: ImageRecord) -> str:
    return "<table></table>\n"


def _serialize_output_tables(markdown: str, config: BaselineConfig) -> str:
    """Apply the configured table representation at the output boundary.

    OCR caches intentionally remain in their original form so parser and
    reconstruction improvements can be replayed without API calls.  The CSV
    writer, training evaluator, and B-list submit artifact nevertheless see
    the exact same canonical representation.
    """

    if config.table_output_format == "html":
        return markdown
    return html_tables_to_markdown(markdown)


def _local_matrix_candidate_markdown(markdown: str, config: BaselineConfig) -> str:
    """Canonicalize, then budget only a structurally validated local table."""

    candidate = _serialize_output_tables(markdown, config)
    if config.table_local_ocr_max_output_bytes:
        candidate = retain_complete_pipe_table_rows(
            candidate,
            max_bytes=config.table_local_ocr_max_output_bytes,
        )
    return candidate


def _build_output_rows(
    *,
    rows_by_name: dict[str, str],
    selected_records: list[ImageRecord],
    submission_file_names: tuple[str, ...] | None,
    missing_markdown: str,
) -> tuple[list[dict[str, str]], int]:
    if submission_file_names is None:
        return (
            [
                {
                    "file_name": record.file_name,
                    "ground_truth": rows_by_name[record.file_name],
                }
                for record in selected_records
            ],
            0,
        )

    missing = 0
    rows: list[dict[str, str]] = []
    for file_name in submission_file_names:
        markdown = rows_by_name.get(file_name)
        if markdown is None:
            markdown = missing_markdown
            missing += 1
        rows.append({"file_name": file_name, "ground_truth": markdown})
    return rows, missing


def _candidate_issue(
    markdown: str,
    config: BaselineConfig,
    *,
    min_chars: int | None = None,
) -> str | None:
    min_chars = config.min_chars if min_chars is None else min_chars
    stripped = markdown.strip()
    if min_chars > 0 and len(stripped) < min_chars:
        return f"candidate output is too short ({len(stripped)} chars)"
    if "markdown parsing task" in stripped.lower():
        return "candidate output appears to contain prompt leakage"
    structure_issue = html_table_structure_issue(markdown)
    if structure_issue is not None:
        return f"candidate {structure_issue}"
    if _looks_truncated(markdown):
        return 'candidate output appears truncated'
    return None


def _table_candidate_issue(
    markdown: str,
    config: BaselineConfig,
    *,
    min_chars: int | None = None,
) -> str | None:
    issue = _candidate_issue(markdown, config, min_chars=min_chars)
    if issue is not None:
        return issue

    table_blocks = markdown.lower().count('<table') + _pipe_table_count(markdown)
    if table_blocks > config.table_max_blocks:
        return (
            'candidate contains too many separate table blocks '
            f'({table_blocks} > {config.table_max_blocks})'
        )

    widest_row = _max_html_cells_per_row(markdown)
    if widest_row > _MAX_HTML_CELLS_PER_ROW:
        return (
            'candidate contains a pathologically wide HTML row '
            f'({widest_row} > {_MAX_HTML_CELLS_PER_ROW} cells)'
        )

    if (
        _table_structure_kind(markdown) == 0
        and _looks_like_unstructured_table_text(markdown)
    ):
        return "candidate output looks like incomplete unstructured table text"
    return None


def _max_html_cells_per_row(markdown: str) -> int:
    widths = [
        len(re.findall(r'<\s*t[dh](?:\s[^<>]*)?>', row, flags=re.IGNORECASE))
        for row in re.findall(
            r'<tr\b[^>]*>.*?</tr\s*>',
            markdown,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]
    return max(widths, default=0)


def _coverage_candidate_issue(markdown: str, config: BaselineConfig) -> str | None:
    """Apply conservative structural checks before accepting a tiled result.

    Repeated rows across neighboring coverage tiles are a reliable indication
    that horizontal/vertical reconstruction lost its alignment.  This check is
    intentionally coverage-only: anchor repair can legitimately repeat small
    headers while it is trying alternate crops.
    """
    issue = _table_candidate_issue(markdown, config)
    if issue is not None:
        return issue

    duplicate_ratio = _duplicate_line_ratio(markdown)
    max_ratio = config.table_max_duplicate_line_ratio
    if max_ratio >= 0 and duplicate_ratio > max_ratio:
        return (
            'coverage candidate has too many duplicate lines '
            f'({duplicate_ratio:.3f} > {max_ratio:.3f})'
        )
    return None


def _should_attempt_table_repair(
    record: ImageRecord,
    markdown: str,
    config: BaselineConfig,
) -> bool:
    stripped = markdown.strip()
    if len(stripped) < _table_repair_trigger_chars(record, config):
        return True

    if _table_structure_kind(markdown) == 0 and _looks_like_unstructured_table_text(markdown):
        return True

    return False


def _table_repair_trigger_chars(record: ImageRecord, config: BaselineConfig) -> int:
    """Set a generic repair threshold from the visible content amount.

    A sparse one-page table should not be retried solely because it has fewer
    than a large fixed character count. Conversely, a dense full-page table
    with only a few thousand characters is usually a partial OCR result.
    """
    threshold = config.table_repair_min_chars
    if config.table_repair_min_chars_per_content_pixel <= 0:
        return threshold
    document = profile_image(
        image_bytes=record.read_bytes(),
        threshold=config.table_repair_content_threshold,
        sample_scale=config.table_repair_content_scale,
    )
    density_threshold = math.ceil(
        document.content_pixels * config.table_repair_min_chars_per_content_pixel
    )
    return max(threshold, density_threshold)


def _table_repair_grid(image_bytes: bytes, config: BaselineConfig) -> tuple[int, int]:
    """Choose a content-box grid, optionally bounded by target tile pixels."""
    rows = config.table_repair_rows
    cols = config.table_repair_cols
    if config.table_repair_target_tile_width <= 0 and config.table_repair_target_tile_height <= 0:
        return rows, cols
    profile = profile_image(
        image_bytes=image_bytes,
        threshold=config.table_repair_content_threshold,
        sample_scale=config.table_repair_content_scale,
        padding=config.table_repair_content_padding,
    )
    if config.table_repair_target_tile_height > 0:
        rows = max(1, math.ceil(profile.content_height / config.table_repair_target_tile_height))
    if config.table_repair_target_tile_width > 0:
        cols = max(1, math.ceil(profile.content_width / config.table_repair_target_tile_width))
    if (
        config.table_repair_vertical_aspect_threshold > 0
        and profile.content_width > 0
        and profile.content_height
        >= profile.content_width * config.table_repair_vertical_aspect_threshold
    ):
        cols = 1
    if config.table_repair_max_calls > 0:
        rows, cols = _fit_grid_to_call_budget(rows, cols, config.table_repair_max_calls)
    return rows, cols


def _fit_grid_to_call_budget(rows: int, cols: int, max_calls: int) -> tuple[int, int]:
    """Choose the least-distorted sub-grid that fits a per-record call budget."""

    if rows <= 0 or cols <= 0:
        raise ValueError("repair grid dimensions must be positive")
    if max_calls <= 0 or rows * cols <= max_calls:
        return rows, cols

    candidates = [
        (candidate_rows, candidate_cols)
        for candidate_rows in range(1, rows + 1)
        for candidate_cols in range(1, cols + 1)
        if candidate_rows * candidate_cols <= max_calls
    ]
    return min(
        candidates,
        key=lambda grid: (
            max(rows / grid[0], cols / grid[1]),
            -grid[0] * grid[1],
            abs((rows / cols) - (grid[0] / grid[1])),
        ),
    )


def _table_repair_failed_part_budget(
    config: BaselineConfig,
    eligible_parts: list[bool],
) -> int | None:
    """Scale the failure circuit breaker when repair grids become larger.

    A fixed limit is appropriate for a 5x5 grid, but would reject the same
    failure rate prematurely on a larger adaptive grid.  When both limits are
    set, the wider one applies: the absolute value preserves the established
    small-grid safeguard while the ratio keeps its meaning across grid sizes.
    """
    if not 0 <= config.table_repair_max_failed_ratio <= 1:
        raise ValueError("table_repair_max_failed_ratio must be between 0 and 1")
    limits = []
    if config.table_repair_max_failed_parts > 0:
        limits.append(config.table_repair_max_failed_parts)
    if config.table_repair_max_failed_ratio > 0:
        limits.append(
            math.ceil(sum(eligible_parts) * config.table_repair_max_failed_ratio)
        )
    return max(limits) if limits else None


def _raise_if_repair_part_repeats(
    markdown: str,
    config: BaselineConfig,
    counts: dict[str, int],
    *,
    calls: int,
) -> None:
    """Abort when a large byte-equivalent OCR result repeats across tiles."""

    max_parts = config.table_repair_max_identical_parts
    if max_parts <= 0 or len(markdown.strip()) < config.table_repair_identical_min_chars:
        return
    normalized = " ".join(markdown.split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    counts[digest] = counts.get(digest, 0) + 1
    if counts[digest] >= max_parts:
        raise _RepeatedRepairOutput(
            "content grid repair repeated the same large OCR result across "
            f"{counts[digest]} tiles",
            calls=calls,
        )


def _repair_is_better(
    previous: str,
    repaired: str,
    min_gain: int,
    *,
    max_duplicate_line_ratio: float = -1,
) -> bool:
    repaired_stripped = repaired.strip()
    if not repaired_stripped:
        return False
    if (
        max_duplicate_line_ratio >= 0
        and _duplicate_line_ratio(repaired) > max_duplicate_line_ratio
    ):
        return False

    previous_stripped = previous.strip()
    previous_kind = _table_structure_kind(previous)
    repaired_kind = _table_structure_kind(repaired)
    if repaired_kind > previous_kind:
        return True
    if not previous_stripped:
        return True

    previous_chars = len(previous_stripped)
    repaired_chars = len(repaired_stripped)
    previous_score = _score_table_candidate(previous).score
    repaired_score = _score_table_candidate(repaired).score
    if repaired_score >= previous_score + 8:
        return True
    return repaired_score >= previous_score and repaired_chars >= previous_chars + min_gain


def _table_structure_kind(markdown: str) -> int:
    if parse_table(markdown) is not None:
        return 2

    lowered = markdown.lower()
    if "<table" in lowered or "</table>" in lowered:
        return 1
    if any(line.strip().count("|") >= 2 for line in markdown.splitlines()):
        return 1
    return 0


def _looks_like_unstructured_table_text(markdown: str) -> bool:
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    if len(lines) < 4:
        return False

    numeric_tokens = re.findall(r"\d+(?:[.,]\d+)?", markdown)
    if len(numeric_tokens) < 8:
        return False

    dense_numeric_lines = sum(
        1 for line in lines if len(re.findall(r"\d+(?:[.,]\d+)?", line)) >= 2
    )
    return dense_numeric_lines >= 3 or (len(lines) >= 6 and len(numeric_tokens) >= 12)


def _score_table_candidate(
    markdown: str,
    *,
    parts: list[str] | None = None,
    slices: list[ImageSlice] | None = None,
) -> TableCandidateScore:
    lowered = markdown.lower()
    table_count = lowered.count('<table') + _pipe_table_count(markdown)
    row_count = lowered.count('<tr') + _pipe_data_row_count(markdown)
    cell_count = (
        len(re.findall(r'<\s*t[dh](?:\s[^<>]*)?>', markdown, flags=re.IGNORECASE))
        + _pipe_cell_count(markdown)
    )
    balanced_html = lowered.count('<table') == lowered.count('</table')
    duplicate_ratio = _duplicate_line_ratio(markdown)
    truncated = _looks_truncated(markdown)
    score = 0.0
    reasons: list[str] = []
    if table_count > 0:
        score += 16
    else:
        reasons.append('no table')
    if balanced_html:
        score += 14
    else:
        reasons.append('unbalanced html')
    if not truncated:
        score += 14
    else:
        reasons.append('truncated')
    score += min(24.0, cell_count / 18)
    score += min(14.0, row_count / 6)
    if parts:
        nonempty_parts = sum(bool(part.strip()) for part in parts)
        score += min(10.0, nonempty_parts * 2)
    if slices:
        expected_parts = sum(not _slice_is_effectively_blank(image_slice) for image_slice in slices)
        actual_parts = sum(bool(part.strip()) for part in parts or [])
        if expected_parts and actual_parts >= max(1, round(expected_parts * 0.75)):
            score += 8
        else:
            reasons.append('low coverage')
    if duplicate_ratio <= 0.18:
        score += 8
    else:
        reasons.append('duplicate lines')
    return TableCandidateScore(
        score=min(100.0, score),
        reason=', '.join(reasons) if reasons else 'ok',
        table_count=table_count,
        row_count=row_count,
        cell_count=cell_count,
        duplicate_line_ratio=duplicate_ratio,
        truncated=truncated,
        balanced_html=balanced_html,
    )


def _looks_truncated(markdown: str) -> bool:
    stripped = markdown.strip()
    if not stripped:
        return True
    if stripped.startswith('```') and not stripped.endswith('```'):
        return True
    lowered = stripped.lower()
    if lowered.count('<table') != lowered.count('</table'):
        return True
    if re.search(r'<(?:td|th|tr|table)[^>]*>$', lowered):
        return True
    if re.search(r'<(?:td|th)[^>]*>[^<]{0,3}$', lowered):
        return True
    return False


def _duplicate_line_ratio(markdown: str) -> float:
    lines = [' '.join(line.split()) for line in markdown.splitlines() if line.strip()]
    if not lines:
        return 1.0
    return 1.0 - (len(set(lines)) / len(lines))


def _pipe_table_count(markdown: str) -> int:
    lines = markdown.splitlines()
    return sum(
        1
        for index, line in enumerate(lines[:-1])
        if line.strip().startswith('|') and _is_pipe_separator(lines[index + 1].strip())
    )


def _pipe_data_row_count(markdown: str) -> int:
    return sum(
        1
        for line in markdown.splitlines()
        if line.strip().startswith('|') and not _is_pipe_separator(line.strip())
    )


def _pipe_cell_count(markdown: str) -> int:
    total = 0
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith('|') and not _is_pipe_separator(stripped):
            total += max(0, len(stripped.strip('|').split('|')))
    return total


def _is_pipe_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
    return bool(cells) and all(re.fullmatch(r':?-{3,}:?', cell.replace(' ', '')) for cell in cells)


def _slice_is_effectively_blank(image_slice: ImageSlice) -> bool:
    if image_slice.content_pixels is None or image_slice.content_ratio is None:
        return False
    return image_slice.content_pixels == 0 and image_slice.content_ratio == 0


def _write_errors_if_needed(errors_csv: Path | None, rows: list[dict[str, str]]) -> None:
    if errors_csv and rows:
        write_errors(errors_csv, rows)


def _baseline_cache_namespace(config: BaselineConfig, strategy: str) -> str:
    if strategy in {"long", "long_tiles"}:
        payload = {
            "schema": (
                BASELINE_CACHE_SCHEMA_VERSION if strategy == "long" else 1
            ),
            "strategy": strategy,
            "jpeg_quality": config.jpeg_quality,
            "long_slice_height": config.long_slice_height,
            "long_slice_overlap": config.long_slice_overlap,
            "long_low_confidence_char_density": config.long_low_confidence_char_density,
            "long_fallback_slice_height": config.long_fallback_slice_height,
            "long_fallback_overlap": config.long_fallback_overlap,
            "long_local_ocr_backend": config.long_local_ocr_backend,
            "long_local_ocr_min_pixels": config.long_local_ocr_min_pixels,
            "long_local_ocr_max_width": config.long_local_ocr_max_width,
            "long_local_ocr_trigger_char_density": config.long_local_ocr_trigger_char_density,
            "long_local_ocr_slice_height": config.long_local_ocr_slice_height,
            "long_local_ocr_overlap": config.long_local_ocr_overlap,
            "long_local_ocr_min_char_density": config.long_local_ocr_min_char_density,
            "long_local_ocr_min_gain": config.long_local_ocr_min_gain,
            "long_min_chars": config.long_min_chars,
            "long_min_success_ratio": config.long_min_success_ratio,
            "long_max_failed_parts": config.long_max_failed_parts,
        }
    elif strategy == "vision_matrix":
        payload = {
            "schema": 1,
            "strategy": strategy,
            "jpeg_quality": config.jpeg_quality,
            "table_repair_target_tile_width": config.table_repair_target_tile_width,
            "table_repair_target_tile_height": config.table_repair_target_tile_height,
            "table_repair_vertical_aspect_threshold": config.table_repair_vertical_aspect_threshold,
            "table_repair_max_calls": config.table_repair_max_calls,
            "table_repair_content_threshold": config.table_repair_content_threshold,
            "table_repair_content_scale": config.table_repair_content_scale,
            "table_repair_content_padding": config.table_repair_content_padding,
        }
    elif strategy == "repair_tiles":
        payload = {
            "schema": 1,
            "strategy": strategy,
            "jpeg_quality": config.jpeg_quality,
            "table_repair_rows": config.table_repair_rows,
            "table_repair_cols": config.table_repair_cols,
            "table_repair_target_tile_width": config.table_repair_target_tile_width,
            "table_repair_target_tile_height": config.table_repair_target_tile_height,
            "table_repair_vertical_aspect_threshold": config.table_repair_vertical_aspect_threshold,
            "table_repair_overlap": config.table_repair_overlap,
            "table_repair_content_threshold": config.table_repair_content_threshold,
            "table_repair_content_scale": config.table_repair_content_scale,
            "table_repair_content_padding": config.table_repair_content_padding,
            "table_repair_min_content_pixels": config.table_repair_min_content_pixels,
            "table_repair_min_content_ratio": config.table_repair_min_content_ratio,
            "table_repair_min_text_pixels": config.table_repair_min_text_pixels,
            "table_repair_header_context_height": config.table_repair_header_context_height,
            "table_repair_left_context_width": config.table_repair_left_context_width,
            "table_repair_snap_boundaries": config.table_repair_snap_boundaries,
            "table_repair_snap_x_boundaries": config.table_repair_snap_x_boundaries,
            "table_repair_snap_y_boundaries": config.table_repair_snap_y_boundaries,
            # A cached successful reading of this exact top-level region stays
            # valid when only the bounded search depth changes.  Keeping the
            # historical depth-zero key lets a refinement experiment reuse all
            # successful parent tiles and spend calls only on missing regions.
            "table_refine_max_depth": 0,
            "table_refine_rows": config.table_refine_rows,
            "table_refine_cols": config.table_refine_cols,
        }
    elif strategy == "coverage_tiles":
        payload = {
            "schema": 1,
            "strategy": strategy,
            "jpeg_quality": config.jpeg_quality,
            "table_target_tile_width": config.table_target_tile_width,
            "table_target_tile_height": config.table_target_tile_height,
            "table_max_rows": config.table_max_rows,
            "table_max_cols": config.table_max_cols,
            "table_overlap_ratio": config.table_overlap_ratio,
            "table_min_overlap": config.table_min_overlap,
            "table_repair_content_threshold": config.table_repair_content_threshold,
            "table_repair_content_scale": config.table_repair_content_scale,
            "table_repair_content_padding": config.table_repair_content_padding,
            "table_repair_header_context_height": config.table_repair_header_context_height,
            "table_repair_left_context_width": config.table_repair_left_context_width,
            "table_repair_snap_boundaries": config.table_repair_snap_boundaries,
            "table_repair_snap_x_boundaries": config.table_repair_snap_x_boundaries,
            "table_repair_snap_y_boundaries": config.table_repair_snap_y_boundaries,
            "table_refine_max_depth": 0,
            "table_refine_rows": config.table_refine_rows,
            "table_refine_cols": config.table_refine_cols,
        }
    elif strategy.startswith("local_") and (
        strategy.endswith("_matrix") or strategy.endswith("_text")
    ):
        # Local OCR observations are raw model outputs, just like remote tile
        # caches.  Keep them reusable when only reconstruction/refinement
        # selection changes; child tiles have deterministic unique names.
        payload: dict[str, object] = {
            "schema": 1,
            "strategy": strategy,
            "jpeg_quality": config.jpeg_quality,
        }
        if strategy.endswith("_matrix"):
            payload.update(
                {
                    "table_repair_rows": config.table_repair_rows,
                    "table_repair_cols": config.table_repair_cols,
                    "table_repair_target_tile_width": config.table_repair_target_tile_width,
                    "table_repair_target_tile_height": config.table_repair_target_tile_height,
                    "table_repair_vertical_aspect_threshold": config.table_repair_vertical_aspect_threshold,
                    "table_repair_max_calls": config.table_repair_max_calls,
                    "table_repair_content_threshold": config.table_repair_content_threshold,
                    "table_repair_content_scale": config.table_repair_content_scale,
                    "table_repair_content_padding": config.table_repair_content_padding,
                }
            )
        else:
            payload.update(
                {
                    "long_local_ocr_slice_height": config.long_local_ocr_slice_height,
                    "long_local_ocr_overlap": config.long_local_ocr_overlap,
                }
            )
    else:
        payload = {
            "schema": BASELINE_CACHE_SCHEMA_VERSION,
            "strategy": strategy,
            "crop_sizes": config.crop_sizes,
            "anchors": config.anchors,
            "jpeg_quality": config.jpeg_quality,
            "min_chars": config.min_chars,
            "table_repair_min_chars": config.table_repair_min_chars,
            "table_repair_min_chars_per_content_pixel": config.table_repair_min_chars_per_content_pixel,
            "table_repair_min_gain": config.table_repair_min_gain,
            "table_repair_rows": config.table_repair_rows,
            "table_repair_cols": config.table_repair_cols,
            "table_repair_target_tile_width": config.table_repair_target_tile_width,
            "table_repair_target_tile_height": config.table_repair_target_tile_height,
            "table_repair_vertical_aspect_threshold": config.table_repair_vertical_aspect_threshold,
            "table_repair_overlap": config.table_repair_overlap,
            "table_repair_content_threshold": config.table_repair_content_threshold,
            "table_repair_content_scale": config.table_repair_content_scale,
            "table_repair_content_padding": config.table_repair_content_padding,
            "table_repair_min_content_pixels": config.table_repair_min_content_pixels,
            "table_repair_min_content_ratio": config.table_repair_min_content_ratio,
            "table_repair_min_text_pixels": config.table_repair_min_text_pixels,
            "table_repair_header_context_height": config.table_repair_header_context_height,
            "table_repair_left_context_width": config.table_repair_left_context_width,
            "table_repair_snap_boundaries": config.table_repair_snap_boundaries,
            "table_repair_snap_x_boundaries": config.table_repair_snap_x_boundaries,
            "table_repair_snap_y_boundaries": config.table_repair_snap_y_boundaries,
            "table_repair_min_success_parts": config.table_repair_min_success_parts,
            "table_repair_min_success_ratio": config.table_repair_min_success_ratio,
            "table_repair_max_calls": config.table_repair_max_calls,
            "table_repair_max_failed_parts": config.table_repair_max_failed_parts,
            "table_repair_max_failed_ratio": config.table_repair_max_failed_ratio,
            "table_repair_max_identical_parts": config.table_repair_max_identical_parts,
            "table_repair_identical_min_chars": config.table_repair_identical_min_chars,
            "table_local_ocr_backend": config.table_local_ocr_backend,
            "table_local_ocr_min_pixels": config.table_local_ocr_min_pixels,
            "table_local_ocr_max_pixels": config.table_local_ocr_max_pixels,
            "table_local_ocr_trigger_max_chars": config.table_local_ocr_trigger_max_chars,
            "table_local_ocr_refine_saturated": config.table_local_ocr_refine_saturated,
            "table_local_ocr_max_refine_depth": config.table_local_ocr_max_refine_depth,
            "table_anchor_max_candidates": config.table_anchor_max_candidates,
            "table_anchor_max_attempts": config.table_anchor_max_attempts,
            "table_mode": config.table_mode,
            "table_target_tile_width": config.table_target_tile_width,
            "table_target_tile_height": config.table_target_tile_height,
            "table_max_rows": config.table_max_rows,
            "table_max_cols": config.table_max_cols,
            "table_overlap_ratio": config.table_overlap_ratio,
            "table_min_overlap": config.table_min_overlap,
            "table_max_blocks": config.table_max_blocks,
            "table_min_score": config.table_min_score,
            "table_max_duplicate_line_ratio": config.table_max_duplicate_line_ratio,
            "table_hybrid_min_content_ratio": config.table_hybrid_min_content_ratio,
            "table_refine_max_depth": config.table_refine_max_depth,
            "table_refine_rows": config.table_refine_rows,
            "table_refine_cols": config.table_refine_cols,
            "table_fragment_max_blocks": config.table_fragment_max_blocks,
            "table_fragment_refine_cols": config.table_fragment_refine_cols,
            "long_aspect_threshold": config.long_aspect_threshold,
            "long_low_confidence_char_density": config.long_low_confidence_char_density,
            "long_fallback_slice_height": config.long_fallback_slice_height,
            "long_fallback_overlap": config.long_fallback_overlap,
            "long_local_ocr_backend": config.long_local_ocr_backend,
            "long_local_ocr_min_pixels": config.long_local_ocr_min_pixels,
            "long_local_ocr_max_width": config.long_local_ocr_max_width,
            "long_local_ocr_trigger_char_density": config.long_local_ocr_trigger_char_density,
            "long_local_ocr_slice_height": config.long_local_ocr_slice_height,
            "long_local_ocr_overlap": config.long_local_ocr_overlap,
            "long_local_ocr_min_char_density": config.long_local_ocr_min_char_density,
            "long_local_ocr_min_gain": config.long_local_ocr_min_gain,
            "long_min_success_ratio": config.long_min_success_ratio,
            "long_max_failed_parts": config.long_max_failed_parts,
        }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    if strategy == "long":
        return (
            f"baseline_long_v{BASELINE_CACHE_SCHEMA_VERSION}_"
            f"h{config.long_slice_height}_"
            f"o{config.long_slice_overlap}_"
            f"q{config.jpeg_quality}_"
            f"{digest}"
        )
    if strategy == "long_tiles":
        return (
            "baseline_long_tiles_v1_"
            f"h{config.long_slice_height}_"
            f"o{config.long_slice_overlap}_"
            f"q{config.jpeg_quality}_"
            f"{digest}"
        )
    if strategy == "repair_tiles":
        return f"baseline_repair_tiles_v1_q{config.jpeg_quality}_{digest}"
    if strategy == "coverage_tiles":
        return f"baseline_coverage_tiles_v1_q{config.jpeg_quality}_{digest}"
    if strategy == "vision_matrix":
        return f"baseline_vision_matrix_v1_q{config.jpeg_quality}_{digest}"
    return (
        f"baseline_v{BASELINE_CACHE_SCHEMA_VERSION}_"
        f"s{'-'.join(map(str, config.crop_sizes))}_"
        f"q{config.jpeg_quality}_"
        f"{digest}"
    )


def _cache_path(cache_dir: Path, file_name: str) -> Path:
    return cache_dir / f"{Path(file_name).stem}.md"


def _record_strategy(record: ImageRecord, config: BaselineConfig) -> str:
    profile = _profile_record(record, config)
    if profile.aspect_ratio <= config.long_aspect_threshold:
        return 'long'
    return 'table'


def _profile_record(record: ImageRecord, config: BaselineConfig) -> DocumentProfile:
    return profile_image(
        image_bytes=record.read_bytes(),
        threshold=config.table_repair_content_threshold,
        sample_scale=config.table_repair_content_scale,
        padding=0,
    )


def _candidate_min_chars(strategy: str, config: BaselineConfig) -> int:
    return config.long_min_chars if strategy == "long" else config.min_chars
