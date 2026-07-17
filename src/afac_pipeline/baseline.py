"""Low-call-count baseline submission generation."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .api import FinixDocClient, FinixDocError, normalize_markdown_payload
from .datasets import ImageRecord
from .images import (
    DEFAULT_CROP_ANCHORS,
    ImageSlice,
    make_anchor_crops,
    make_content_grid_slices,
    make_vertical_slices,
)
from .pipeline import _merge_sliced_markdown, write_errors, write_submission


BASELINE_CACHE_SCHEMA_VERSION = 1


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
    retries: int = 0
    retry_sleep_seconds: float = 60.0
    min_chars: int = 20
    table_repair_min_chars: int = 0
    table_repair_min_gain: int = 300
    table_repair_rows: int = 4
    table_repair_cols: int = 4
    table_repair_overlap: int = 120
    table_repair_content_threshold: int = 245
    table_repair_content_scale: float = 0.04
    table_repair_content_padding: int = 200
    table_repair_min_success_parts: int = 4
    long_slice_height: int = 12000
    long_slice_overlap: int = 400
    long_min_chars: int = 20
    on_error: str = "raise"
    errors_csv: Path | None = None
    submission_file_names: tuple[str, ...] | None = None
    missing_markdown: str = "<table></table>\n"


@dataclass(frozen=True)
class BaselineStats:
    total_discovered: int
    processed: int
    cache_hits: int
    api_calls: int
    fallbacks: int
    template_missing: int
    output_csv: Path


def run_baseline_submission(
    *,
    records: Iterable[ImageRecord],
    client: FinixDocClient,
    config: BaselineConfig,
) -> BaselineStats:
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
        strategy = _record_strategy(record)
        strategy_cache_dir = config.cache_dir / _baseline_cache_namespace(config, strategy)
        strategy_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = _cache_path(strategy_cache_dir, record.file_name)
        markdown: str

        try:
            cache_hit = False
            if config.resume and cache_path.exists():
                cached_text = cache_path.read_text(encoding="utf-8")
                markdown = normalize_markdown_payload(cached_text)
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
                markdown, calls = _call_record_baseline(client, record, config, strategy=strategy)
                api_calls += calls
                cache_path.write_text(markdown, encoding="utf-8")
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

        rows_by_name[record.file_name] = markdown

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


def _call_table_record(
    client: FinixDocClient,
    record: ImageRecord,
    config: BaselineConfig,
) -> tuple[str, int]:
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
    for crop in crops:
        try:
            calls += 1
            markdown = _call_with_retries(client, crop, config)
            issue = _candidate_issue(markdown, config)
            if issue:
                raise FinixDocError(issue)
            print(
                f"  selected {crop.file_name} "
                f"x={crop.x0}:{crop.x1} y={crop.y0}:{crop.y1} chars={len(markdown)}"
            )
            repaired = _maybe_repair_short_table(
                client=client,
                record=record,
                config=config,
                markdown=markdown,
            )
            if repaired is not None:
                repaired_markdown, repair_calls = repaired
                return repaired_markdown, calls + repair_calls
            return markdown, calls
        except Exception as exc:
            errors.append(f"{crop.file_name}: {type(exc).__name__}: {exc}")
            print(f"  crop failed {crop.file_name}: {type(exc).__name__}: {exc}")
            if config.sleep_seconds > 0:
                time.sleep(config.sleep_seconds)

    if config.table_repair_min_chars > 0:
        try:
            repaired_markdown, repair_calls = _call_table_content_grid_record(
                client,
                record,
                config,
                previous_markdown="",
            )
            return repaired_markdown, calls + repair_calls
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
    if config.table_repair_min_chars <= 0:
        return None
    if len(markdown.strip()) >= config.table_repair_min_chars:
        return None

    try:
        repaired_markdown, calls = _call_table_content_grid_record(
            client,
            record,
            config,
            previous_markdown=markdown,
        )
    except Exception as exc:
        print(
            f"  content grid repair failed {record.file_name}: "
            f"{type(exc).__name__}: {exc}"
        )
        return None
    if len(repaired_markdown) < len(markdown) + config.table_repair_min_gain:
        print(
            f"  kept crop result after content grid repair "
            f"old_chars={len(markdown)} new_chars={len(repaired_markdown)}"
        )
        return markdown, calls
    return repaired_markdown, calls


def _call_table_content_grid_record(
    client: FinixDocClient,
    record: ImageRecord,
    config: BaselineConfig,
    *,
    previous_markdown: str,
) -> tuple[str, int]:
    slices = make_content_grid_slices(
        file_name=record.file_name,
        image_bytes=record.read_bytes(),
        rows=config.table_repair_rows,
        cols=config.table_repair_cols,
        threshold=config.table_repair_content_threshold,
        sample_scale=config.table_repair_content_scale,
        padding=config.table_repair_content_padding,
        x_overlap=config.table_repair_overlap,
        y_overlap=config.table_repair_overlap,
        jpeg_quality=config.jpeg_quality,
    )
    print(
        f"  trying content grid repair {record.file_name}: "
        f"{config.table_repair_rows}x{config.table_repair_cols} "
        f"threshold={config.table_repair_content_threshold}"
    )

    parts: list[str] = []
    errors: list[str] = []
    calls = 0
    for image_slice in slices:
        try:
            calls += 1
            markdown = _call_with_retries(
                client,
                image_slice,
                config,
                allow_unclosed_fence=True,
                balance_html_tables=True,
            )
            issue = _candidate_issue(markdown, config, min_chars=10)
            if issue:
                raise FinixDocError(issue)
            print(
                f"    repaired {image_slice.file_name} "
                f"row={image_slice.row}/{image_slice.rows} "
                f"col={image_slice.col}/{image_slice.cols} "
                f"chars={len(markdown)}"
            )
            parts.append(markdown)
        except Exception as exc:
            errors.append(f"{image_slice.file_name}: {type(exc).__name__}: {exc}")
            print(f"    grid slice failed {image_slice.file_name}: {type(exc).__name__}: {exc}")
            parts.append("")
            if config.sleep_seconds > 0:
                time.sleep(config.sleep_seconds)

    success_parts = sum(bool(part.strip()) for part in parts)
    if success_parts < config.table_repair_min_success_parts:
        raise FinixDocError(
            "content grid repair produced too few usable parts "
            f"({success_parts}/{len(slices)}); last errors: " + " | ".join(errors[-3:])
        )

    repaired = _merge_sliced_markdown(slices, parts)
    issue = _candidate_issue(repaired, config, min_chars=config.min_chars)
    if issue:
        raise FinixDocError(f"content grid repair result is invalid: {issue}")
    if len(repaired.strip()) <= len(previous_markdown.strip()):
        raise FinixDocError(
            "content grid repair did not improve output length "
            f"({len(repaired.strip())} <= {len(previous_markdown.strip())})"
        )
    print(
        f"  selected content grid repair {record.file_name} "
        f"success_parts={success_parts}/{len(slices)} chars={len(repaired)}"
    )
    return repaired, calls


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

    parts: list[str] = []
    errors: list[str] = []
    calls = 0
    for slice_index, image_slice in enumerate(slices, start=1):
        try:
            calls += 1
            markdown = _call_with_retries(client, image_slice, config)
            issue = _candidate_issue(
                markdown,
                config,
                min_chars=config.long_min_chars,
            )
            if issue:
                raise FinixDocError(issue)
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
            if config.sleep_seconds > 0 and slice_index < len(slices):
                time.sleep(config.sleep_seconds)

    if not any(part.strip() for part in parts):
        raise FinixDocError(
            "all long-page slice candidates failed; last errors: " + " | ".join(errors[-3:])
        )

    return _merge_sliced_markdown(slices, parts), calls


def _call_with_retries(
    client: FinixDocClient,
    crop: ImageSlice,
    config: BaselineConfig,
    *,
    allow_unclosed_fence: bool = False,
    balance_html_tables: bool = False,
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
        except Exception:
            attempt += 1
            if attempt > config.retries:
                raise
            print(
                f"    retry {attempt}/{config.retries} after "
                f"{config.retry_sleep_seconds:.0f}s: {crop.file_name}"
            )
            time.sleep(config.retry_sleep_seconds)


def _placeholder_markdown(record: ImageRecord) -> str:
    return "<table></table>\n"


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
    lowered = markdown.lower()
    if lowered.count("<table") != lowered.count("</table>"):
        return "candidate HTML table tag count is not balanced"
    return None


def _write_errors_if_needed(errors_csv: Path | None, rows: list[dict[str, str]]) -> None:
    if errors_csv and rows:
        write_errors(errors_csv, rows)


def _baseline_cache_namespace(config: BaselineConfig, strategy: str) -> str:
    if strategy == "long":
        payload = {
            "schema": BASELINE_CACHE_SCHEMA_VERSION,
            "strategy": strategy,
            "jpeg_quality": config.jpeg_quality,
            "long_slice_height": config.long_slice_height,
            "long_slice_overlap": config.long_slice_overlap,
            "long_min_chars": config.long_min_chars,
        }
    else:
        payload = {
            "schema": BASELINE_CACHE_SCHEMA_VERSION,
            "crop_sizes": config.crop_sizes,
            "anchors": config.anchors,
            "jpeg_quality": config.jpeg_quality,
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
    return (
        f"baseline_v{BASELINE_CACHE_SCHEMA_VERSION}_"
        f"s{'-'.join(map(str, config.crop_sizes))}_"
        f"q{config.jpeg_quality}_"
        f"{digest}"
    )


def _cache_path(cache_dir: Path, file_name: str) -> Path:
    return cache_dir / f"{Path(file_name).stem}.md"


def _record_strategy(record: ImageRecord) -> str:
    return "long" if "long_rest_A" in record.source else "table"


def _candidate_min_chars(strategy: str, config: BaselineConfig) -> int:
    return config.long_min_chars if strategy == "long" else config.min_chars
