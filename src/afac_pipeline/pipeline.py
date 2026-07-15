"""Prediction orchestration, caching, retries, and CSV output."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .api import FinixDocClient, FinixDocError, normalize_markdown_payload
from .datasets import ImageRecord
from .images import make_vertical_slices


@dataclass(frozen=True)
class PredictionConfig:
    output_csv: Path
    cache_dir: Path
    dry_run: bool = False
    offset: int = 0
    limit: int | None = None
    sleep_seconds: float = 0.0
    resume: bool = True
    slice_height: int | None = None
    slice_overlap: int = 0
    max_width: int | None = None
    jpeg_quality: int = 95
    retries: int = 1
    retry_sleep_seconds: float = 60.0
    on_error: str = "raise"
    errors_csv: Path | None = None


@dataclass(frozen=True)
class PredictionStats:
    total_discovered: int
    processed: int
    cache_hits: int
    api_calls: int
    output_csv: Path


def run_prediction(
    *,
    records: Iterable[ImageRecord],
    client: FinixDocClient | None,
    config: PredictionConfig,
) -> PredictionStats:
    all_records = list(records)
    records_after_offset = all_records[config.offset :]
    selected_records = records_after_offset[: config.limit] if config.limit else records_after_offset

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if config.errors_csv:
        config.errors_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    cache_hits = 0
    api_calls = 0

    for index, record in enumerate(selected_records, start=1):
        cache_path = _cache_path(config.cache_dir, record.file_name)
        markdown: str

        try:
            cache_hit = False
            if config.resume and cache_path.exists():
                cached_text = cache_path.read_text(encoding="utf-8")
                try:
                    markdown = normalize_markdown_payload(cached_text)
                except FinixDocError as exc:
                    print(
                        f"[{index}/{len(selected_records)}] invalid cache "
                        f"{record.file_name}: {exc}"
                    )
                    cache_path.unlink()
                else:
                    if markdown != cached_text:
                        cache_path.write_text(markdown, encoding="utf-8")
                    cache_hits += 1
                    cache_hit = True
                    print(f"[{index}/{len(selected_records)}] cache hit {record.file_name}")

            if cache_hit:
                pass
            elif config.dry_run:
                markdown = _dry_run_markdown(record)
                cache_path.write_text(markdown, encoding="utf-8")
                print(f"[{index}/{len(selected_records)}] dry-run {record.file_name}")
            else:
                if client is None:
                    raise ValueError("FinixDocClient is required when dry_run is false")
                markdown, calls = _call_record(client, record, config)
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
            print(f"[{index}/{len(selected_records)}] ERROR {record.file_name}: {error['error']}")
            if config.on_error == "raise":
                _write_errors_if_needed(config.errors_csv, errors)
                raise
            if config.on_error == "message":
                markdown = f"ERROR: {error['error']}\n"
            elif config.on_error == "empty":
                markdown = ""
            else:
                raise ValueError(f"Unsupported on_error mode: {config.on_error}")

        rows.append({"file_name": record.file_name, "ground_truth": markdown})

    write_submission(config.output_csv, rows)
    _write_errors_if_needed(config.errors_csv, errors)
    return PredictionStats(
        total_discovered=len(all_records),
        processed=len(rows),
        cache_hits=cache_hits,
        api_calls=api_calls,
        output_csv=config.output_csv,
    )


def write_submission(output_csv: Path, rows: list[dict[str, str]]) -> None:
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["file_name", "ground_truth"])
        writer.writeheader()
        writer.writerows(rows)


def write_errors(errors_csv: Path, rows: list[dict[str, str]]) -> None:
    with errors_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["file_name", "source", "error"])
        writer.writeheader()
        writer.writerows(rows)


def _write_errors_if_needed(errors_csv: Path | None, rows: list[dict[str, str]]) -> None:
    if errors_csv and rows:
        write_errors(errors_csv, rows)


def _call_record(
    client: FinixDocClient,
    record: ImageRecord,
    config: PredictionConfig,
) -> tuple[str, int]:
    image_bytes = record.read_bytes()
    slices = make_vertical_slices(
        file_name=record.file_name,
        image_bytes=image_bytes,
        slice_height=config.slice_height,
        overlap=config.slice_overlap,
        max_width=config.max_width,
        jpeg_quality=config.jpeg_quality,
    )

    if len(slices) == 1 and slices[0].file_name == record.file_name:
        print(f"calling API {record.file_name}")
    else:
        print(
            f"calling API {record.file_name} as {len(slices)} vertical slices "
            f"(height={config.slice_height}, overlap={config.slice_overlap})"
        )

    parts: list[str] = []
    for slice_index, image_slice in enumerate(slices, start=1):
        if len(slices) > 1:
            print(
                f"  slice {slice_index}/{len(slices)} "
                f"{image_slice.file_name} y={image_slice.y0}:{image_slice.y1}"
            )
        parts.append(_call_with_retries(client, image_slice.file_name, image_slice.image_bytes, config))
        if config.sleep_seconds > 0 and slice_index < len(slices):
            time.sleep(config.sleep_seconds)
    return _merge_markdown_parts(parts), len(slices)


def _call_with_retries(
    client: FinixDocClient,
    file_name: str,
    image_bytes: bytes,
    config: PredictionConfig,
) -> str:
    attempt = 0
    while True:
        try:
            return client.call_with_file(file_name, image_bytes)
        except Exception:
            attempt += 1
            if attempt > config.retries:
                raise
            print(
                f"  retry {attempt}/{config.retries} after "
                f"{config.retry_sleep_seconds:.0f}s: {file_name}"
            )
            time.sleep(config.retry_sleep_seconds)


def _merge_markdown_parts(parts: list[str]) -> str:
    if not parts:
        return ""

    merged_lines = _strip_edge_blank_lines(parts[0].splitlines())
    for part in parts[1:]:
        next_lines = _strip_edge_blank_lines(part.splitlines())
        overlap = _line_overlap(merged_lines, next_lines, max_lines=30)
        merged_lines.extend(next_lines[overlap:])
    return "\n".join(merged_lines).strip() + "\n"


def _strip_edge_blank_lines(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _line_overlap(left: list[str], right: list[str], max_lines: int) -> int:
    limit = min(max_lines, len(left), len(right))
    for size in range(limit, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def _cache_path(cache_dir: Path, file_name: str) -> Path:
    stem = Path(file_name).stem
    return cache_dir / f"{stem}.md"


def _dry_run_markdown(record: ImageRecord) -> str:
    return (
        f"# {record.file_name}\n\n"
        "MVP dry-run placeholder. Run without `--dry-run` to call FinixDoc-VL.\n"
    )
