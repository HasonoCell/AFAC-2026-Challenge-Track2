"""Submission CSV validation utilities."""

from __future__ import annotations

import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .api import FinixDocError, html_table_structure_issue, normalize_markdown_payload
from .images import ImageSlice
from .merge import merge_sliced_markdown
from .tables import html_tables_to_markdown, parse_html_table, parse_sliced_table


SUBMISSION_COLUMNS = ["file_name", "ground_truth"]
DEFAULT_MAX_SIZE_BYTES = 100_000_000


def _ensure_csv_field_limit() -> None:
    limit = 10_000_000
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            if limit <= 1:
                raise
            limit = min(limit // 10, sys.maxsize)


_ensure_csv_field_limit()


@dataclass(frozen=True)
class SubmissionValidationResult:
    submission_csv: Path
    file_size_bytes: int
    row_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class SubmissionEnsembleResult:
    output_csv: Path
    row_count: int
    fallback_count: int
    fallback_file_names: tuple[str, ...]


@dataclass(frozen=True)
class SubmissionOverlayResult:
    output_csv: Path
    row_count: int
    override_count: int
    override_file_names: tuple[str, ...]
    skipped_count: int
    skipped_file_names: tuple[str, ...]


@dataclass(frozen=True)
class CachedGridRemergeResult:
    output_csv: Path
    row_count: int
    remerged_count: int
    remerged_file_names: tuple[str, ...]
    skipped_cached_file_names: tuple[str, ...]


@dataclass(frozen=True)
class SubmissionCompactionResult:
    """A submission rewritten to stay within a conservative per-cell budget."""

    output_csv: Path
    row_count: int
    compacted_count: int
    compacted_file_names: tuple[str, ...]
    max_field_bytes: int
    compact_all_html_tables: bool


_CACHED_GRID_TILE_RE = re.compile(
    r"^(?P<stem>.+)_content_r(?P<row>\d+)_c(?P<col>\d+)\.md$"
)
_HTML_TABLE_BLOCK = re.compile(
    r"<table\b[^>]*>.*?</table\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)


def remerge_cached_grid_submission(
    *,
    base_csv: Path,
    cache_roots: Iterable[Path],
    output_csv: Path,
    rows: int = 5,
    cols: int = 5,
    min_success_parts: int = 4,
    min_success_ratio: float = 0.60,
    max_duplicate_line_ratio: float = 0.30,
) -> CachedGridRemergeResult:
    """Rebuild multi-band table rows from cached OCR tiles without API calls.

    This operation is deliberately content-preserving.  It only replaces a
    base row when the cached tiles do not increase the HTML table count and
    keep the expanded cell grid identical.  Thus a cache remerge can improve
    table continuity and recover the tiles' ``th``/``td`` convention without
    silently changing OCR content or accepting an incomplete tile set.
    """

    if rows <= 0 or cols <= 0:
        raise ValueError("cached grid rows and cols must be positive")
    if min_success_parts < 0:
        raise ValueError("min_success_parts must be >= 0")
    if not 0 <= min_success_ratio <= 1:
        raise ValueError("min_success_ratio must be between 0 and 1")
    if not 0 <= max_duplicate_line_ratio <= 1:
        raise ValueError("max_duplicate_line_ratio must be between 0 and 1")

    base = _read_submission_rows(base_csv)
    cache_dirs = _index_cached_grid_directories(cache_roots)
    required_parts = max(min_success_parts, math.ceil(rows * cols * min_success_ratio))
    selected: dict[str, str] = {}
    skipped: list[str] = []

    for file_name, base_markdown in base.items():
        stem = Path(file_name).stem
        record_cache_dir = cache_dirs.get(stem)
        if record_cache_dir is None:
            continue
        parts_by_position = _read_cached_grid_parts(record_cache_dir, stem, rows, cols)
        if len(parts_by_position) < required_parts:
            skipped.append(file_name)
            continue
        slices: list[ImageSlice] = []
        parts: list[str] = []
        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                slices.append(
                    ImageSlice(
                        file_name=f"{stem}_content_r{row:03d}_c{col:03d}.jpg",
                        image_bytes=b"",
                        x0=col - 1,
                        x1=col,
                        y0=row - 1,
                        y1=row,
                        width=1,
                        height=1,
                        row=row,
                        col=col,
                        rows=rows,
                        cols=cols,
                    )
                )
                parts.append(parts_by_position.get((row, col), ""))

        remerged = merge_sliced_markdown(slices, parts)
        base_shape = _html_table_shape(base_markdown)
        remerged_shape = _html_table_shape(remerged)
        if (
            base_shape[0] == 0
            or remerged_shape[0] == 0
            or remerged_shape[0] > base_shape[0]
            or base_shape[1] != remerged_shape[1]
            or not _table_grid_equivalent(base_markdown, remerged)
            or _validate_ground_truth_text(remerged) is not None
            or _duplicate_line_ratio(remerged) > max_duplicate_line_ratio
        ):
            skipped.append(file_name)
            continue
        selected[file_name] = remerged

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        for file_name, markdown in base.items():
            writer.writerow(
                {
                    "file_name": file_name,
                    "ground_truth": selected.get(file_name, markdown),
                }
            )
    return CachedGridRemergeResult(
        output_csv=output_csv,
        row_count=len(base),
        remerged_count=len(selected),
        remerged_file_names=tuple(selected),
        skipped_cached_file_names=tuple(skipped),
    )


def compact_submission_for_platform(
    *,
    base_csv: Path,
    output_csv: Path,
    max_field_bytes: int,
    compact_all_html_tables: bool = False,
    allow_non_table_oversize: bool = False,
    allow_compacted_oversize: bool = False,
) -> SubmissionCompactionResult:
    """Compact oversized HTML table cells without discarding OCR text.

    Some competition evaluators are substantially less tolerant of a single
    enormous HTML table than of the same table represented in Markdown.  This
    operation normally touches only fields over the explicit byte budget.
    Processability mode can convert every complete HTML table block, and verifies
    every converted table retains exactly the same expanded header and cells.
    A field that cannot be compacted below the budget is an error rather than
    a silently truncated prediction.  The optional non-table exemption is for
    platforms whose practical failure mode is HTML-tree complexity rather than
    raw text length; it leaves long plain Markdown untouched and only applies
    the budget to fields that actually contain an HTML table.  In that mode,
    callers can also retain a compacted pipe table that remains longer than the
    selection threshold: the threshold is then an HTML-complexity control, not
    a raw CSV-field limit.
    """

    if max_field_bytes <= 0:
        raise ValueError("max_field_bytes must be positive")

    base = _read_submission_rows(base_csv)
    compacted: dict[str, str] = {}
    for file_name, markdown in base.items():
        should_compact = compact_all_html_tables and "<table" in markdown.lower()
        if not should_compact and len(markdown.encode("utf-8")) <= max_field_bytes:
            continue
        if allow_non_table_oversize and "<table" not in markdown.lower():
            continue
        rewritten = _compact_html_tables(markdown)
        if (
            len(rewritten.encode("utf-8")) > max_field_bytes
            and not allow_compacted_oversize
        ):
            raise ValueError(
                "submission field remains above the platform budget after "
                f"lossless table compaction: {file_name} "
                f"({len(rewritten.encode('utf-8'))} > {max_field_bytes} bytes)"
            )
        issue = _validate_ground_truth_text(rewritten)
        if issue is not None:
            raise ValueError(
                "table compaction produced malformed markdown for "
                f"{file_name}: {issue}"
            )
        compacted[file_name] = rewritten

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        for file_name, markdown in base.items():
            writer.writerow(
                {
                    "file_name": file_name,
                    "ground_truth": compacted.get(file_name, markdown),
                }
            )
    return SubmissionCompactionResult(
        output_csv=output_csv,
        row_count=len(base),
        compacted_count=len(compacted),
        compacted_file_names=tuple(compacted),
        max_field_bytes=max_field_bytes,
        compact_all_html_tables=compact_all_html_tables,
    )


def _compact_html_tables(markdown: str) -> str:
    """Replace complete HTML table blocks by value-equivalent pipe tables."""

    if not _HTML_TABLE_BLOCK.search(markdown):
        raise ValueError("oversized submission field has no complete HTML table block")
    try:
        return html_tables_to_markdown(markdown)
    except ValueError as exc:
        raise ValueError(f"HTML table compaction failed: {exc}") from exc


def _index_cached_grid_directories(cache_roots: Iterable[Path]) -> dict[str, Path]:
    indexed: dict[str, tuple[int, Path]] = {}
    for cache_root in cache_roots:
        if not cache_root.exists():
            raise ValueError(f"cache root does not exist: {cache_root}")
        counts: dict[tuple[str, Path], int] = {}
        for tile_path in cache_root.rglob("*_content_r*_c*.md"):
            match = _CACHED_GRID_TILE_RE.fullmatch(tile_path.name)
            if match is None:
                continue
            key = (match.group("stem"), tile_path.parent)
            counts[key] = counts.get(key, 0) + 1
        for (stem, directory), count in counts.items():
            current = indexed.get(stem)
            if current is None or count > current[0]:
                indexed[stem] = (count, directory)
    return {stem: directory for stem, (_, directory) in indexed.items()}


def _read_cached_grid_parts(
    cache_dir: Path,
    stem: str,
    rows: int,
    cols: int,
) -> dict[tuple[int, int], str]:
    parts: dict[tuple[int, int], str] = {}
    for tile_path in cache_dir.glob(f"{stem}_content_r*_c*.md"):
        match = _CACHED_GRID_TILE_RE.fullmatch(tile_path.name)
        if match is None:
            continue
        row = int(match.group("row"))
        col = int(match.group("col"))
        if not (1 <= row <= rows and 1 <= col <= cols):
            continue
        parts[(row, col)] = normalize_markdown_payload(
            tile_path.read_text(encoding="utf-8")
        )
    return parts


def _html_table_shape(markdown: str) -> tuple[int, int, int]:
    return (
        _html_open_tag_count(markdown, "table"),
        _html_open_tag_count(markdown, "tr"),
        _html_open_tag_count(markdown, "td") + _html_open_tag_count(markdown, "th"),
    )


def _html_open_tag_count(markdown: str, tag: str) -> int:
    return len(
        re.findall(
            rf"<\s*{tag}(?:\s[^<>]*)?>",
            markdown,
            flags=re.IGNORECASE,
        )
    )


def _table_grid_equivalent(left: str, right: str) -> bool:
    """Compare reconstructed cell grids instead of raw HTML tag counts.

    Joining ragged triangular bands into one table adds explicit trailing blank
    cells.  Those cells are structural padding, not new OCR content.  The
    sliced-table parser expands both representations to the same rectangular
    grid, so equality here preserves every recognized value and its position
    while allowing the safer one-table serialization.
    """

    left_table = parse_sliced_table(left)
    right_table = parse_sliced_table(right)
    return (
        left_table is not None
        and right_table is not None
        and left_table.header == right_table.header
        and left_table.rows == right_table.rows
    )


def build_submission_overlay(
    *,
    base_csv: Path,
    override_csvs: Iterable[Path],
    output_csv: Path,
    min_override_to_base_ratio: float = 0.0,
    min_override_char_gain: int = 0,
    max_override_duplicate_line_ratio: float | None = None,
) -> SubmissionOverlayResult:
    """Replace a checked subset of a full submission while preserving its order.

    Each override file may contain only the records it recomputed. This avoids
    regenerating unrelated documents when a validated, route-specific repair is
    being evaluated. Every override name must already exist in the base file,
    and duplicate overrides are rejected to make the result deterministic.

    ``min_override_to_base_ratio`` provides a final generic guard against
    nondeterministic OCR regressions. At ``1.0``, a partial result must be at
    least as long as the existing complete-submission result before it can
    replace it. Malformed table markup is also rejected here: an overlay must
    never turn a platform-safe base submission into an unparseable one.
    ``min_override_char_gain`` filters trivial non-deterministic changes that
    merely happen to be a few characters longer than the base output.
    ``max_override_duplicate_line_ratio`` is an optional anti-runaway guard for
    tiled OCR: it rejects an otherwise well-formed override when nearly all of
    its nonblank lines are repeated.  This is deliberately opt-in because
    some document formats use repeated short lines legitimately.
    """
    if min_override_to_base_ratio < 0:
        raise ValueError("min_override_to_base_ratio must be >= 0")
    if min_override_char_gain < 0:
        raise ValueError("min_override_char_gain must be >= 0")
    if (
        max_override_duplicate_line_ratio is not None
        and not 0 <= max_override_duplicate_line_ratio <= 1
    ):
        raise ValueError("max_override_duplicate_line_ratio must be between 0 and 1")
    base = _read_submission_rows(base_csv)
    overrides: dict[str, str] = {}
    for override_csv in override_csvs:
        for file_name, markdown in _read_submission_rows(override_csv).items():
            if file_name not in base:
                raise ValueError(
                    f"override file_name {file_name!r} is absent from base submission: "
                    f"{override_csv}"
                )
            if file_name in overrides:
                raise ValueError(f"duplicate override file_name {file_name!r}")
            overrides[file_name] = markdown

    selected_overrides: dict[str, str] = {}
    skipped_file_names: list[str] = []
    for file_name, markdown in overrides.items():
        base_chars = len(base[file_name].strip())
        override_chars = len(markdown.strip())
        if (
            override_chars < base_chars * min_override_to_base_ratio
            or override_chars - base_chars < min_override_char_gain
            or _validate_ground_truth_text(markdown) is not None
            or (
                max_override_duplicate_line_ratio is not None
                and _duplicate_line_ratio(markdown) > max_override_duplicate_line_ratio
            )
        ):
            skipped_file_names.append(file_name)
            continue
        selected_overrides[file_name] = markdown

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        for file_name, base_markdown in base.items():
            writer.writerow(
                {
                    "file_name": file_name,
                    "ground_truth": selected_overrides.get(file_name, base_markdown),
                }
            )
    return SubmissionOverlayResult(
        output_csv=output_csv,
        row_count=len(base),
        override_count=len(selected_overrides),
        override_file_names=tuple(selected_overrides),
        skipped_count=len(skipped_file_names),
        skipped_file_names=tuple(skipped_file_names),
    )


def _duplicate_line_ratio(markdown: str) -> float:
    """Measure repeated nonblank lines without counting whitespace differences."""
    lines = [" ".join(line.split()) for line in markdown.splitlines() if line.strip()]
    if not lines:
        return 1.0
    return 1.0 - (len(set(lines)) / len(lines))


def build_conservative_submission_ensemble(
    *,
    primary_csv: Path,
    fallback_csv: Path,
    output_csv: Path,
    max_primary_chars: int = 600,
    max_primary_to_fallback_ratio: float = 0.10,
    require_fallback_html_table: bool = True,
) -> SubmissionEnsembleResult:
    """Keep the primary submission except for objectively collapsed outputs.

    This is intentionally a narrow safety ensemble rather than a length-based
    ranking system: a fallback can replace the primary result only when the
    primary is both absolutely tiny and a small fraction of the fallback.
    """
    if max_primary_chars < 0:
        raise ValueError("max_primary_chars must be >= 0")
    if not 0 <= max_primary_to_fallback_ratio <= 1:
        raise ValueError("max_primary_to_fallback_ratio must be between 0 and 1")

    primary = _read_submission_rows(primary_csv)
    fallback = _read_submission_rows(fallback_csv)
    primary_names = set(primary)
    fallback_names = set(fallback)
    if primary_names != fallback_names:
        missing_fallback = sorted(primary_names - fallback_names)
        missing_primary = sorted(fallback_names - primary_names)
        raise ValueError(
            "submission file names do not match; "
            f"missing from fallback: {_format_sample(missing_fallback)}; "
            f"missing from primary: {_format_sample(missing_primary)}"
        )

    selected_fallback: list[str] = []
    rows: list[dict[str, str]] = []
    for file_name, primary_markdown in primary.items():
        fallback_markdown = fallback[file_name]
        if _should_use_fallback_submission(
            primary_markdown=primary_markdown,
            fallback_markdown=fallback_markdown,
            max_primary_chars=max_primary_chars,
            max_primary_to_fallback_ratio=max_primary_to_fallback_ratio,
            require_fallback_html_table=require_fallback_html_table,
        ):
            markdown = fallback_markdown
            selected_fallback.append(file_name)
        else:
            markdown = primary_markdown
        rows.append({"file_name": file_name, "ground_truth": markdown})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return SubmissionEnsembleResult(
        output_csv=output_csv,
        row_count=len(rows),
        fallback_count=len(selected_fallback),
        fallback_file_names=tuple(selected_fallback),
    )


def validate_submission_csv(
    *,
    submission_csv: Path,
    expected_file_names: Iterable[str],
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
    max_field_bytes: int | None = None,
    allow_empty: bool = False,
    expected_label: str = "A-list",
) -> SubmissionValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    expected_names = set(expected_file_names)
    if max_field_bytes is not None and max_field_bytes <= 0:
        raise ValueError("max_field_bytes must be positive when specified")

    if not submission_csv.exists():
        return SubmissionValidationResult(
            submission_csv=submission_csv,
            file_size_bytes=0,
            row_count=0,
            errors=(f"submission file does not exist: {submission_csv}",),
            warnings=(),
        )

    file_size = submission_csv.stat().st_size
    if file_size > max_size_bytes:
        errors.append(
            f"submission file is {file_size} bytes, exceeding limit {max_size_bytes} bytes"
        )

    try:
        with submission_csv.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except csv.Error as exc:
        return SubmissionValidationResult(
            submission_csv=submission_csv,
            file_size_bytes=file_size,
            row_count=0,
            errors=tuple(errors + [f"failed to parse CSV: {exc}"]),
            warnings=tuple(warnings),
        )
    except UnicodeDecodeError as exc:
        return SubmissionValidationResult(
            submission_csv=submission_csv,
            file_size_bytes=file_size,
            row_count=0,
            errors=tuple(errors + [f"CSV must be UTF-8 encoded: {exc}"]),
            warnings=tuple(warnings),
        )

    if fieldnames != SUBMISSION_COLUMNS:
        errors.append(
            "CSV header must be exactly "
            f"{','.join(SUBMISSION_COLUMNS)}; got {','.join(fieldnames) or '<empty>'}"
        )

    seen: set[str] = set()
    duplicate_names: set[str] = set()
    actual_names: set[str] = set()
    empty_names = 0
    empty_outputs: list[str] = []
    error_outputs: list[str] = []
    malformed_outputs: list[str] = []
    oversized_outputs: list[str] = []

    for row_index, row in enumerate(rows, start=2):
        file_name = (row.get("file_name") or "").strip()
        ground_truth = row.get("ground_truth") or ""
        if not file_name:
            empty_names += 1
            continue
        if file_name in seen:
            duplicate_names.add(file_name)
        seen.add(file_name)
        actual_names.add(file_name)

        if not ground_truth.strip():
            empty_outputs.append(file_name)
        if ground_truth.lstrip().startswith("ERROR:"):
            error_outputs.append(file_name)
        if (
            max_field_bytes is not None
            and len(ground_truth.encode("utf-8")) > max_field_bytes
        ):
            oversized_outputs.append(file_name)
        issue = _validate_ground_truth_text(ground_truth)
        if issue:
            malformed_outputs.append(f"{file_name} (row {row_index}: {issue})")

    missing_names = sorted(expected_names - actual_names)
    unknown_names = sorted(actual_names - expected_names)

    if empty_names:
        errors.append(f"{empty_names} row(s) have an empty file_name")
    if duplicate_names:
        errors.append(
            f"duplicate file_name values: {_format_sample(sorted(duplicate_names))}"
        )
    if missing_names:
        errors.append(
            f"missing expected {expected_label} files: {_format_sample(missing_names)}"
        )
    if unknown_names:
        errors.append(f"unknown file_name values: {_format_sample(unknown_names)}")
    if empty_outputs:
        message = f"empty ground_truth values: {_format_sample(empty_outputs)}"
        if allow_empty:
            warnings.append(message)
        else:
            errors.append(message)
    if error_outputs:
        errors.append(f"ground_truth contains ERROR markers: {_format_sample(error_outputs)}")
    if oversized_outputs:
        errors.append(
            "ground_truth exceeds per-field UTF-8 byte budget "
            f"({max_field_bytes}): {_format_sample(oversized_outputs)}"
        )
    if malformed_outputs:
        errors.append(
            "malformed ground_truth values: " + _format_sample(malformed_outputs, limit=5)
        )

    if len(rows) != len(expected_names):
        errors.append(
            f"row count is {len(rows)}, expected {len(expected_names)} for current {expected_label} data"
        )

    if file_size > 90_000_000 and file_size <= max_size_bytes:
        warnings.append(
            f"submission file is {file_size} bytes; close to 100MB platform limit"
        )

    return SubmissionValidationResult(
        submission_csv=submission_csv,
        file_size_bytes=file_size,
        row_count=len(rows),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _read_submission_rows(submission_csv: Path) -> dict[str, str]:
    with submission_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != SUBMISSION_COLUMNS:
            raise ValueError(
                "submission CSV header must be exactly "
                f"{','.join(SUBMISSION_COLUMNS)}: {submission_csv}"
            )
        rows: dict[str, str] = {}
        for row in reader:
            file_name = (row.get("file_name") or "").strip()
            if not file_name:
                raise ValueError(f"submission contains an empty file_name: {submission_csv}")
            if file_name in rows:
                raise ValueError(f"submission contains duplicate file_name {file_name}: {submission_csv}")
            rows[file_name] = row.get("ground_truth") or ""
    return rows


def _should_use_fallback_submission(
    *,
    primary_markdown: str,
    fallback_markdown: str,
    max_primary_chars: int,
    max_primary_to_fallback_ratio: float,
    require_fallback_html_table: bool,
) -> bool:
    primary_chars = len(primary_markdown.strip())
    fallback_chars = len(fallback_markdown.strip())
    if primary_chars > max_primary_chars or fallback_chars <= 0:
        return False
    if primary_chars >= fallback_chars * max_primary_to_fallback_ratio:
        return False
    return not require_fallback_html_table or "<table" in fallback_markdown.lower()


def format_validation_result(result: SubmissionValidationResult) -> str:
    status = "OK" if result.ok else "FAIL"
    lines = [
        f"status: {status}",
        f"submission_csv: {result.submission_csv}",
        f"file_size_bytes: {result.file_size_bytes}",
        f"row_count: {result.row_count}",
    ]
    if result.errors:
        lines.append("errors:")
        lines.extend(f"- {error}" for error in result.errors)
    if result.warnings:
        lines.append("warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)


def _validate_ground_truth_text(text: str) -> str | None:
    if "MVP dry-run placeholder" in text:
        return "dry-run placeholder output must not be submitted"

    try:
        normalize_markdown_payload(text)
    except FinixDocError as exc:
        return str(exc)

    return _validate_html_table_structure(text)


def _validate_html_table_structure(text: str) -> str | None:
    return html_table_structure_issue(text)


def _format_sample(values: list[str], limit: int = 10) -> str:
    sample = values[:limit]
    suffix = "" if len(values) <= limit else f" ... (+{len(values) - limit} more)"
    return ", ".join(sample) + suffix
