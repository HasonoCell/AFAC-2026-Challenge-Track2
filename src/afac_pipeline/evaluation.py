"""Local proxy evaluation for training-set prediction CSV files."""

from __future__ import annotations

import csv
import difflib
import hashlib
import math
import re
import sys
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from statistics import mean, median
from typing import Iterable

from .submission import SUBMISSION_COLUMNS
from .tables import MarkdownTable, parse_html_table, parse_markdown_pipe_table


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


_CHAR_SIMILARITY_LIMIT = 20_000
_MAX_SEQUENCE_ITEMS = 4_096
# ``difflib.SequenceMatcher`` has quadratic worst-case behavior.  Long OCR
# tables frequently have 10k+ short numeric tokens, so the former 20k limit
# could make a small subset experiment exhaust memory before any metric was
# emitted.  Keep exact token alignment only at a deliberately conservative
# size; larger inputs use the bounded fingerprints below.
_LONG_TEXT_TOKEN_LIMIT = 2_048
_MIN_CHUNK_SIZE = 128
_HTML_ROW_BLOCK = re.compile(
    r"<tr\b[^>]*>(.*?)</tr\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
_HTML_TABLE_BLOCK = re.compile(
    r"<table\b[^>]*>.*?</table\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
_HTML_CELL_BLOCK = re.compile(
    r"<t[dh]\b[^>]*>(.*?)</t[dh]\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class TableStats:
    table_count: int
    rows: int
    cells: int
    header_cells: int
    data_cells: int
    max_columns: int


@dataclass(frozen=True)
class EvaluationRow:
    file_name: str
    prediction_chars: int
    ground_truth_chars: int
    text_similarity: float
    markdown_similarity: float
    read_order_similarity: float
    table_structure_score: float
    overall_proxy: float


@dataclass(frozen=True)
class EvaluationSummary:
    prediction_csv: Path
    evaluated: int
    missing_predictions: tuple[str, ...]
    unknown_predictions: tuple[str, ...]
    rows: tuple[EvaluationRow, ...]

    @property
    def overall_mean(self) -> float:
        return _mean(row.overall_proxy for row in self.rows)

    @property
    def overall_median(self) -> float:
        return _median(row.overall_proxy for row in self.rows)

    @property
    def text_mean(self) -> float:
        return _mean(row.text_similarity for row in self.rows)

    @property
    def table_mean(self) -> float:
        return _mean(row.table_structure_score for row in self.rows)

    @property
    def read_order_mean(self) -> float:
        return _mean(row.read_order_similarity for row in self.rows)


def evaluate_prediction_csv(
    *,
    prediction_csv: Path,
    ground_truths: dict[str, str],
) -> EvaluationSummary:
    predictions = read_prediction_csv(prediction_csv)
    expected_names = set(ground_truths)
    actual_names = set(predictions)
    rows = tuple(
        evaluate_pair(
            file_name=file_name,
            prediction=predictions[file_name],
            ground_truth=ground_truths[file_name],
        )
        for file_name in sorted(expected_names & actual_names)
    )
    return EvaluationSummary(
        prediction_csv=prediction_csv,
        evaluated=len(rows),
        missing_predictions=tuple(sorted(expected_names - actual_names)),
        unknown_predictions=tuple(sorted(actual_names - expected_names)),
        rows=rows,
    )


def read_prediction_csv(prediction_csv: Path) -> dict[str, str]:
    with prediction_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != SUBMISSION_COLUMNS:
            raise ValueError(
                "prediction CSV header must be exactly "
                f"{','.join(SUBMISSION_COLUMNS)}"
            )
        return {
            row["file_name"]: row["ground_truth"]
            for row in reader
            if row.get("file_name")
        }


def evaluate_pair(
    *,
    file_name: str,
    prediction: str,
    ground_truth: str,
) -> EvaluationRow:
    normalized_prediction = _normalize_text(prediction)
    normalized_ground_truth = _normalize_text(ground_truth)
    aligned_table_similarity = None
    if max(len(normalized_prediction), len(normalized_ground_truth)) > _CHAR_SIMILARITY_LIMIT:
        aligned_table_similarity = _aligned_table_text_similarity(
            prediction,
            ground_truth,
        )
    text_similarity = (
        aligned_table_similarity
        if aligned_table_similarity is not None
        else _similarity(normalized_prediction, normalized_ground_truth)
    )
    markdown_similarity = _similarity(
        _normalize_markdown(prediction),
        _normalize_markdown(ground_truth),
    )
    read_order_similarity = _sequence_similarity(
        _read_order_blocks(prediction),
        _read_order_blocks(ground_truth),
    )
    table_structure_score = _table_structure_score(
        _table_stats(prediction),
        _table_stats(ground_truth),
    )
    overall_proxy = mean(
        [
            text_similarity,
            read_order_similarity,
            table_structure_score,
        ]
    )
    return EvaluationRow(
        file_name=file_name,
        prediction_chars=len(prediction),
        ground_truth_chars=len(ground_truth),
        text_similarity=text_similarity,
        markdown_similarity=markdown_similarity,
        read_order_similarity=read_order_similarity,
        table_structure_score=table_structure_score,
        overall_proxy=overall_proxy,
    )


def _aligned_table_text_similarity(left: str, right: str) -> float | None:
    """Compare very large, shape-aligned tables cell by cell.

    The old bounded character fingerprints were intentionally cheap, but one
    missing early value shifted every later fingerprint and could score a
    400-row table near zero despite most cells being exactly correct.  When
    table and row topology already agree, local cell alignment is both bounded
    and a materially closer proxy for the official text-edit component.
    """

    left_tables = _table_text_grids(left)
    right_tables = _table_text_grids(right)
    if not left_tables or len(left_tables) != len(right_tables):
        return None

    # Tables are compared cell-by-cell below.  They must be removed from both
    # representations here: B submissions are pipe Markdown while train GTs
    # are often HTML, and stripping only HTML makes every pipe table count a
    # second time as unrelated prose.
    left_context = _non_table_context(left)
    right_context = _non_table_context(right)
    weighted_score = 0.0
    total_weight = 0

    def add_pair(left_text: str, right_text: str) -> None:
        nonlocal weighted_score, total_weight
        weight = max(1, len(left_text), len(right_text))
        score = (
            100.0
            if left_text == right_text
            else _similarity(left_text, right_text)
        )
        weighted_score += score * weight
        total_weight += weight

    def add_row(left_row: list[str], right_row: list[str]) -> None:
        if len(left_row) == len(right_row):
            for left_text, right_text in zip(left_row, right_row):
                add_pair(left_text, right_text)
            return

        # A single omitted OCR cell shifts every later cell, but the matching
        # prefix/suffix remains direct evidence.  Preserve it before comparing
        # only the small unmatched middle; this avoids a quadratic LCS over a
        # 5k-cell row while retaining the intended one-cell-shift tolerance.
        prefix = 0
        while (
            prefix < len(left_row)
            and prefix < len(right_row)
            and left_row[prefix] == right_row[prefix]
        ):
            add_pair(left_row[prefix], right_row[prefix])
            prefix += 1
        left_end = len(left_row)
        right_end = len(right_row)
        suffix_pairs: list[tuple[str, str]] = []
        while (
            left_end > prefix
            and right_end > prefix
            and left_row[left_end - 1] == right_row[right_end - 1]
        ):
            left_end -= 1
            right_end -= 1
            suffix_pairs.append((left_row[left_end], right_row[right_end]))
        left_middle = " ".join(left_row[prefix:left_end])
        right_middle = " ".join(right_row[prefix:right_end])
        if left_middle or right_middle:
            add_pair(left_middle, right_middle)
        for left_text, right_text in reversed(suffix_pairs):
            add_pair(left_text, right_text)

    if left_context or right_context:
        add_pair(left_context, right_context)
    for left_rows, right_rows in zip(left_tables, right_tables):
        # Byte-safe submission budgets can retain a complete table header and
        # a complete prefix of data rows.  Treat omitted rows as empty cells
        # instead of falling back to brittle fixed character fingerprints.
        # This remains bounded, preserves row order, and charges every absent
        # ground-truth cell as a miss rather than granting it a free match.
        for left_row, right_row in zip_longest(left_rows, right_rows, fillvalue=[]):
            add_row(left_row, right_row)
    if not total_weight:
        return None
    return weighted_score / total_weight


def _table_text_grids(text: str) -> list[list[list[str]]]:
    return [
        [
            [_normalize_text(cell) for cell in table.header],
            *[[_normalize_text(cell) for cell in row] for row in table.rows],
        ]
        for table in _document_tables(text)
    ]


def _non_table_context(markdown: str) -> str:
    """Normalize only prose outside complete HTML or pipe tables."""

    without_html = _HTML_TABLE_BLOCK.sub("\n", markdown)
    lines = without_html.splitlines()
    retained: list[str] = []
    index = 0
    while index < len(lines):
        if index + 1 < len(lines) and (
            _is_pipe_table_row(lines[index].strip())
            and _is_pipe_separator(lines[index + 1].strip())
        ):
            width = len(_split_pipe_row(lines[index].strip()))
            index += 2
            while (
                index < len(lines)
                and _is_pipe_table_row(lines[index].strip())
                and len(_split_pipe_row(lines[index].strip())) == width
            ):
                index += 1
            continue
        retained.append(lines[index])
        index += 1
    return _normalize_text("\n".join(retained))


def write_evaluation_rows(output_csv: Path, rows: Iterable[EvaluationRow]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file_name",
                "prediction_chars",
                "ground_truth_chars",
                "text_similarity",
                "markdown_similarity",
                "read_order_similarity",
                "table_structure_score",
                "overall_proxy",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "file_name": row.file_name,
                    "prediction_chars": row.prediction_chars,
                    "ground_truth_chars": row.ground_truth_chars,
                    "text_similarity": f"{row.text_similarity:.6f}",
                    "markdown_similarity": f"{row.markdown_similarity:.6f}",
                    "read_order_similarity": f"{row.read_order_similarity:.6f}",
                    "table_structure_score": f"{row.table_structure_score:.6f}",
                    "overall_proxy": f"{row.overall_proxy:.6f}",
                }
            )


def format_evaluation_summary(summary: EvaluationSummary, *, worst_k: int = 10) -> str:
    lines = [
        f"prediction_csv: {summary.prediction_csv}",
        f"evaluated: {summary.evaluated}",
        f"missing_predictions: {len(summary.missing_predictions)}",
        f"unknown_predictions: {len(summary.unknown_predictions)}",
        f"overall_proxy_mean: {summary.overall_mean:.4f}",
        f"overall_proxy_median: {summary.overall_median:.4f}",
        f"text_similarity_mean: {summary.text_mean:.4f}",
        f"table_structure_mean: {summary.table_mean:.4f}",
        f"read_order_similarity_mean: {summary.read_order_mean:.4f}",
    ]
    worst_rows = sorted(summary.rows, key=lambda row: row.overall_proxy)[:worst_k]
    if worst_rows:
        lines.append("worst_samples:")
        lines.extend(
            "- "
            f"{row.file_name}: overall={row.overall_proxy:.2f}, "
            f"text={row.text_similarity:.2f}, "
            f"table={row.table_structure_score:.2f}, "
            f"read_order={row.read_order_similarity:.2f}, "
            f"chars={row.prediction_chars}/{row.ground_truth_chars}"
            for row in worst_rows
        )
    if summary.missing_predictions:
        lines.append("missing_sample: " + _format_sample(summary.missing_predictions))
    if summary.unknown_predictions:
        lines.append("unknown_sample: " + _format_sample(summary.unknown_predictions))
    return "\n".join(lines)


def _similarity(left: str, right: str) -> float:
    if not left and not right:
        return 100.0
    if not left or not right:
        return 0.0
    if max(len(left), len(right)) <= _CHAR_SIMILARITY_LIMIT:
        return _sequence_matcher_ratio(left, right)

    # Fixed character chunks are extremely brittle for long OCR tables: one
    # missing cell shifts every following chunk and can turn an otherwise
    # useful reconstruction into an apparent zero.  Match whitespace-delimited
    # OCR tokens first when the sequence is bounded; this preserves order while
    # tolerating local insertions and deletions.  Unbroken payloads still use
    # bounded character fingerprints below.
    left_tokens = left.split()
    right_tokens = right.split()
    if (
        max(len(left_tokens), len(right_tokens)) <= _LONG_TEXT_TOKEN_LIMIT
        and max(len(left_tokens), len(right_tokens)) > 1
    ):
        return _sequence_matcher_ratio(left_tokens, right_tokens)

    chunk_size = max(
        _MIN_CHUNK_SIZE,
        math.ceil(max(len(left), len(right)) / _MAX_SEQUENCE_ITEMS),
    )
    return _sequence_matcher_ratio(
        _chunk_fingerprints(left, chunk_size),
        _chunk_fingerprints(right, chunk_size),
    )


def _sequence_similarity(left: list[str], right: list[str]) -> float:
    if not left and not right:
        return 100.0
    if not left or not right:
        return 0.0
    if max(len(left), len(right)) <= _MAX_SEQUENCE_ITEMS:
        return _sequence_matcher_ratio(left, right)

    group_size = math.ceil(max(len(left), len(right)) / _MAX_SEQUENCE_ITEMS)
    return _sequence_matcher_ratio(
        _sequence_fingerprints(left, group_size),
        _sequence_fingerprints(right, group_size),
    )


def _sequence_matcher_ratio(left: object, right: object) -> float:
    return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio() * 100


def _chunk_fingerprints(text: str, chunk_size: int) -> list[bytes]:
    return [
        hashlib.blake2b(
            text[start : start + chunk_size].encode("utf-8"),
            digest_size=8,
        ).digest()
        for start in range(0, len(text), chunk_size)
    ]


def _sequence_fingerprints(values: list[str], group_size: int) -> list[bytes]:
    return [
        hashlib.blake2b(
            "\n".join(values[start : start + group_size]).encode("utf-8"),
            digest_size=8,
        ).digest()
        for start in range(0, len(values), group_size)
    ]


def _normalize_text(markdown: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", markdown, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"```[a-zA-Z0-9_-]*", " ", text)
    text = text.replace("|", " ")
    text = re.sub(r"(?m)^\s*:?-{3,}:?(?:\s+:?-{3,}:?)*\s*$", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_markdown(markdown: str) -> str:
    return re.sub(r"\s+", " ", markdown).strip()


def _read_order_blocks(markdown: str) -> list[str]:
    # A whole HTML table may arrive on one physical line or one ``tr`` per
    # line.  Use the first few stable row-key cells for table reading order and
    # ordinary normalized lines for surrounding prose.  Comparing every cell
    # makes a single OCR error turn the whole row into an unrelated block.
    blocks: list[str] = []
    cursor = 0
    for row_match in _HTML_ROW_BLOCK.finditer(markdown):
        blocks.extend(_non_html_table_order_blocks(markdown[cursor : row_match.start()]))
        cells = [
            _normalize_text(cell)
            for cell in _HTML_CELL_BLOCK.findall(row_match.group(1))
        ]
        row_key = " | ".join([cell for cell in cells if cell][:4])
        if row_key:
            blocks.append(row_key)
        cursor = row_match.end()
    blocks.extend(_non_html_table_order_blocks(markdown[cursor:]))
    return blocks


def _non_html_table_order_blocks(markdown: str) -> list[str]:
    blocks: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if _is_pipe_table_row(stripped):
            if _is_pipe_separator(stripped):
                continue
            cells = [_normalize_text(cell) for cell in _split_pipe_row(stripped)]
            normalized = " | ".join([cell for cell in cells if cell][:4])
        else:
            normalized = _normalize_text(line)
        if normalized:
            blocks.append(normalized)
    return blocks


def _table_stats(markdown: str) -> TableStats:
    tables = _document_tables(markdown)
    return TableStats(
        table_count=len(tables),
        rows=sum(1 + len(table.rows) for table in tables),
        cells=sum(
            len(table.header) + sum(len(row) for row in table.rows)
            for table in tables
        ),
        header_cells=sum(len(table.header) for table in tables),
        data_cells=sum(sum(len(row) for row in table.rows) for table in tables),
        max_columns=max((len(table.header) for table in tables), default=0),
    )


def _table_structure_score(left: TableStats, right: TableStats) -> float:
    scores = [
        _count_similarity(left.table_count, right.table_count),
        _count_similarity(left.rows, right.rows),
        _count_similarity(left.cells, right.cells),
        _count_similarity(left.header_cells, right.header_cells),
        _count_similarity(left.data_cells, right.data_cells),
        _count_similarity(left.max_columns, right.max_columns),
    ]
    return mean(scores)


def _document_tables(markdown: str) -> list[MarkdownTable]:
    """Extract table grids independent of HTML versus pipe serialization.

    B-list submissions are pipe-only, while older train labels and raw OCR
    cache can contain HTML.  A local metric that rewards HTML tags themselves
    would steer the pipeline toward an un-submittable format.  Parse both
    notations into the same expanded cell grid before scoring topology.
    """

    tables: list[MarkdownTable] = []
    for match in _HTML_TABLE_BLOCK.finditer(markdown):
        table = parse_html_table(match.group(0))
        if table is not None:
            tables.append(table)

    without_html = _HTML_TABLE_BLOCK.sub("\n", markdown)
    lines = without_html.splitlines()
    index = 0
    while index + 1 < len(lines):
        if not (
            _is_pipe_table_row(lines[index].strip())
            and _is_pipe_separator(lines[index + 1].strip())
        ):
            index += 1
            continue
        width = len(_split_pipe_row(lines[index].strip()))
        block = [lines[index].strip(), lines[index + 1].strip()]
        index += 2
        while (
            index < len(lines)
            and _is_pipe_table_row(lines[index].strip())
            and len(_split_pipe_row(lines[index].strip())) == width
        ):
            block.append(lines[index].strip())
            index += 1
        table = parse_markdown_pipe_table("\n".join(block))
        if table is not None:
            tables.append(table)
    return tables


def _count_similarity(left: int, right: int) -> float:
    if left == right == 0:
        return 100.0
    return max(0.0, 100.0 * (1.0 - abs(left - right) / max(left, right)))


def _html_span_values(markdown: str, attribute: str) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in re.findall(
            rf"\b{attribute}\s*=\s*[\"']?(\d+)",
            markdown,
            flags=re.IGNORECASE,
        )
        if int(value) >= 1
    )


def _html_open_tag_count(markdown: str, tag: str) -> int:
    return len(
        re.findall(
            rf"<\s*{tag}(?:\s[^<>]*)?>",
            markdown,
            flags=re.IGNORECASE,
        )
    )


def _is_pipe_table_row(line: str) -> bool:
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 2


def _is_pipe_separator(line: str) -> bool:
    cells = _split_pipe_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def _split_pipe_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return mean(items) if items else 0.0


def _median(values: Iterable[float]) -> float:
    items = list(values)
    return median(items) if items else 0.0


def _format_sample(values: tuple[str, ...], limit: int = 10) -> str:
    sample = values[:limit]
    suffix = "" if len(values) <= limit else f" ... (+{len(values) - limit} more)"
    return ", ".join(sample) + suffix
