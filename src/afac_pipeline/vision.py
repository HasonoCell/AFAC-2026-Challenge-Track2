"""Coordinate reconstruction plus a macOS Vision diagnostic OCR adapter."""

from __future__ import annotations

import bisect
import hashlib
import math
import platform
import re
import shutil
import statistics
import subprocess
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from html import escape
from pathlib import Path

from .images import ImageSlice
from .tables import MarkdownTable, table_to_html


class VisionOCRError(RuntimeError):
    pass


@dataclass(frozen=True)
class VisionObservation:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    text: str


@dataclass(frozen=True)
class VisionMatrixResult:
    markdown: str
    rows: int
    cols: int
    populated_cells: int
    total_cells: int
    header_sequence_inliers: int
    row_sequence_inliers: int
    table_count: int = 1
    row_starts: tuple[int, ...] = ()
    header_starts: tuple[int, ...] = ()
    top_y: float = 0.0
    bottom_y: float = 0.0

    @property
    def coverage(self) -> float:
        if self.total_cells <= 0:
            return 0.0
        return self.populated_cells / self.total_cells


def render_observations_in_reading_order(
    observations: list[VisionObservation],
    *,
    minimum_confidence: float = 0.25,
) -> str:
    """Render prose OCR boxes in page reading order without table inference.

    RapidOCR normally emits one box per printed text line on the narrow,
    single-column long documents in this corpus.  Treating those boxes as a
    numeric matrix is both slow and semantically wrong.  This renderer groups
    only nearby baseline boxes, removes overlap-tile duplicates geometrically,
    and preserves paragraph gaps.  It intentionally makes no heading or table
    guesses: the remote Markdown remains preferable whenever it is complete.
    """

    usable = [
        item
        for item in observations
        if item.confidence >= minimum_confidence and item.text.strip()
    ]
    if not usable:
        return ""
    median_height = statistics.median(item.height for item in usable)
    y_tolerance = max(3.0, median_height * 0.65)
    bands: list[list[VisionObservation]] = []
    centers: list[float] = []
    for item in sorted(usable, key=lambda candidate: (candidate.y, candidate.x)):
        if bands and abs(item.y - centers[-1]) <= y_tolerance:
            bands[-1].append(item)
            centers[-1] = statistics.median(candidate.y for candidate in bands[-1])
        else:
            bands.append([item])
            centers.append(item.y)

    lines: list[tuple[float, str]] = []
    for center, band in zip(centers, bands, strict=True):
        fragments: list[VisionObservation] = []
        for item in sorted(band, key=lambda candidate: candidate.x):
            text = item.text.strip()
            duplicate_index = next(
                (
                    index
                    for index, previous in enumerate(fragments)
                    if (
                        text == previous.text.strip()
                        and abs(item.x - previous.x)
                        <= max(item.width, previous.width) * 0.35
                    )
                    or _horizontal_box_overlap(item, previous) >= 0.80
                ),
                None,
            )
            if duplicate_index is not None:
                previous = fragments[duplicate_index]
                # Adjacent OCR tiles can read their shared long line with
                # punctuation differences.  The boxes overlap almost fully,
                # so select one observation rather than concatenating both.
                if item.confidence > previous.confidence:
                    fragments[duplicate_index] = item
                continue
            fragments.append(item)
        line = "".join(item.text.strip() for item in fragments)
        if not line:
            continue
        if (
            lines
            and line == lines[-1][1]
            and abs(center - lines[-1][0]) <= y_tolerance
        ):
            continue
        lines.append((center, line))
    if not lines:
        return ""

    gaps = [later - earlier for (earlier, _), (later, _) in zip(lines, lines[1:])]
    typical_gap = statistics.median(gap for gap in gaps if gap > 0) if gaps else median_height
    paragraph_gap = max(median_height * 1.8, typical_gap * 1.45)
    paragraphs: list[str] = [lines[0][1]]
    for (previous_y, previous_line), (current_y, line) in zip(lines, lines[1:]):
        gap = current_y - previous_y
        # Ground truth represents a wrapped prose paragraph as one Markdown
        # block.  Joining its physical scan lines retains the same normalized
        # text while keeping reading-order blocks comparable.  A short,
        # isolated preceding line is a likely printed section/list heading and
        # therefore stays separate even when its following gap is modest.
        starts_paragraph = (
            gap > paragraph_gap
            or _is_reading_order_marker(previous_line)
            or _is_reading_order_marker(line)
            or (len(previous_line) <= 80 and gap > typical_gap * 1.20)
        )
        if starts_paragraph:
            paragraphs.append(line)
        else:
            paragraphs[-1] += line
    return "\n\n".join(paragraphs)


def _horizontal_box_overlap(
    first: VisionObservation,
    second: VisionObservation,
) -> float:
    first_left = first.x - first.width / 2
    first_right = first.x + first.width / 2
    second_left = second.x - second.width / 2
    second_right = second.x + second.width / 2
    overlap = max(0.0, min(first_right, second_right) - max(first_left, second_left))
    return overlap / max(1.0, min(first.width, second.width))


def _is_reading_order_marker(text: str) -> bool:
    """Recognize printed list/section starts without assigning Markdown level."""

    compact = re.sub(r"\s+", "", text)
    return bool(
        re.match(r"^(?:[（(]?\d{1,3}[）).、]|[a-zA-Z][).、]|第[一二三四五六七八九十百]+)", compact)
    )


@dataclass(frozen=True)
class _WideYearSchema:
    """Evidence-backed metadata layout for a wide annual-value matrix.

    Local OCR only emits numeric columns into the coordinate lattice.  The
    corresponding table schema must therefore state which metadata columns are
    numeric, rather than assuming every cash-value table has the same four
    leading columns.
    """

    headers: tuple[str, ...]
    numeric_metadata_columns: tuple[int, ...]
    term_column: int
    age_column: int
    gender_column: int
    payment_column: int
    text_metadata_columns: tuple[int, ...] = ()
    uses_two_row_annual_header: bool = False
    headerless_body: bool = False
    minimum_age_consistency: float = 0.75


@dataclass(frozen=True)
class _HeaderlessWideYearCandidate:
    """Strong non-header evidence for the common four-metadata layout.

    This is deliberately narrower than the normal semantic-header route.  It
    permits recovery only when the body itself exposes a repeated term, an
    explicit gender column, two leading numeric metadata columns, and a dense
    annual-value lattice.  That makes it suitable for crops where OCR missed a
    genuine header, without turning arbitrary numeric text into a table.
    """

    schema: _WideYearSchema
    data_top: float


_NUMERIC_CELL = re.compile(r"[+-]?[\d,.]+")


def run_vision_numeric_matrix_ocr(
    *,
    slices: list[ImageSlice],
    cache_dir: Path,
    workers: int = 4,
) -> VisionMatrixResult | None:
    """OCR grid slices locally and reconstruct a coordinate-aligned matrix."""

    if not slices or not vision_ocr_available():
        return None
    # ``cache_dir`` is record-specific. Compile once in the shared strategy
    # namespace instead of paying the Swift build cost for every document.
    binary = _compile_vision_helper(cache_dir.parent)
    tile_cache_dir = cache_dir / "tiles"
    tile_cache_dir.mkdir(parents=True, exist_ok=True)

    def read_slice(image_slice: ImageSlice) -> list[VisionObservation]:
        stem = Path(image_slice.file_name).stem
        image_path = tile_cache_dir / f"{stem}.jpg"
        tsv_path = tile_cache_dir / f"{stem}.tsv"
        if not tsv_path.exists():
            if not image_path.exists():
                image_path.write_bytes(image_slice.image_bytes)
            completed = subprocess.run(
                [str(binary), str(image_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if completed.returncode != 0:
                raise VisionOCRError(
                    completed.stderr.strip()
                    or f"Vision helper exited with {completed.returncode}"
                )
            tsv_path.write_text(completed.stdout, encoding="utf-8")
        return _parse_vision_tsv(tsv_path.read_text(encoding="utf-8"), image_slice)

    observations: list[VisionObservation] = []
    worker_count = min(max(1, workers), len(slices))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(read_slice, image_slice) for image_slice in slices]
        for future in as_completed(futures):
            observations.extend(future.result())
    return reconstruct_numeric_matrix_from_observations(observations)


def reconstruct_numeric_matrix_from_observations(
    observations: list[VisionObservation],
) -> VisionMatrixResult | None:
    """Build one or more HTML tables from regular year/age value matrices.

    This route intentionally requires strong semantic and sequential evidence.
    It does not attempt to guess arbitrary document layouts.
    """

    if len(observations) < 20:
        return None
    wide_result = _reconstruct_wide_year_matrix(observations)
    if wide_result is not None:
        return wide_result
    headerless_wide_result = _reconstruct_headerless_wide_year_matrix(observations)
    if headerless_wide_result is not None:
        return headerless_wide_result
    headerless_triangle_result = _reconstruct_headerless_triangular_numeric_grid(
        observations
    )
    if headerless_triangle_result is not None:
        return headerless_triangle_result
    headerless_fixed_result = _reconstruct_headerless_fixed_value_grid(observations)
    if headerless_fixed_result is not None:
        return headerless_fixed_result
    headerless_records_result = _reconstruct_headerless_sequential_records(
        observations
    )
    if headerless_records_result is not None:
        return headerless_records_result
    header_candidates = _matrix_header_candidates(observations)
    if not header_candidates:
        return None
    if len(header_candidates) == 1:
        result = _reconstruct_single_numeric_matrix(
            observations,
            semantic_header=header_candidates[0],
        )
        if result is None:
            return None
        semantic_header = header_candidates[0]
        context = _context_markdown(
            observations,
            lower=-math.inf,
            upper=semantic_header.y - max(4.0, semantic_header.height * 0.40),
        )
        if context:
            return replace(result, markdown=f"{context}\n\n{result.markdown}")
        return result

    results: list[VisionMatrixResult | None] = []
    for index, semantic_header in enumerate(header_candidates):
        next_header = (
            header_candidates[index + 1]
            if index + 1 < len(header_candidates)
            else None
        )
        lower = semantic_header.y - max(8.0, semantic_header.height * 0.60)
        upper = (
            next_header.y - max(8.0, next_header.height * 0.60)
            if next_header is not None
            else math.inf
        )
        segment = [item for item in observations if lower <= item.y < upper]
        results.append(
            _reconstruct_single_numeric_matrix(
                segment,
                semantic_header=semantic_header,
            )
        )

    row_start_evidence = [
        result.row_starts[0]
        for result in results
        if result is not None and result.row_starts
    ]
    if row_start_evidence:
        shared_row_start, shared_row_start_inliers = Counter(
            row_start_evidence
        ).most_common(1)[0]
    else:
        shared_row_start, shared_row_start_inliers = 0, 0

    # A repeated family of matrices often shares the same year lattice.  If a
    # single crop loses most first-column labels, reuse only a page-level
    # consensus supported by at least two independently reconstructed tables.
    if shared_row_start_inliers >= 2:
        for index, result in enumerate(results):
            if result is not None:
                continue
            semantic_header = header_candidates[index]
            next_header = (
                header_candidates[index + 1]
                if index + 1 < len(header_candidates)
                else None
            )
            lower = semantic_header.y - max(8.0, semantic_header.height * 0.60)
            upper = (
                next_header.y - max(8.0, next_header.height * 0.60)
                if next_header is not None
                else math.inf
            )
            segment = [item for item in observations if lower <= item.y < upper]
            results[index] = _reconstruct_single_numeric_matrix(
                segment,
                semantic_header=semantic_header,
                row_start_hint=shared_row_start,
            )

    # A page containing exactly two shallow matrices cannot produce the
    # two-table consensus above when one table loses its row-axis glyphs.  Use
    # one sibling only when that sibling is itself exceptionally strong, then
    # require the repaired table to independently match its complete topology
    # and dense header axis.  This is useful for paired male/female cash-value
    # tables while remaining too strict to coerce unrelated adjacent tables.
    successful = [
        (index, result)
        for index, result in enumerate(results)
        if result is not None
    ]
    missing = [index for index, result in enumerate(results) if result is None]
    if len(header_candidates) == 2 and len(successful) == 1 and len(missing) == 1:
        _, sibling = successful[0]
        assert sibling is not None
        sibling_is_strong = (
            sibling.coverage >= 0.90
            and sibling.header_starts
            and sibling.row_starts
            and sibling.header_sequence_inliers
            >= max(5, math.ceil((sibling.cols - 1) * 0.70))
            and sibling.row_sequence_inliers
            >= max(3, math.ceil(sibling.rows * 0.50))
        )
        if sibling_is_strong:
            index = missing[0]
            semantic_header = header_candidates[index]
            next_header = (
                header_candidates[index + 1]
                if index + 1 < len(header_candidates)
                else None
            )
            lower = semantic_header.y - max(8.0, semantic_header.height * 0.60)
            upper = (
                next_header.y - max(8.0, next_header.height * 0.60)
                if next_header is not None
                else math.inf
            )
            repaired = _reconstruct_single_numeric_matrix(
                [item for item in observations if lower <= item.y < upper],
                semantic_header=semantic_header,
                row_start_hint=sibling.row_starts[0],
                header_start_hint=sibling.header_starts[0],
                column_count_hint=sibling.cols,
                row_count_hint=sibling.rows,
            )
            if (
                repaired is not None
                and repaired.coverage >= 0.85
                and repaired.cols == sibling.cols
                and repaired.rows == sibling.rows
                and repaired.header_starts == sibling.header_starts
                and repaired.row_starts == sibling.row_starts
                and repaired.header_sequence_inliers
                >= max(5, math.ceil((repaired.cols - 1) * 0.70))
            ):
                results[index] = repaired

    # A partial multi-table reconstruction is too risky to replace a remote
    # fallback: every strongly detected matrix must pass its own lattice guard.
    if any(result is None for result in results):
        return None
    complete_results = [result for result in results if result is not None]
    sections: list[str] = []
    for index, result in enumerate(complete_results):
        semantic_header = header_candidates[index]
        if index == 0:
            context_lower = -math.inf
        else:
            previous_header = header_candidates[index - 1]
            gap = semantic_header.y - previous_header.y
            context_lower = max(previous_header.y, semantic_header.y - gap * 0.22)
        context_upper = semantic_header.y - max(4.0, semantic_header.height * 0.40)
        context = _context_markdown(
            observations,
            lower=context_lower,
            upper=context_upper,
        )
        if context:
            sections.append(context)
        sections.append(result.markdown)

    return VisionMatrixResult(
        markdown="\n\n".join(sections),
        rows=sum(result.rows for result in complete_results),
        cols=max(result.cols for result in complete_results),
        populated_cells=sum(result.populated_cells for result in complete_results),
        total_cells=sum(result.total_cells for result in complete_results),
        header_sequence_inliers=sum(
            result.header_sequence_inliers for result in complete_results
        ),
        row_sequence_inliers=sum(
            result.row_sequence_inliers for result in complete_results
        ),
        table_count=len(complete_results),
        row_starts=tuple(
            result.row_starts[0] for result in complete_results
        ),
        header_starts=tuple(
            result.header_starts[0] for result in complete_results
        ),
        top_y=complete_results[0].top_y,
        bottom_y=complete_results[-1].bottom_y,
    )


def _reconstruct_headerless_triangular_numeric_grid(
    observations: list[VisionObservation],
) -> VisionMatrixResult | None:
    """Recover a pure-numeric, data-only triangular age/value matrix."""

    numeric = _merge_split_numeric_fragments(
        [item for item in observations if _numeric_value(item.text) is not None]
    )
    if len(numeric) < 300:
        return None
    median_height = statistics.median(item.height for item in numeric)
    x_groups = _drop_sparse_close_x_groups(
        [
            group
            for group in _cluster_observations(
                numeric,
                axis="x",
                tolerance=max(8.0, median_height * 0.55),
            )
            if len(group) >= 3
        ]
    )
    if not 14 <= len(x_groups) <= 128:
        return None
    observed_x_centers = [_axis_mean(group, "x") for group in x_groups]
    gaps = [
        right - left
        for left, right in zip(observed_x_centers, observed_x_centers[1:])
    ]
    plausible_gaps = _central_gap_sample(gaps)
    if not plausible_gaps:
        return None
    column_step = statistics.median(plausible_gaps)
    if column_step <= 0:
        return None
    x_centers = [observed_x_centers[0]]
    inserted_columns = 0
    for left, right in zip(observed_x_centers, observed_x_centers[1:]):
        gap = right - left
        units = round(gap / column_step)
        if (
            2 <= units <= 3
            and abs(gap - units * column_step) <= column_step * 0.25
        ):
            inserted_columns += units - 1
            if inserted_columns > 4:
                return None
            x_centers.extend(left + column_step * index for index in range(1, units))
        elif gap > column_step * 1.60:
            return None
        x_centers.append(right)

    y_groups = [
        group
        for group in _cluster_observations(
            numeric,
            axis="y",
            tolerance=max(5.0, median_height * 0.45),
        )
        if len(group) >= 2
    ]
    if not 30 <= len(y_groups) <= 200:
        return None
    y_centers = [_axis_mean(group, "y") for group in y_groups]
    row_gaps = [
        right - left for left, right in zip(y_centers, y_centers[1:])
    ]
    plausible_row_gaps = _central_gap_sample(row_gaps)
    if not plausible_row_gaps:
        return None
    row_step = statistics.median(plausible_row_gaps)

    grid = [["" for _ in x_centers] for _ in y_centers]
    confidence = [[0.0 for _ in x_centers] for _ in y_centers]
    for item in numeric:
        column = _nearest_index(x_centers, item.x)
        row = _nearest_index(y_centers, item.y)
        if abs(x_centers[column] - item.x) > _axis_assignment_limit(
            x_centers,
            column,
        ):
            continue
        if abs(y_centers[row] - item.y) > row_step * 0.45:
            continue
        text = _normalize_numeric_text(item.text)
        if _numeric_value(text) is None:
            continue
        if item.confidence >= confidence[row][column]:
            grid[row][column] = text
            confidence[row][column] = item.confidence

    observed_ages = [
        (
            value
            if (value := _integer_value(row[0])) is not None and 0 <= value <= 200
            else None
        )
        for row in grid
    ]
    age_evidence = sum(value is not None for value in observed_ages)
    if age_evidence < math.ceil(len(grid) * 0.70):
        return None
    inferred = _infer_monotonic_age_sequence(observed_ages)
    if inferred is None:
        return None
    ages, age_inliers = inferred
    if age_inliers < math.ceil(age_evidence * 0.70):
        return None
    for row, age in zip(grid, ages, strict=True):
        row[0] = str(age)

    right_edges = [
        max((index for index, value in enumerate(row[1:], start=1) if value), default=0)
        for row in grid
    ]
    if any(edge == 0 for edge in right_edges):
        return None
    nonincreasing_pairs = sum(
        right <= left + 1
        for left, right in zip(right_edges, right_edges[1:])
    )
    if nonincreasing_pairs < math.ceil((len(right_edges) - 1) * 0.75):
        return None
    leading_edge = statistics.median(right_edges[: min(5, len(right_edges))])
    trailing_edge = statistics.median(right_edges[-min(5, len(right_edges)):])
    if leading_edge - trailing_edge < max(8, len(x_centers) // 4):
        return None

    populated_cells = sum(bool(cell) for row in grid for cell in row)
    total_cells = len(grid) * len(x_centers)
    coverage = populated_cells / total_cells
    if not 0.25 <= coverage <= 0.80:
        return None
    table_rows = tuple(tuple(row) for row in grid)
    markdown = table_to_html(
        MarkdownTable(
            header=table_rows[0],
            rows=table_rows[1:],
            header_is_explicit=False,
        )
    )
    return VisionMatrixResult(
        markdown=markdown,
        rows=len(grid),
        cols=len(x_centers),
        populated_cells=populated_cells,
        total_cells=total_cells,
        header_sequence_inliers=0,
        row_sequence_inliers=age_inliers,
        row_starts=(ages[0],),
        top_y=y_centers[0],
        bottom_y=y_centers[-1],
    )


def _reconstruct_headerless_fixed_value_grid(
    observations: list[VisionObservation],
) -> VisionMatrixResult | None:
    """Recover a data-only fixed-width actuarial table.

    A recurring training/B layout has no printed header row: every physical
    row is ``交费期 | 性别 | 年龄 | 数值...``.  Markdown OCR tends to invent a
    different width for each crop, while local OCR retains a very strong
    coordinate lattice.  This route is intentionally narrow: it requires
    repeated ``N年交`` terms, explicit binary gender labels, adjacent age
    progression, at least four value columns, and high cell coverage.
    """

    numeric = [
        item for item in observations if _numeric_value(item.text) is not None
    ]
    nonnumeric = [
        item for item in observations if _numeric_value(item.text) is None
    ]
    if len(numeric) < 100:
        return None
    lattice_observations = nonnumeric + _merge_split_numeric_fragments(numeric)
    if len(lattice_observations) < 100:
        return None
    median_height = statistics.median(item.height for item in lattice_observations)
    x_groups = _cluster_observations(
        lattice_observations,
        axis="x",
        tolerance=max(8.0, median_height * 0.45),
    )
    maximum_x_evidence = max(map(len, x_groups), default=0)
    minimum_x_evidence = max(8, math.ceil(maximum_x_evidence * 0.18))
    x_groups = [group for group in x_groups if len(group) >= minimum_x_evidence]
    x_centers = [_axis_mean(group, "x") for group in x_groups]
    if not 7 <= len(x_centers) <= 20:
        return None

    term_columns = Counter(
        _nearest_index(x_centers, item.x)
        for item in lattice_observations
        if re.fullmatch(r"\d{1,2}年交", re.sub(r"\s+", "", item.text))
    )
    gender_columns = Counter(
        _nearest_index(x_centers, item.x)
        for item in lattice_observations
        if item.text.strip() in {"男性", "女性"}
    )
    if not term_columns or not gender_columns:
        return None
    term_column, term_column_inliers = term_columns.most_common(1)[0]
    gender_column, gender_column_inliers = gender_columns.most_common(1)[0]
    age_column = gender_column + 1
    if (
        term_column != 0
        or gender_column != term_column + 1
        or age_column >= len(x_centers) - 3
    ):
        return None

    y_groups = _cluster_observations(
        lattice_observations,
        axis="y",
        tolerance=max(5.0, median_height * 0.45),
    )
    minimum_row_cells = max(5, math.ceil(len(x_centers) * 0.65))
    dense_y_groups = [group for group in y_groups if len(group) >= minimum_row_cells]
    if len(dense_y_groups) < 20:
        return None
    y_centers = [_axis_mean(group, "y") for group in dense_y_groups]
    row_count = len(y_centers)
    if (
        term_column_inliers < math.ceil(row_count * 0.70)
        or gender_column_inliers < math.ceil(row_count * 0.70)
    ):
        return None
    row_gaps = [
        right - left for left, right in zip(y_centers, y_centers[1:])
    ]
    plausible_row_gaps = _central_gap_sample(row_gaps)
    if not plausible_row_gaps:
        return None
    row_step = statistics.median(plausible_row_gaps)

    grid = [["" for _ in x_centers] for _ in y_centers]
    confidence = [[0.0 for _ in x_centers] for _ in y_centers]
    for item in lattice_observations:
        column = _nearest_index(x_centers, item.x)
        row = _nearest_index(y_centers, item.y)
        if abs(x_centers[column] - item.x) > _axis_assignment_limit(
            x_centers,
            column,
        ):
            continue
        if abs(y_centers[row] - item.y) > row_step * 0.45:
            continue
        text = item.text.strip()
        if column >= age_column:
            text = _normalize_numeric_text(text)
            if _numeric_value(text) is None:
                continue
        if item.confidence >= confidence[row][column]:
            grid[row][column] = text
            confidence[row][column] = item.confidence

    observed_terms = [
        (index, re.sub(r"\s+", "", row[term_column]))
        for index, row in enumerate(grid)
        if re.fullmatch(r"\d{1,2}年交", re.sub(r"\s+", "", row[term_column]))
    ]
    if len(observed_terms) < math.ceil(row_count * 0.70):
        return None
    for index, row in enumerate(grid):
        if re.fullmatch(
            r"\d{1,2}年交",
            re.sub(r"\s+", "", row[term_column]),
        ):
            row[term_column] = re.sub(r"\s+", "", row[term_column])
            continue
        _, term = min(
            observed_terms,
            key=lambda item: abs(item[0] - index),
        )
        row[term_column] = term

    observed_genders = [
        (index, row[gender_column])
        for index, row in enumerate(grid)
        if row[gender_column] in {"男性", "女性"}
    ]
    if len(observed_genders) < math.ceil(row_count * 0.70):
        return None
    for index, row in enumerate(grid):
        if row[gender_column] in {"男性", "女性"}:
            continue
        _, gender = min(
            observed_genders,
            key=lambda item: abs(item[0] - index),
        )
        row[gender_column] = gender

    group_starts = [0]
    for index in range(1, row_count):
        if (
            grid[index][term_column] != grid[index - 1][term_column]
            or grid[index][gender_column] != grid[index - 1][gender_column]
        ):
            group_starts.append(index)
    group_starts.append(row_count)
    age_inliers = 0
    age_evidence = 0
    age_starts: list[int] = []
    for start, end in zip(group_starts, group_starts[1:]):
        if end - start < 3:
            return None
        observed_ages = [
            value
            for index in range(start, end)
            if (value := _integer_value(grid[index][age_column])) is not None
            and 0 <= value <= 120
        ]
        if len(observed_ages) < math.ceil((end - start) * 0.60):
            return None
        age_observations = [
            (
                value
                if (value := _integer_value(grid[index][age_column])) is not None
                and 0 <= value <= 120
                else None
            )
            for index in range(start, end)
        ]
        inferred = _infer_monotonic_age_sequence(age_observations)
        if inferred is None:
            return None
        inferred_ages, inliers = inferred
        # A short trailing group can contain two or three OCR-confused age
        # glyphs while every other metadata/cell coordinate remains exact.
        # Seventy percent still requires a dominant monotonic progression and
        # rejects constant/random numeric columns.
        if inliers < math.ceil(len(observed_ages) * 0.70):
            return None
        age_starts.append(inferred_ages[0])
        age_evidence += len(observed_ages)
        age_inliers += inliers
        for index, age in enumerate(inferred_ages, start=start):
            grid[index][age_column] = str(age)

    populated_cells = sum(bool(cell) for row in grid for cell in row)
    total_cells = len(grid) * len(x_centers)
    if populated_cells < math.ceil(total_cells * 0.80):
        return None
    table_rows = tuple(tuple(row) for row in grid)
    markdown = table_to_html(
        MarkdownTable(
            header=table_rows[0],
            rows=table_rows[1:],
            header_is_explicit=False,
        )
    )
    return VisionMatrixResult(
        markdown=markdown,
        rows=len(grid),
        cols=len(x_centers),
        populated_cells=populated_cells,
        total_cells=total_cells,
        header_sequence_inliers=0,
        row_sequence_inliers=age_inliers,
        row_starts=tuple(age_starts),
        top_y=y_centers[0],
        bottom_y=y_centers[-1],
    )


def _reconstruct_headerless_sequential_records(
    observations: list[VisionObservation],
) -> VisionMatrixResult | None:
    """Recover a tall headerless record table with a resetting sequence axis."""

    if len(observations) < 150:
        return None
    numeric = _merge_split_numeric_fragments(
        [item for item in observations if _numeric_value(item.text) is not None]
    )
    nonnumeric = [
        item for item in observations if _numeric_value(item.text) is None
    ]
    lattice = numeric + nonnumeric
    median_height = statistics.median(item.height for item in lattice)
    raw_x_groups = _cluster_observations(
        lattice,
        axis="x",
        tolerance=max(8.0, median_height * 0.45),
    )
    maximum_x_evidence = max(map(len, raw_x_groups), default=0)
    minimum_x_evidence = max(12, math.ceil(maximum_x_evidence * 0.25))
    x_groups = [
        group for group in raw_x_groups if len(group) >= minimum_x_evidence
    ]
    if not 5 <= len(x_groups) <= 12:
        return None
    x_centers = [_axis_mean(group, "x") for group in x_groups]
    lattice = [item for group in x_groups for item in group]

    y_groups = _cluster_observations(
        lattice,
        axis="y",
        tolerance=max(5.0, median_height * 0.45),
    )
    minimum_row_cells = max(4, math.ceil(len(x_centers) * 0.60))
    y_groups = [group for group in y_groups if len(group) >= minimum_row_cells]
    if not 30 <= len(y_groups) <= 500:
        return None
    y_centers = [_axis_mean(group, "y") for group in y_groups]
    row_gaps = [
        right - left for left, right in zip(y_centers, y_centers[1:])
    ]
    plausible_row_gaps = _central_gap_sample(row_gaps)
    if not plausible_row_gaps:
        return None
    row_step = statistics.median(plausible_row_gaps)
    if row_step <= 0:
        return None

    grid = [["" for _ in x_centers] for _ in y_centers]
    confidence = [[0.0 for _ in x_centers] for _ in y_centers]
    for item in lattice:
        column = _nearest_index(x_centers, item.x)
        row = _nearest_index(y_centers, item.y)
        if abs(x_centers[column] - item.x) > _axis_assignment_limit(
            x_centers,
            column,
        ):
            continue
        if abs(y_centers[row] - item.y) > row_step * 0.45:
            continue
        text = item.text.strip()
        if _numeric_value(text) is not None:
            text = _normalize_numeric_text(text)
        if text and item.confidence >= confidence[row][column]:
            grid[row][column] = text
            confidence[row][column] = item.confidence

    sequence_candidates: list[tuple[int, int, int, int]] = []
    for column in range(1, len(x_centers) - 1):
        values = [
            (
                value
                if (value := _integer_value(row[column])) is not None
                and 0 <= value <= 10_000
                else None
            )
            for row in grid
        ]
        evidence = sum(value is not None for value in values)
        if evidence < math.ceil(len(grid) * 0.80):
            continue
        increments = 0
        resets = 0
        invalid = 0
        for left, right in zip(values, values[1:]):
            if left is None or right is None:
                continue
            if right == left + 1:
                increments += 1
            elif left >= 5 and 0 <= right <= 10:
                resets += 1
            else:
                invalid += 1
        compared = increments + resets + invalid
        if (
            increments < 30
            or compared == 0
            or invalid > max(2, math.floor(compared * 0.05))
            or resets > 12
        ):
            continue
        sequence_candidates.append((increments, evidence, -invalid, column))
    if not sequence_candidates:
        return None
    _, sequence_evidence, _, sequence_column = max(sequence_candidates)

    # A horizontal tile boundary can OCR one physical row twice: metadata and
    # sequence on one side, then the trailing value on the other.  Remove only
    # a non-sequence row sandwiched between two proven consecutive sequence
    # values, merging its otherwise missing cells into the preceding row.
    cleaned_grid: list[list[str]] = []
    for index, row in enumerate(grid):
        sequence = _integer_value(row[sequence_column])
        if sequence is not None:
            cleaned_grid.append(row)
            continue
        previous_sequence = (
            _integer_value(cleaned_grid[-1][sequence_column])
            if cleaned_grid
            else None
        )
        next_sequence = (
            _integer_value(grid[index + 1][sequence_column])
            if index + 1 < len(grid)
            else None
        )
        if (
            cleaned_grid
            and previous_sequence is not None
            and next_sequence == previous_sequence + 1
        ):
            for column, value in enumerate(row):
                if value and not cleaned_grid[-1][column]:
                    cleaned_grid[-1][column] = value
            continue
        cleaned_grid.append(row)
    grid = cleaned_grid

    # Repair an isolated metadata OCR error only when both adjacent physical
    # rows independently agree. This preserves real group transitions.
    for index in range(1, len(grid) - 1):
        for column in range(sequence_column):
            left = grid[index - 1][column]
            right = grid[index + 1][column]
            if left and left == right and grid[index][column] != left:
                grid[index][column] = left

    repeated_text_columns = 0
    for column in range(sequence_column):
        values = [row[column] for row in grid if row[column]]
        if not values:
            continue
        mode, mode_count = Counter(values).most_common(1)[0]
        if _numeric_value(mode) is not None:
            continue
        if mode_count >= 10 and mode_count >= math.ceil(len(values) * 0.70):
            repeated_text_columns += 1
            for row in grid:
                row[column] = mode
    if repeated_text_columns < 2:
        return None
    _normalize_repeated_payment_shorthand(
        grid,
        columns=range(sequence_column),
    )

    value_columns = range(sequence_column + 1, len(x_centers))
    numeric_value_cells = sum(
        _numeric_value(row[column]) is not None
        for row in grid
        for column in value_columns
    )
    if numeric_value_cells < math.ceil(
        len(grid) * len(tuple(value_columns)) * 0.70
    ):
        return None

    populated_cells = sum(bool(cell) for row in grid for cell in row)
    total_cells = len(grid) * len(x_centers)
    if populated_cells < math.ceil(total_cells * 0.82):
        return None
    table_rows = tuple(tuple(row) for row in grid)
    markdown = table_to_html(
        MarkdownTable(
            header=table_rows[0],
            rows=table_rows[1:],
            header_is_explicit=False,
        )
    )
    first_sequence = _integer_value(grid[0][sequence_column])
    return VisionMatrixResult(
        markdown=markdown,
        rows=len(grid),
        cols=len(x_centers),
        populated_cells=populated_cells,
        total_cells=total_cells,
        header_sequence_inliers=0,
        row_sequence_inliers=sequence_evidence,
        row_starts=((first_sequence,) if first_sequence is not None else ()),
        top_y=y_centers[0],
        bottom_y=y_centers[-1],
    )


def _infer_monotonic_age_sequence(
    observations: list[int | None],
) -> tuple[list[int], int] | None:
    """Fit a bounded age path while allowing one or two omitted physical rows."""

    if not observations:
        return None
    # score = mismatching observed glyphs, total absolute error, skipped-age
    # gaps.  Exact OCR evidence dominates; the later terms only resolve ties.
    states: dict[int, tuple[tuple[int, int, int], list[int]]] = {}
    first = observations[0]
    for age in range(121):
        mismatch = int(first is not None and first != age)
        error = abs(first - age) if first is not None else 0
        states[age] = ((mismatch, error, 0), [age])
    for observed in observations[1:]:
        next_states: dict[int, tuple[tuple[int, int, int], list[int]]] = {}
        for age in range(121):
            candidates = [
                (previous_age, states[previous_age])
                for previous_age in range(max(0, age - 3), age)
                if previous_age in states
            ]
            if not candidates:
                continue
            previous_age, (previous_score, previous_path) = min(
                candidates,
                key=lambda item: item[1][0],
            )
            score = (
                previous_score[0] + int(observed is not None and observed != age),
                previous_score[1] + (
                    abs(observed - age) if observed is not None else 0
                ),
                previous_score[2] + (age - previous_age - 1),
            )
            next_states[age] = (score, [*previous_path, age])
        states = next_states
        if not states:
            return None
    _, (_, path) = min(states.items(), key=lambda item: item[1][0])
    inliers = sum(
        observed is not None and observed == age
        for observed, age in zip(observations, path, strict=True)
    )
    return path, inliers


def _reconstruct_wide_year_matrix(
    observations: list[VisionObservation],
    *,
    headerless_candidate: _HeaderlessWideYearCandidate | None = None,
) -> VisionMatrixResult | None:
    """Reconstruct a very wide benefit matrix with repeated age groups.

    These pages have four metadata columns followed by a sequential series of
    ``第N保单年度`` columns.  The local OCR commonly suppresses repeated
    metadata, but its numeric coordinates are highly accurate.  This route
    therefore infers only evidence-backed lattices: sequential year headers,
    triangular right-edge resets, and observed age progression.
    """

    semantic_header = _wide_year_header_candidate(observations)
    if semantic_header is None and headerless_candidate is None:
        return None
    if semantic_header is not None:
        schema = _wide_year_schema(semantic_header.text)
        if schema is None:
            return None
        header_y = semantic_header.y
        # Only the three-metadata schema has a second, numeric annual-header
        # row beneath the semantic title.  Applying that wide exclusion band
        # to an ordinary one-row header silently drops its first data row on
        # tightly packed actuarial tables.
        header_band = (
            max(40.0, semantic_header.height * 2.0)
            if schema.uses_two_row_annual_header
            else max(20.0, semantic_header.height * 1.15)
        )
        observed_years = {
            int(value)
            for item in observations
            if abs(item.y - header_y) <= header_band
            for value in re.findall(r"第\s*(\d+)\s*保单年度", item.text)
        }
        # Some forms place the annual sequence in a separate numeric header row
        # (``1, 2, ...``) below a shared ``保单年度末`` label.  Accept it only
        # when the same header band supplies a substantial sequence beginning at
        # one; ordinary body numbers lie outside this narrow band.
        if not observed_years:
            observed_years = {
                value
                for item in observations
                if abs(item.y - header_y) <= header_band
                if (value := _integer_value(item.text)) is not None and 1 <= value <= 254
            }
        compact_header = re.sub(r"\s+", "", semantic_header.text)
        has_first_year_label = 1 in observed_years or bool(
            re.search(r"(?:第\s*)?1\s*保单年度", compact_header)
        )
        # OCR can collapse a long `1 2 3 ...` annual-label row into one text
        # box.  A semantic header that still explicitly names annual policy
        # years and its first year may rely on the independently dense body
        # lattice for the remaining extent.  Do not fabricate a tail here.
        if not has_first_year_label or (
            len(observed_years) < 5 and "保单年度" not in compact_header
        ):
            return None
    else:
        assert headerless_candidate is not None
        schema = headerless_candidate.schema
        # The body-only route has no inferred textual context.  Place the
        # synthetic boundary immediately before its first numeric row so all
        # candidate body observations remain eligible for reconstruction.
        header_y = headerless_candidate.data_top - 1.0
        header_band = 0.0
        observed_years: set[int] = set()

    numeric_data = [
        item
        for item in observations
        # A merged annual heading often has its ``1..N`` labels on a second
        # line.  Exclude the complete header band so those labels never become
        # a phantom first data row.
        if item.y > header_y + header_band
        and _numeric_value(item.text) is not None
    ]
    if len(numeric_data) < 100:
        return None
    numeric_data = _merge_split_numeric_fragments(numeric_data)
    median_height = statistics.median(item.height for item in numeric_data)
    x_groups = _cluster_observations(
        numeric_data,
        axis="x",
        tolerance=max(8.0, median_height * 0.55),
    )
    # A triangular tail can contain only a few values.  Three independent row
    # observations are enough to retain it; singleton OCR jitter is discarded.
    x_groups = [group for group in x_groups if len(group) >= 3]
    x_groups = _drop_sparse_close_x_groups(x_groups)
    x_centers = [_axis_mean(group, "x") for group in x_groups]
    matrix_numeric_data = [item for group in x_groups for item in group]
    if semantic_header is not None and _has_text_metadata_layout(
        schema=schema,
        observations=observations,
        header_y=header_y,
        header_height=semantic_header.height,
        first_numeric_x=x_centers[0],
    ):
        schema = _text_metadata_variant(
            schema,
            uses_two_row_annual_header=(
                "保单年度末" in re.sub(r"\s+", "", semantic_header.text)
                and "第1保单年度" not in re.sub(r"\s+", "", semantic_header.text)
            ),
        )
    numeric_metadata_count = len(schema.numeric_metadata_columns)
    # The headerless route needs a broad numeric signature, but an explicit
    # annual header proves a shorter 5--11 year schedule just as well.  Keep
    # at least five observed annual columns; that matches the independent
    # semantic sequence guard above and avoids treating a few prose numbers as
    # a cash-value lattice.
    minimum_columns = numeric_metadata_count + 5 if (
        semantic_header is not None or schema.headerless_body
    ) else 12
    if not minimum_columns <= len(x_centers) <= 254:
        return None
    if semantic_header is not None and schema.text_metadata_columns:
        schema = _adapt_text_metadata_schema(
            schema=schema,
            observations=observations,
            header_y=header_y,
            header_height=semantic_header.height,
            age_x=x_centers[0],
        )
    year_count = len(x_centers) - numeric_metadata_count
    if semantic_header is not None and observed_years and year_count < max(observed_years):
        # A triangular table's last annual columns are supported by only a
        # handful of rows.  OCR can miss that short right tail even though the
        # semantic header explicitly proves its extent.  Keep at most four
        # *trailing*, value-free columns; never fill cells and never bridge an
        # internal coordinate gap.
        missing_tail_columns = max(observed_years) - year_count
        annual_centers = x_centers[numeric_metadata_count:]
        annual_gaps = _central_gap_sample(
            [
                right - left
                for left, right in zip(annual_centers, annual_centers[1:])
            ]
        )
        maximum_missing_tail = 8 if schema.uses_two_row_annual_header else 4
        if (
            not 1 <= missing_tail_columns <= maximum_missing_tail
            or not annual_gaps
        ):
            return None
        annual_step = statistics.median(annual_gaps)
        if annual_step <= 0:
            return None
        x_centers.extend(
            x_centers[-1] + annual_step * index
            for index in range(1, missing_tail_columns + 1)
        )
        year_count += missing_tail_columns
    if semantic_header is None and not schema.headerless_body and year_count < 12:
        return None

    y_groups = _cluster_observations(
        matrix_numeric_data,
        axis="y",
        tolerance=max(5.0, median_height * 0.45),
    )
    dense_y_groups = [group for group in y_groups if len(group) >= 4]
    if len(dense_y_groups) < 20:
        return None
    dense_y = [_axis_mean(group, "y") for group in dense_y_groups]
    gaps = [right - left for left, right in zip(dense_y, dense_y[1:])]
    plausible_gaps = _central_gap_sample(gaps)
    if not plausible_gaps:
        return None
    row_step = statistics.median(plausible_gaps)
    y_candidates = [(_axis_mean(group, "y"), len(group)) for group in y_groups]
    y_centers = _contiguous_row_centers(
        candidates=y_candidates,
        first=min(dense_y),
        step=row_step,
    )
    if len(y_centers) < 20:
        return None

    grid = [
        ["" for _ in range(year_count + len(schema.headers))]
        for _ in y_centers
    ]
    confidence = [[0.0 for _ in row] for row in grid]
    for item in matrix_numeric_data:
        x_index = _nearest_index(x_centers, item.x)
        y_index = _nearest_index(y_centers, item.y)
        if abs(x_centers[x_index] - item.x) > _axis_assignment_limit(
            x_centers,
            x_index,
        ):
            continue
        if abs(y_centers[y_index] - item.y) > row_step * 0.45:
            continue
        if x_index < numeric_metadata_count:
            column = schema.numeric_metadata_columns[x_index]
        else:
            column = len(schema.headers) + (x_index - numeric_metadata_count)
        text = _normalize_numeric_text(item.text)
        if item.confidence >= confidence[y_index][column]:
            grid[y_index][column] = text
            confidence[y_index][column] = item.confidence

    right_edges = [
        max(
            (
                column
                for column, value in enumerate(row[len(schema.headers):], start=0)
                if value
            ),
            default=-1,
        )
        for row in grid
    ]
    minimum_group_rows = max(4, min(20, year_count // 4))
    reset_jump = max(4, year_count // 5)
    group_starts = [0]
    for index in range(1, len(right_edges)):
        if (
            index - group_starts[-1] >= minimum_group_rows
            # A sparse OCR tail can look like the beginning of a new
            # triangular group.  It cannot, however, establish a valid age
            # progression by itself, so retain it in the preceding group.
            and len(right_edges) - index >= minimum_group_rows
            and right_edges[index] - right_edges[index - 1] >= reset_jump
        ):
            group_starts.append(index)
    if schema.headerless_body:
        group_starts = sorted(set(group_starts + _headerless_metadata_starts(
            observations, y_centers, row_step, header_y, x_centers[numeric_metadata_count]
        )))

    group_lengths: list[int] = []
    inferred_ages: list[int] = []
    age_evidence_count = 0
    age_inliers = 0
    for index, start in enumerate(group_starts):
        end = group_starts[index + 1] if index + 1 < len(group_starts) else len(grid)
        group_lengths.append(end - start)
        # Most forms have one row per age, but another common layout emits a
        # male/female pair for each age.  Infer the repetition from the age
        # column itself instead of assuming either visual convention.
        age_patterns: list[tuple[int, int, list[int], int, int]] = []
        for rows_per_age in (1, 2):
            # The document can begin on either member of a male/female pair,
            # so test both phase alignments for the two-row convention.
            for phase in range(rows_per_age):
                offsets = [
                    value - ((row_index - start + phase) // rows_per_age)
                    for row_index in range(start, end)
                    if (value := _integer_value(grid[row_index][schema.age_column]))
                    is not None
                    and 0 <= value <= 200
                ]
                if offsets:
                    age_start, inliers = Counter(offsets).most_common(1)[0]
                    age_patterns.append(
                        (rows_per_age, phase, offsets, age_start, inliers)
                    )
        if not age_patterns:
            return None
        (
            rows_per_age,
            age_phase,
            group_age_offsets,
            group_age_start,
            group_age_inliers,
        ) = max(age_patterns, key=lambda pattern: pattern[4])
        minimum_group_age_inliers = max(
            3,
            math.ceil(len(group_age_offsets) * schema.minimum_age_consistency),
        )
        if (
            not 0 <= group_age_start <= 200
            or group_age_inliers < minimum_group_age_inliers
        ):
            return None
        age_evidence_count += len(group_age_offsets)
        age_inliers += group_age_inliers
        inferred_ages.extend(
            group_age_start + ((row_index - start + age_phase) // rows_per_age)
            for row_index in range(start, end)
        )
    if (
        age_evidence_count < 10
        or age_inliers < math.ceil(age_evidence_count * schema.minimum_age_consistency)
    ):
        return None
    for index, age in enumerate(inferred_ages):
        grid[index][schema.age_column] = str(age)

    if schema.text_metadata_columns:
        if not _fill_text_metadata_columns(
            grid=grid,
            observations=observations,
            y_centers=y_centers,
            row_step=row_step,
            header_y=header_y,
            annual_x=x_centers[numeric_metadata_count],
            text_columns=schema.text_metadata_columns,
        ):
            return None
    else:
        term_observations = [
            item.text.strip()
            for item in observations
            if header_y < item.y <= y_centers[-1] + row_step
            and item.x < x_centers[0]
            and item.text.strip()
            and _numeric_value(item.text) is None
        ]
        if not term_observations:
            return None
        term, term_inliers = Counter(term_observations).most_common(1)[0]
        if term_inliers < 3:
            return None
        for row in grid:
            row[schema.term_column] = term

    gender_observations = [
        item
        for item in observations
        if item.text.strip() in {"男", "女"}
        and header_y < item.y <= y_centers[-1] + row_step
    ]
    gender_confidence = [0.0 for _ in grid]
    for item in gender_observations:
        row_index = _nearest_index(y_centers, item.y)
        if abs(y_centers[row_index] - item.y) > row_step * 0.45:
            continue
        if item.confidence >= gender_confidence[row_index]:
            grid[row_index][schema.gender_column] = item.text.strip()
            gender_confidence[row_index] = item.confidence
    paired_groups = len(group_lengths) // 2
    pairs_have_equal_lengths = paired_groups >= 2 and all(
        group_lengths[index] == group_lengths[index + 1]
        for index in range(0, paired_groups * 2, 2)
    )
    if gender_observations and pairs_have_equal_lengths:
        first_gender = min(gender_observations, key=lambda item: item.y).text.strip()
        for group_index, start in enumerate(group_starts):
            end = (
                group_starts[group_index + 1]
                if group_index + 1 < len(group_starts)
                else len(grid)
            )
            gender = (
                first_gender
                if group_index % 2 == 0
                else ("女" if first_gender == "男" else "男")
            )
            for row_index in range(start, end):
                if not grid[row_index][schema.gender_column]:
                    grid[row_index][schema.gender_column] = gender

    # Repeated payment-period cells are often suppressed by OCR.  Propagate an
    # observed small integer only within a pair of equally long gender groups.
    if pairs_have_equal_lengths:
        for pair_start in range(0, paired_groups * 2, 2):
            first = group_starts[pair_start]
            after_pair = (
                group_starts[pair_start + 2]
                if pair_start + 2 < len(group_starts)
                else len(grid)
            )
            values = [
                int(grid[index][schema.payment_column])
                for index in range(first, after_pair)
                if grid[index][schema.payment_column].isdigit()
                and 0 < int(grid[index][schema.payment_column]) <= 100
            ]
            if not values:
                continue
            payment, payment_inliers = Counter(values).most_common(1)[0]
            if payment_inliers < math.ceil(len(values) * 0.60):
                continue
            for row_index in range(first, after_pair):
                grid[row_index][schema.payment_column] = str(payment)

    _normalize_repeated_payment_shorthand(
        grid,
        columns=range(len(schema.headers)),
    )
    _normalize_dominant_metadata_columns(
        grid,
        columns=(
            column
            for column in range(len(schema.headers))
            if column != schema.age_column
        ),
    )
    _normalize_repeated_zero_decimal_columns(
        grid,
        columns=range(len(schema.headers), len(grid[0])),
    )

    headers = schema.headers + tuple(
        f"第{index}保单年度" for index in range(1, year_count + 1)
    )
    table_rows = tuple(tuple(row) for row in grid)
    markdown = (
        _two_row_annual_header_table_to_html(
            metadata_headers=schema.headers,
            year_count=year_count,
            rows=table_rows,
        )
        if schema.uses_two_row_annual_header
        else table_to_html(MarkdownTable(
            header=table_rows[0] if schema.headerless_body else headers,
            rows=table_rows[1:] if schema.headerless_body else table_rows,
            header_is_explicit=not schema.headerless_body,
        ))
    )
    context = (
        _context_markdown(
            observations,
            lower=-math.inf,
            upper=header_y - max(4.0, semantic_header.height * 0.40),
        )
        if semantic_header is not None
        else ""
    )
    if context:
        markdown = f"{context}\n\n{markdown}"
    populated_cells = sum(bool(cell) for row in grid for cell in row)
    total_cells = len(grid) * len(headers)
    if populated_cells < math.ceil(total_cells * 0.35):
        return None
    return VisionMatrixResult(
        markdown=markdown,
        rows=len(grid),
        cols=len(headers),
        populated_cells=populated_cells,
        total_cells=total_cells,
        header_sequence_inliers=len(observed_years),
        row_sequence_inliers=age_inliers,
        row_starts=tuple(group_starts),
        header_starts=(1,) if semantic_header is not None else (),
        top_y=header_y,
        bottom_y=y_centers[-1],
    )


def _normalize_repeated_payment_shorthand(
    grid: list[list[str]],
    *,
    columns: range,
) -> None:
    """Repair a repeated OCR ``交`` token when the row schema proves a term.

    ``交`` alone is not a payment-period value in these actuarial tables, while
    ``趸交`` is.  Apply the correction only when one metadata column is
    overwhelmingly the shorthand and a separate metadata column independently
    repeats a value containing ``年``.
    """

    modes: dict[int, tuple[str, int, int]] = {}
    for column in columns:
        values = [row[column].strip() for row in grid if row[column].strip()]
        if len(values) < 10:
            continue
        mode, count = Counter(values).most_common(1)[0]
        if count >= math.ceil(len(values) * 0.70):
            modes[column] = (mode, count, len(values))
    payment_columns = [
        column for column, (mode, _, _) in modes.items() if mode == "交"
    ]
    has_term_column = any(
        "年" in mode and column not in payment_columns
        for column, (mode, _, _) in modes.items()
    )
    if not has_term_column:
        return
    for column in payment_columns:
        for row in grid:
            row[column] = "趸交"


def _normalize_dominant_metadata_columns(
    grid: list[list[str]],
    *,
    columns: Iterable[int],
) -> None:
    """Fill sparse metadata gaps when nearly every row proves one value.

    Long actuarial matrices commonly repeat the term, payment method, and
    gender on every physical row.  Local OCR can miss a few of those cells or
    confuse one character (for example ``5年`` as ``5午``).  Restricting the
    correction to an 85% row-level mode avoids flattening genuine group
    transitions.
    """

    for column in columns:
        values = [row[column].strip() for row in grid if row[column].strip()]
        if len(values) < 10:
            continue
        mode, mode_count = Counter(values).most_common(1)[0]
        if (
            _numeric_value(mode) is not None
            or mode_count < math.ceil(len(grid) * 0.85)
        ):
            continue
        for row in grid:
            current = row[column].strip()
            if not current:
                row[column] = mode
                continue
            if (
                len(mode) >= 2
                and len(current) == len(mode)
                and sum(left != right for left, right in zip(current, mode, strict=True))
                == 1
            ):
                row[column] = mode


def _normalize_repeated_zero_decimal_columns(
    grid: list[list[str]],
    *,
    columns: range,
) -> None:
    """Restore an all-zero fixed-decimal value column from sparse OCR."""

    zero_spellings = {"0", "00", "000", "0.0", "0.00"}
    for column in columns:
        values = [row[column].strip() for row in grid if row[column].strip()]
        if (
            len(values) < 10
            or not set(values).issubset(zero_spellings)
            or values.count("0.00") < math.ceil(len(grid) * 0.60)
        ):
            continue
        for row in grid:
            row[column] = "0.00"


def _is_wide_year_header_text(text: str) -> bool:
    return _wide_year_schema(text) is not None


def _wide_year_schema(text: str) -> _WideYearSchema | None:
    """Recognize supported annual cash-value header layouts by semantics.

    Both layouts below occur across the training corpus.  They differ only in
    whether a separate insurance-period column is printed; selection depends
    exclusively on visible header words, never on a document identity.
    """

    compact = re.sub(r"\s+", "", text)
    four_metadata_markers = ("保险期间", "交费期间", "性别", "保单年度")
    if all(marker in compact for marker in four_metadata_markers):
        # Some source tables label this column merely as ``年龄`` and print
        # both insurance period and payment period as text (for example,
        # ``5年`` and ``趸交``).  Only the age then belongs in the numeric
        # lattice.  The explicit four-label header is what keeps this route
        # separate from arbitrary prose plus a numeric grid.
        if "投保年龄" not in compact:
            return _WideYearSchema(
                headers=("保险期间", "交费期间", "投保年龄", "性别"),
                numeric_metadata_columns=(2,),
                term_column=0,
                age_column=2,
                gender_column=3,
                payment_column=1,
                text_metadata_columns=(0, 1),
                uses_two_row_annual_header=(
                    "保单年度末" in compact and "第1保单年度" not in compact
                ),
            )
        return _WideYearSchema(
            headers=("保险期间", "交费期间", "投保年龄", "性别"),
            numeric_metadata_columns=(1, 2),
            term_column=0,
            age_column=2,
            gender_column=3,
            payment_column=1,
            uses_two_row_annual_header=(
                "保单年度末" in compact and "第1保单年度" not in compact
            ),
        )
    # The annual values can instead be headed by a merged ``保单年度末``
    # cell, with 1..N in a following row.  Here payment period and gender are
    # textual, leaving only age in the numeric coordinate lattice.
    if all(marker in compact for marker in ("保单年度末", "交费期间", "性别", "投保")):
        return _WideYearSchema(
            headers=("交费期间", "性别", "投保年龄（周岁）"),
            numeric_metadata_columns=(2,),
            term_column=0,
            age_column=2,
            gender_column=1,
            payment_column=0,
            uses_two_row_annual_header=True,
        )
    return None


def _fill_text_metadata_columns(
    *,
    grid: list[list[str]],
    observations: list[VisionObservation],
    y_centers: list[float],
    row_step: float,
    header_y: float,
    annual_x: float,
    text_columns: tuple[int, ...],
) -> bool:
    """Recover repeated textual metadata from their own physical columns.

    A wide cash-value table may carry two text fields before its numeric age
    column.  Reusing one global mode for both silently swaps or drops one of
    them, so require an independently repeated x-band for every field and
    populate each band only on its corresponding rows.
    """

    candidates = [
        item
        for item in observations
        if header_y < item.y <= y_centers[-1] + row_step
        and item.x < annual_x
        and item.text.strip()
        and item.text.strip() not in {"男", "女"}
        and _numeric_value(item.text) is None
    ]
    if not candidates:
        return False
    tolerance = max(12.0, statistics.median(item.height for item in candidates) * 0.60)
    supported_bands: list[tuple[float, list[VisionObservation], int, float]] = []
    for band in _cluster_observations(candidates, axis="x", tolerance=tolerance):
        values = [item.text.strip() for item in band]
        mode, mode_count = Counter(values).most_common(1)[0]
        consistency = mode_count / len(values)
        if mode_count >= 3 and consistency >= 0.25:
            supported_bands.append(
                (
                    _axis_mean(band, "x"),
                    band,
                    mode_count,
                    consistency,
                )
            )
    if len(supported_bands) < len(text_columns):
        return False
    # Select by repeated evidence, then restore the visual left-to-right order
    # for the logical metadata columns.  This rejects a one-off OCR artefact
    # without making the column choice depend on a document identity.
    selected = sorted(
        sorted(
            supported_bands,
            key=lambda value: (-value[2], -value[3], value[0]),
        )[: len(text_columns)],
        key=lambda value: value[0],
    )
    for column, (_, band, _, _) in zip(text_columns, selected):
        confidence = [0.0 for _ in grid]
        for item in band:
            row_index = _nearest_index(y_centers, item.y)
            if abs(y_centers[row_index] - item.y) > row_step * 0.45:
                continue
            if item.confidence >= confidence[row_index]:
                grid[row_index][column] = item.text.strip()
                confidence[row_index] = item.confidence
    return True


def _adapt_text_metadata_schema(
    *,
    schema: _WideYearSchema,
    observations: list[VisionObservation],
    header_y: float,
    header_height: float,
    age_x: float,
) -> _WideYearSchema:
    """Adopt a four-metadata table's visible left-to-right header order.

    Insurers do not use one canonical order for payment period, insurance
    period, gender, and age.  The numeric lattice only establishes the age
    coordinate, while the table header independently establishes the other
    three.  Adapt only when every label has a nearby, separate header box;
    fragmented headers retain the conservative schema selected above.
    """

    positions = _header_label_positions(
        observations=observations,
        header_y=header_y,
        header_height=header_height,
    )
    if any(value is None for value in positions.values()):
        return schema
    # A nearby explanatory sentence can contain all four words.  Requiring
    # the OCR age label to line up with the numeric metadata coordinate keeps
    # this adaptation tied to the actual table header.
    assert positions["投保年龄"] is not None
    if abs(positions["投保年龄"] - age_x) > max(100.0, header_height * 4.0):
        return schema
    ordered_names = tuple(
        name for name, _ in sorted(positions.items(), key=lambda item: item[1])
    )
    new_index = {name: index for index, name in enumerate(ordered_names)}
    return replace(
        schema,
        headers=ordered_names,
        numeric_metadata_columns=(new_index["投保年龄"],),
        term_column=new_index["保险期间"],
        age_column=new_index["投保年龄"],
        gender_column=new_index["性别"],
        payment_column=new_index["交费期间"],
        text_metadata_columns=tuple(
            sorted(
                (new_index["保险期间"], new_index["交费期间"]),
            )
        ),
    )


def _header_label_positions(
    *,
    observations: list[VisionObservation],
    header_y: float,
    header_height: float,
) -> dict[str, float | None]:
    band = max(48.0, header_height * 1.25)

    def position_for(marker: str) -> float | None:
        matches = [
            item
            for item in observations
            if marker in re.sub(r"\s+", "", item.text)
            and abs(item.y - header_y) <= band
        ]
        if not matches:
            return None
        return min(matches, key=lambda item: abs(item.y - header_y)).x

    return {
        "保险期间": position_for("保险期间"),
        "交费期间": position_for("交费期间"),
        "投保年龄": position_for("年龄"),
        "性别": position_for("性别"),
    }


def _has_text_metadata_layout(
    *,
    schema: _WideYearSchema,
    observations: list[VisionObservation],
    header_y: float,
    header_height: float,
    first_numeric_x: float,
) -> bool:
    """Detect when the standard header's payment field is textual in-body."""

    if schema.text_metadata_columns or len(schema.headers) != 4:
        return False
    positions = _header_label_positions(
        observations=observations,
        header_y=header_y,
        header_height=header_height,
    )
    if any(value is None for value in positions.values()):
        return False
    values = [value for value in positions.values() if value is not None]
    # A single merged header box can contain every label; it does not establish
    # a physical column order.  Require four separately located labels.
    if len({round(value, 1) for value in values}) != 4:
        return False
    age_x = positions["投保年龄"]
    assert age_x is not None
    return abs(age_x - first_numeric_x) <= max(100.0, header_height * 4.0)


def _text_metadata_variant(
    schema: _WideYearSchema,
    *,
    uses_two_row_annual_header: bool,
) -> _WideYearSchema:
    """Switch a four-field annual table from numeric to textual payment data."""

    return replace(
        schema,
        numeric_metadata_columns=(2,),
        term_column=0,
        age_column=2,
        gender_column=3,
        payment_column=1,
        text_metadata_columns=(0, 1),
        uses_two_row_annual_header=uses_two_row_annual_header,
    )


def _reconstruct_headerless_wide_year_matrix(
    observations: list[VisionObservation],
) -> VisionMatrixResult | None:
    """Recover a standard wide matrix only from independent body evidence.

    A detected table header always takes precedence.  This fallback is for
    very large documents whose local tiles preserve the body but miss the top
    header altogether.  The candidate guard below deliberately requires more
    than a numeric grid, so it does not broaden the generic matrix route.
    """

    candidate = _headerless_text_candidate(observations) or _headerless_wide_year_candidate(observations)
    if candidate is None:
        return None
    return _reconstruct_wide_year_matrix(
        observations,
        headerless_candidate=candidate,
    )


def _headerless_wide_year_candidate(
    observations: list[VisionObservation],
) -> _HeaderlessWideYearCandidate | None:
    """Validate the body signature of a four-metadata annual-value table."""

    numeric_observations = [
        item for item in observations if _numeric_value(item.text) is not None
    ]
    if len(numeric_observations) < 100:
        return None
    numeric_observations = _merge_split_numeric_fragments(numeric_observations)
    median_height = statistics.median(item.height for item in numeric_observations)
    x_groups = _cluster_observations(
        numeric_observations,
        axis="x",
        tolerance=max(8.0, median_height * 0.55),
    )
    x_groups = [group for group in x_groups if len(group) >= 3]
    x_groups = _drop_sparse_close_x_groups(x_groups)
    if not 14 <= len(x_groups) <= 254:
        return None

    # The first two numeric columns must be well-observed metadata: a small
    # payment-period integer followed by a plausible age.  Annual values may
    # be sparse because the right side of every group is triangular.
    payment_values = [
        _integer_value(item.text)
        for item in x_groups[0]
        if (value := _integer_value(item.text)) is not None and 0 < value <= 100
    ]
    age_values = [
        _integer_value(item.text)
        for item in x_groups[1]
        if (value := _integer_value(item.text)) is not None and 0 <= value <= 200
    ]
    minimum_metadata_evidence = max(10, len(x_groups[0]) // 4)
    if (
        len(payment_values) < minimum_metadata_evidence
        or len(age_values) < minimum_metadata_evidence
    ):
        return None

    first_numeric_x = _axis_mean(x_groups[0], "x")
    terms = Counter(
        item.text.strip()
        for item in observations
        if item.x < first_numeric_x
        and item.text.strip()
        and _numeric_value(item.text) is None
    )
    if not terms or terms.most_common(1)[0][1] < minimum_metadata_evidence:
        return None
    gender_count = sum(
        item.text.strip() in {"男", "女"} for item in observations
    )
    if gender_count < minimum_metadata_evidence:
        return None

    return _HeaderlessWideYearCandidate(
        schema=_WideYearSchema(
            headers=("保险期间", "交费期间", "投保年龄", "性别"),
            numeric_metadata_columns=(1, 2),
            term_column=0,
            age_column=2,
            gender_column=3,
            payment_column=1,
        ),
        data_top=min(item.y for item in numeric_observations),
    )


def _headerless_text_candidate(
    observations: list[VisionObservation],
) -> _HeaderlessWideYearCandidate | None:
    numeric = _merge_split_numeric_fragments(
        [item for item in observations if _numeric_value(item.text) is not None]
    )
    if len(numeric) < 100:
        return None
    height = statistics.median(item.height for item in numeric)
    groups = _drop_sparse_close_x_groups([
        group for group in _cluster_observations(
            numeric, axis="x", tolerance=max(8.0, height * 0.55)
        ) if len(group) >= 3
    ])
    if not 6 <= len(groups) <= 64:
        return None
    age_x = _axis_mean(groups[0], "x")
    minimum = max(10, len(groups[0]) // 4)
    ages = [
        _integer_value(item.text) for item in groups[0]
        if (_integer_value(item.text) is not None and 0 <= _integer_value(item.text) <= 200)
    ]
    if len(ages) < minimum:
        return None
    text = [
        item for item in observations
        if item.x < age_x and item.text.strip() not in {"", "男", "女"}
        and _numeric_value(item.text) is None
    ]
    if not text:
        return None
    tolerance = max(12.0, statistics.median(item.height for item in text) * 0.60)
    text_bands = [
        group for group in _cluster_observations(text, axis="x", tolerance=tolerance)
        if Counter(item.text.strip() for item in group).most_common(1)[0][1] >= minimum
    ]
    genders = sum(item.text.strip() in {"男", "女"} for item in observations)
    if len(text_bands) < 2 or genders < minimum:
        return None
    return _HeaderlessWideYearCandidate(
        schema=_WideYearSchema(
            headers=("保险期间", "交费期间", "投保年龄", "性别"),
            numeric_metadata_columns=(2,), term_column=0, age_column=2,
            gender_column=3, payment_column=1, text_metadata_columns=(0, 1),
            headerless_body=True, minimum_age_consistency=0.70,
        ),
        data_top=min(item.y for item in numeric),
    )


def _headerless_metadata_starts(
    observations: list[VisionObservation],
    y_centers: list[float],
    row_step: float,
    header_y: float,
    annual_x: float,
) -> list[int]:
    text = [
        item for item in observations
        if header_y < item.y <= y_centers[-1] + row_step and item.x < annual_x
        and item.text.strip() not in {"", "男", "女"}
        and _numeric_value(item.text) is None
    ]
    if not text:
        return []
    tolerance = max(12.0, statistics.median(item.height for item in text) * 0.60)
    bands = [
        group for group in _cluster_observations(text, axis="x", tolerance=tolerance)
        if Counter(item.text.strip() for item in group).most_common(1)[0][1] >= 3
    ]
    if len(bands) < 2:
        return []
    bands = sorted(sorted(bands, key=len, reverse=True)[:2], key=lambda group: _axis_mean(group, "x"))
    values = [["", "", ""] for _ in y_centers]
    for col, band in enumerate(bands):
        for item in band:
            row = _nearest_index(y_centers, item.y)
            if abs(y_centers[row] - item.y) <= row_step * 0.45:
                values[row][col] = item.text.strip()
    for item in observations:
        if item.text.strip() in {"男", "女"}:
            row = _nearest_index(y_centers, item.y)
            if abs(y_centers[row] - item.y) <= row_step * 0.45:
                values[row][2] = item.text.strip()
    raw = []
    for index in range(3, len(values) - 3):
        before, after = values[index - 3:index], values[index:index + 3]
        if any(not all(row) for row in before + after):
            continue
        left = tuple(Counter(row[col] for row in before).most_common(1)[0][0] for col in range(3))
        right = tuple(Counter(row[col] for row in after).most_common(1)[0][0] for col in range(3))
        if left != right:
            raw.append(index)
    if not raw:
        return []
    runs, current = [], [raw[0]]
    for index in raw[1:]:
        if index - current[-1] <= 3:
            current.append(index)
        else:
            runs.append(round(statistics.median(current)))
            current = [index]
    runs.append(round(statistics.median(current)))
    return runs


def _two_row_annual_header_table_to_html(
    *,
    metadata_headers: tuple[str, ...],
    year_count: int,
    rows: tuple[tuple[str, ...], ...],
) -> str:
    """Emit the common rowspan/colspan topology of annual-value tables."""

    first_header = "".join(
        f'<th rowspan="2">{escape(value)}</th>' for value in metadata_headers
    )
    annual_header = f'<th colspan="{year_count}">保单年度末</th>'
    years = "".join(f"<td>{year}</td>" for year in range(1, year_count + 1))
    body = "\n".join(
        "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        "<table>\n"
        f"<tr>{first_header}{annual_header}</tr>\n"
        f"<tr>{years}</tr>\n"
        f"{body}\n"
        "</table>\n"
    )


def _drop_sparse_close_x_groups(
    groups: list[list[VisionObservation]],
) -> list[list[VisionObservation]]:
    """Discard OCR line artifacts that sit beside a well-supported column."""

    if len(groups) < 4:
        return groups
    centers = [_axis_mean(group, "x") for group in groups]
    typical_gap = statistics.median(
        _central_gap_sample(
            [right - left for left, right in zip(centers, centers[1:])]
        )
    )
    keep = [True] * len(groups)
    for index in range(len(groups) - 1):
        gap = centers[index + 1] - centers[index]
        if gap >= typical_gap * 0.45:
            continue
        left_count = len(groups[index])
        right_count = len(groups[index + 1])
        sparse = index if left_count < right_count else index + 1
        dense_count = max(left_count, right_count)
        if len(groups[sparse]) <= dense_count * 0.50:
            keep[sparse] = False
    return [group for group, retain in zip(groups, keep) if retain]


def _merge_split_numeric_fragments(
    observations: list[VisionObservation],
) -> list[VisionObservation]:
    """Rejoin numeric tokens split by a vertical OCR-tile boundary.

    A full currency cell may be emitted as two adjacent valid-looking numbers
    (for example ``153,1`` and ``33.27``).  Their x gap is substantially
    smaller than the regular table pitch and they share a baseline.  Joining
    only such pairs before lattice clustering prevents them becoming phantom
    columns while preserving ordinary adjacent numeric columns.
    """

    if len(observations) < 8:
        return observations
    observations = _merge_row_local_decimal_fragments(observations)
    median_height = statistics.median(item.height for item in observations)
    x_groups = _cluster_observations(
        observations,
        axis="x",
        tolerance=max(8.0, median_height * 0.55),
    )
    centers = [_axis_mean(group, "x") for group in x_groups]
    gaps = [right - left for left, right in zip(centers, centers[1:])]
    plausible_gaps = _central_gap_sample(gaps)
    if not plausible_gaps:
        return observations
    typical_gap = statistics.median(plausible_gaps)
    if typical_gap <= 0:
        return observations

    consumed: set[int] = set()
    merged: list[VisionObservation] = []
    y_tolerance = max(4.0, median_height * 0.55)
    for group_index, (left_group, right_group) in enumerate(
        zip(x_groups, x_groups[1:])
    ):
        if centers[group_index + 1] - centers[group_index] >= typical_gap * 0.55:
            continue
        # A sparse nearby OCR artifact must be handled by the later column
        # filter, never concatenated with a well-supported real column.
        if min(len(left_group), len(right_group)) < max(len(left_group), len(right_group)) * 0.60:
            continue
        right_by_y = sorted(enumerate(right_group), key=lambda item: item[1].y)
        right_y = [item.y for _, item in right_by_y]
        for left_index, left in enumerate(left_group):
            if id(left) in consumed:
                continue
            lower = bisect.bisect_left(right_y, left.y - y_tolerance)
            upper = bisect.bisect_right(right_y, left.y + y_tolerance)
            candidates = [
                (right_index, right)
                for right_index, right in right_by_y[lower:upper]
                if id(right) not in consumed
            ]
            if not candidates:
                continue
            right_index, right = min(candidates, key=lambda item: abs(item[1].y - left.y))
            combined = _normalize_numeric_text(left.text) + _normalize_numeric_text(right.text)
            if _numeric_value(combined) is None:
                continue
            consumed.add(id(left))
            consumed.add(id(right))
            merged.append(
                VisionObservation(
                    x=(left.x + right.x) / 2,
                    y=(left.y + right.y) / 2,
                    width=(right.x + right.width / 2) - (left.x - left.width / 2),
                    height=max(left.height, right.height),
                    confidence=min(left.confidence, right.confidence),
                    text=combined,
                )
            )
    if not merged:
        return observations
    return [item for item in observations if id(item) not in consumed] + merged


def _merge_row_local_decimal_fragments(
    observations: list[VisionObservation],
) -> list[VisionObservation]:
    """Join one-off decimal fragments without requiring column-level support."""

    if len(observations) < 2:
        return observations
    median_height = statistics.median(item.height for item in observations)
    rows = _cluster_observations(
        observations,
        axis="y",
        tolerance=max(4.0, median_height * 0.45),
    )
    consumed: set[int] = set()
    merged: list[VisionObservation] = []
    for row in rows:
        ordered = sorted(row, key=lambda item: item.x)
        for left, right in zip(ordered, ordered[1:]):
            if id(left) in consumed or id(right) in consumed:
                continue
            if abs(left.y - right.y) > max(4.0, median_height * 0.45):
                continue
            box_gap = (
                right.x
                - right.width / 2
                - (left.x + left.width / 2)
            )
            if not -median_height * 0.10 <= box_gap <= median_height * 0.75:
                continue
            left_text = _normalize_numeric_text(left.text)
            right_text = _normalize_numeric_text(right.text)
            if re.fullmatch(r"\.\d{1,4}", right_text):
                combined = left_text + right_text
            elif (
                re.fullmatch(r"\d{3,6}", left_text)
                and re.fullmatch(r"\d{2}", right_text)
            ):
                # At a tile edge the decimal point itself is often the only
                # missing glyph: ``408`` + ``47`` represents ``408.47``.
                combined = left_text + "." + right_text
            else:
                continue
            if _numeric_value(combined) is None:
                continue
            consumed.update((id(left), id(right)))
            merged.append(
                VisionObservation(
                    x=(left.x + right.x) / 2,
                    y=(left.y + right.y) / 2,
                    width=(
                        right.x
                        + right.width / 2
                        - (left.x - left.width / 2)
                    ),
                    height=max(left.height, right.height),
                    confidence=min(left.confidence, right.confidence),
                    text=combined,
                )
            )
    if not merged:
        return observations
    return [item for item in observations if id(item) not in consumed] + merged


def _wide_year_header_candidate(
    observations: list[VisionObservation],
) -> VisionObservation | None:
    """Find a wide-table header whether OCR emits one line or several boxes."""

    combined_candidates = [
        item for item in observations if _is_wide_year_header_text(item.text)
    ]
    if combined_candidates:
        return max(combined_candidates, key=lambda item: item.confidence)

    first_year_candidates = [
        item
        for item in observations
        if re.search(r"第\s*1\s*保单年度", item.text)
    ]
    first_year_candidates.extend(
        item for item in observations if "保单年度末" in re.sub(r"\s+", "", item.text)
    )
    for first_year in sorted(
        first_year_candidates,
        key=lambda item: item.confidence,
        reverse=True,
    ):
        band = max(40.0, first_year.height * 2.5)
        peers = [
            item
            for item in observations
            if abs(item.y - first_year.y) <= band
        ]
        combined_text = "".join(
            item.text for item in sorted(peers, key=lambda item: item.x)
        )
        if not _is_wide_year_header_text(combined_text):
            continue
        return VisionObservation(
            x=first_year.x,
            y=first_year.y,
            width=first_year.width,
            height=statistics.median(item.height for item in peers),
            confidence=first_year.confidence,
            text=combined_text,
        )
    return None


def _reconstruct_single_numeric_matrix(
    observations: list[VisionObservation],
    *,
    semantic_header: VisionObservation,
    row_start_hint: int | None = None,
    header_start_hint: int | None = None,
    column_count_hint: int | None = None,
    row_count_hint: int | None = None,
) -> VisionMatrixResult | None:
    """Reconstruct a regular matrix under the best justified x anchor.

    RapidOCR reports bounding-box centres.  In financial tables, currency
    cells are often right-aligned, so their centres drift by several pixels as
    the number of digits changes even though their physical column is fixed.
    Evaluate the three box anchors independently and select only a result
    whose visible annual axis is plausible; this keeps the representation
    choice local to a table instead of hard-coding one publisher's alignment.
    """

    candidates: list[VisionMatrixResult] = []
    for anchor in ("center", "right", "left"):
        anchored_observations = [
            replace(item, x=_horizontal_anchor(item, anchor))
            for item in observations
        ]
        anchored_header = replace(
            semantic_header,
            x=_horizontal_anchor(semantic_header, anchor),
        )
        result = _reconstruct_single_numeric_matrix_for_anchor(
            anchored_observations,
            semantic_header=anchored_header,
            row_start_hint=row_start_hint,
            header_start_hint=header_start_hint,
            column_count_hint=column_count_hint,
            row_count_hint=row_count_hint,
        )
        if result is not None:
            candidates.append(result)
    if not candidates:
        return None

    def score(result: VisionMatrixResult) -> tuple[int, int, float, int]:
        header_start = result.header_starts[0] if result.header_starts else -1
        return (
            int(0 <= header_start <= 254),
            result.header_sequence_inliers,
            result.coverage,
            result.populated_cells,
        )

    return max(candidates, key=score)


def _horizontal_anchor(
    observation: VisionObservation,
    anchor: str,
) -> float:
    if anchor == "center":
        return observation.x
    if anchor == "right":
        return observation.x + observation.width / 2
    if anchor == "left":
        return observation.x - observation.width / 2
    raise ValueError(f"unknown horizontal anchor {anchor!r}")


def _reconstruct_single_numeric_matrix_for_anchor(
    observations: list[VisionObservation],
    *,
    semantic_header: VisionObservation,
    row_start_hint: int | None = None,
    header_start_hint: int | None = None,
    column_count_hint: int | None = None,
    row_count_hint: int | None = None,
) -> VisionMatrixResult | None:
    header_y = semantic_header.y
    header_axis = _numeric_header_axis_observations(
        observations,
        semantic_header=semantic_header,
    )
    minimum_body_y = header_y + max(4.0, semantic_header.height * 0.35)
    if header_axis:
        minimum_body_y = max(
            minimum_body_y,
            max(item.y + item.height / 2 for item in header_axis)
            + max(2.0, semantic_header.height * 0.05),
        )

    numeric_data = [
        item
        for item in observations
        if item.y > minimum_body_y
        and _numeric_value(item.text) is not None
    ]
    if len(numeric_data) < 16:
        return None
    # Tile seams can split a decimal value (``914`` + ``.2``) into two nearby
    # numeric columns.  The wide-matrix route already normalizes this before
    # clustering; doing the same here prevents false annual columns in regular
    # cash-value tables without changing any non-adjacent observations.
    numeric_data = _merge_split_numeric_fragments(numeric_data)

    x_tolerance = max(8.0, statistics.median(item.height for item in numeric_data) * 0.35)
    x_groups = _cluster_observations(numeric_data, axis="x", tolerance=x_tolerance)
    maximum_column_evidence = max(map(len, x_groups), default=0)
    minimum_column_evidence = max(3, math.ceil(maximum_column_evidence * 0.18))
    x_groups = [group for group in x_groups if len(group) >= minimum_column_evidence]
    x_centers = [_axis_mean(group, "x") for group in x_groups]
    # On a repeated shallow matrix, OCR can confuse nearly every isolated row
    # label (notably ``1``/``5`` as ``I``/``L``) while retaining all value
    # columns.  A successfully reconstructed sibling proves the exact table
    # width and header origin.  Restore only one missing *leading* coordinate;
    # the header sequence below must still independently align to that
    # coordinate, so an internal or trailing omission remains rejected.
    if (
        column_count_hint is not None
        and header_start_hint is not None
        and len(x_centers) == column_count_hint - 1
        and len(x_centers) >= 4
    ):
        x_gaps = [
            right - left for left, right in zip(x_centers, x_centers[1:])
        ]
        plausible_x_gaps = _central_gap_sample(x_gaps)
        if plausible_x_gaps:
            x_step = statistics.median(plausible_x_gaps)
            if x_step > 0:
                x_centers.insert(0, x_centers[0] - x_step)
    if not 4 <= len(x_centers) <= 256:
        return None
    if column_count_hint is not None and len(x_centers) != column_count_hint:
        return None

    header_band_tolerance = max(16.0, semantic_header.height * 1.5)
    header_offsets: list[int] = []
    header_observations = header_axis or observations
    for item in header_observations:
        if not header_axis and abs(item.y - header_y) > header_band_tolerance:
            continue
        value = _integer_value(item.text)
        if value is None:
            continue
        column = _nearest_index(x_centers, item.x)
        if column <= 0 or abs(x_centers[column] - item.x) > _axis_assignment_limit(
            x_centers,
            column,
        ):
            continue
        header_offsets.append(value - (column - 1))
    if not header_offsets:
        return None
    header_start, header_sequence_inliers = Counter(header_offsets).most_common(1)[0]
    if header_sequence_inliers < max(3, len(x_centers) // 5):
        return None
    if header_start_hint is not None and header_start != header_start_hint:
        return None
    if header_axis:
        maximum_observed_header = max(
            value
            for item in header_axis
            if (value := _integer_value(item.text)) is not None
        )
        represented_header_end = header_start + len(x_centers) - 2
        missing_tail = maximum_observed_header - represented_header_end
        if 1 <= missing_tail <= 4 and len(x_centers) >= 4:
            annual_gaps = _central_gap_sample(
                [
                    right - left
                    for left, right in zip(x_centers[1:], x_centers[2:])
                ]
            )
            if annual_gaps:
                annual_step = statistics.median(annual_gaps)
                if annual_step > 0:
                    x_centers.extend(
                        x_centers[-1] + annual_step * index
                        for index in range(1, missing_tail + 1)
                    )
                    # Re-evaluate the visible header evidence against the
                    # completed tail. No header cell is synthesized; the
                    # observed axis only restores its proven coordinates.
                    header_offsets = []
                    for item in header_axis:
                        value = _integer_value(item.text)
                        if value is None:
                            continue
                        column = _nearest_index(x_centers, item.x)
                        if (
                            column <= 0
                            or abs(x_centers[column] - item.x)
                            > _axis_assignment_limit(x_centers, column)
                        ):
                            continue
                        header_offsets.append(value - (column - 1))
                    if not header_offsets:
                        return None
                    (
                        header_start,
                        header_sequence_inliers,
                    ) = Counter(header_offsets).most_common(1)[0]
                    if header_start_hint is not None and header_start != header_start_hint:
                        return None

    table_numeric = [
        item
        for item in numeric_data
        if x_centers[0] - _edge_allowance(x_centers, 0)
        <= item.x
        <= x_centers[-1] + _edge_allowance(x_centers, len(x_centers) - 1)
    ]
    y_tolerance = max(5.0, statistics.median(item.height for item in table_numeric) * 0.45)
    y_groups = _cluster_observations(table_numeric, axis="y", tolerance=y_tolerance)
    dense_y_groups = [
        group for group in y_groups if len(group) >= max(4, len(x_centers) // 2)
    ]
    if len(dense_y_groups) < 3:
        return None
    dense_y = [_axis_mean(group, "y") for group in dense_y_groups]
    raw_gaps = [right - left for left, right in zip(dense_y, dense_y[1:])]
    plausible_gaps = _central_gap_sample(raw_gaps)
    if not plausible_gaps:
        return None
    row_step = statistics.median(plausible_gaps)
    if row_step <= 0:
        return None
    first_y = min(dense_y)
    y_candidates = [(_axis_mean(group, "y"), len(group)) for group in y_groups]
    y_centers = _contiguous_row_centers(
        candidates=y_candidates,
        first=first_y,
        step=row_step,
    )
    row_count = len(y_centers)
    if not 4 <= row_count <= 10_000:
        return None
    if row_count_hint is not None and row_count != row_count_hint:
        return None

    grid = [["" for _ in x_centers] for _ in y_centers]
    confidence = [[0.0 for _ in x_centers] for _ in y_centers]
    for item in table_numeric:
        column = _nearest_index(x_centers, item.x)
        row = _nearest_index(y_centers, item.y)
        if abs(x_centers[column] - item.x) > _axis_assignment_limit(x_centers, column):
            continue
        if abs(y_centers[row] - item.y) > row_step * 0.42:
            continue
        text = _normalize_numeric_text(item.text)
        if _numeric_value(text) is None:
            continue
        if item.confidence >= confidence[row][column]:
            grid[row][column] = text
            confidence[row][column] = item.confidence

    if _collapse_phantom_split_numeric_column(
        grid=grid,
        confidence=confidence,
        x_centers=x_centers,
        header_axis=header_axis,
        header_start=header_start,
    ):
        header_offsets = []
        for item in header_axis:
            value = _integer_value(item.text)
            if value is None:
                continue
            column = _nearest_index(x_centers, item.x)
            if (
                column <= 0
                or abs(x_centers[column] - item.x)
                > _axis_assignment_limit(x_centers, column)
            ):
                continue
            header_offsets.append(value - (column - 1))
        if not header_offsets:
            return None
        header_start, header_sequence_inliers = Counter(
            header_offsets
        ).most_common(1)[0]

        while (
            len(x_centers) > 4
            and all(not row[-1] for row in grid)
        ):
            x_centers.pop()
            for row in grid:
                row.pop()
            for row in confidence:
                row.pop()

    row_offsets: list[int] = []
    for index, row in enumerate(grid):
        value = _integer_value(row[0])
        if value is not None:
            row_offsets.append(value - index)
    if row_offsets:
        row_start, row_sequence_inliers = Counter(row_offsets).most_common(1)[0]
    else:
        row_start, row_sequence_inliers = 0, 0
    minimum_row_inliers = max(4, row_count // 12)
    if row_sequence_inliers < minimum_row_inliers:
        if row_start_hint is None:
            return None
        hint_inliers = sum(offset == row_start_hint for offset in row_offsets)
        strong_sibling_topology = (
            header_start_hint is not None
            and column_count_hint is not None
            and row_count_hint is not None
            and header_sequence_inliers
            >= max(5, math.ceil((len(x_centers) - 1) * 0.70))
        )
        minimum_hint_inliers = 1 if strong_sibling_topology else 2
        if row_offsets and hint_inliers < max(
            minimum_hint_inliers,
            math.ceil(len(row_offsets) * 0.60),
        ):
            return None
        row_start = row_start_hint
    for index, row in enumerate(grid):
        row[0] = str(row_start + index)

    headers = (semantic_header.text,) + tuple(
        str(header_start + index) for index in range(len(x_centers) - 1)
    )
    populated_cells = sum(bool(cell) for row in grid for cell in row)
    total_cells = len(grid) * len(x_centers)
    if populated_cells < max(20, math.ceil(total_cells * 0.35)):
        return None
    markdown = table_to_html(
        MarkdownTable(
            header=headers,
            rows=tuple(tuple(row) for row in grid),
            # Regular cash-value matrices use an ordinary first row in the
            # competition schema.  The separate ``第N保单年度`` wide route
            # retains its explicit header topology.
            header_is_explicit=False,
        )
    )
    return VisionMatrixResult(
        markdown=markdown,
        rows=len(grid),
        cols=len(x_centers),
        populated_cells=populated_cells,
        total_cells=total_cells,
        header_sequence_inliers=header_sequence_inliers,
        row_sequence_inliers=row_sequence_inliers,
        row_starts=(row_start,),
        header_starts=(header_start,),
        top_y=header_y,
        bottom_y=y_centers[-1],
    )


def _collapse_phantom_split_numeric_column(
    *,
    grid: list[list[str]],
    confidence: list[list[float]],
    x_centers: list[float],
    header_axis: list[VisionObservation],
    header_start: int,
) -> bool:
    """Collapse one tile-seam fragment column proven by the printed axis.

    Dense triangular tables sometimes split one currency column into a
    left-prefix cluster and a right-suffix cluster.  That creates a synthetic
    header origin of ``-1`` even though the visible header is the complete
    ``0..N`` sequence.  Require both signals plus repeated row-level joins
    before removing the phantom coordinate.
    """

    axis_values = sorted(
        {
            value
            for item in header_axis
            if (value := _integer_value(item.text)) is not None
        }
    )
    if (
        header_start != -1
        or len(axis_values) < 8
        or axis_values[0] < 0
        or len(x_centers) < len(axis_values) + 2
    ):
        return False
    gaps = [
        right - left for left, right in zip(x_centers[1:], x_centers[2:])
    ]
    typical_sample = _central_gap_sample(gaps)
    if not typical_sample:
        return False
    typical_gap = statistics.median(typical_sample)
    if typical_gap <= 0:
        return False

    candidates: list[tuple[int, int]] = []
    for column in range(2, len(x_centers) - 1):
        split_gap = x_centers[column + 1] - x_centers[column]
        span_from_previous = x_centers[column + 1] - x_centers[column - 1]
        if (
            split_gap >= typical_gap * 0.55
            or not typical_gap * 0.75
            <= span_from_previous
            <= typical_gap * 1.25
        ):
            continue
        join_count = sum(
            _join_phantom_numeric_cells(row[column], row[column + 1]) is not None
            for row in grid
        )
        if join_count >= max(5, math.ceil(len(grid) * 0.04)):
            candidates.append((join_count, column))
    if not candidates:
        return False

    _, column = max(candidates)
    for row_index, row in enumerate(grid):
        left = row[column]
        right = row[column + 1]
        joined = _join_phantom_numeric_cells(left, right)
        if joined is not None:
            combined = joined
        elif left and right:
            combined = (
                left
                if confidence[row_index][column]
                >= confidence[row_index][column + 1]
                else right
            )
        else:
            combined = left or right
        row[column + 1] = combined
        confidence[row_index][column + 1] = max(
            confidence[row_index][column],
            confidence[row_index][column + 1],
        )
        row.pop(column)
        confidence[row_index].pop(column)
    x_centers.pop(column)
    return True


def _join_phantom_numeric_cells(left: str, right: str) -> str | None:
    left = _normalize_numeric_text(left)
    right = _normalize_numeric_text(right)
    if not left or not right:
        return None
    if re.fullmatch(r"\d{1,3}", left) and re.fullmatch(
        r"\d{1,2}\.\d{1,2}",
        right,
    ):
        return left + right
    if re.fullmatch(r"\d{3,6}", left) and re.fullmatch(r"\d{2}", right):
        return left + "." + right
    return None


def _matrix_header_candidates(
    observations: list[VisionObservation],
) -> list[VisionObservation]:
    candidates = [item for item in observations if _is_matrix_header_text(item.text)]
    # A common actuarial-table corner header is printed in two stacked cells:
    # ``保单年度`` (or the more specific ``保单年度末``) above ``投保年龄``.
    # OCR reports those cells separately,
    # while the numeric annual headings share the first line.  Synthesize the
    # semantic corner only when both visible labels are aligned in the same
    # first column and close enough to belong to one header band.  This is
    # intentionally narrower than joining arbitrary nearby words, which could
    # turn prose into a table header.
    top_labels = [
        item
        for item in observations
        if "保单年度" in re.sub(r"\s+", "", item.text)
    ]
    age_labels = [
        item
        for item in observations
        if (
            "投保年龄" in re.sub(r"\s+", "", item.text)
            # Some insurers print the row-axis corner as just ``年龄`` with
            # ``（周岁）`` in a separate box.  It is still a safe matrix label
            # here because it must pair with a nearby explicit 保单年度 label
            # and, when diagonally offset, a visible consecutive annual axis.
            or re.fullmatch(
                r"年龄(?:[（(]周岁[）)])?",
                re.sub(r"\s+", "", item.text),
            )
        )
    ]
    for top_label in top_labels:
        for age_label in age_labels:
            vertical_gap = abs(age_label.y - top_label.y)
            max_gap = max(48.0, (top_label.height + age_label.height) * 2.5)
            alignment_limit = max(
                24.0,
                (top_label.width + age_label.width) * 0.35,
            )
            if not 0 < vertical_gap <= max_gap:
                continue
            x_offset = abs(age_label.x - top_label.x)
            if x_offset > alignment_limit and not _has_adjacent_annual_sequence(
                observations,
                annual_label=top_label,
                minimum_x=max(top_label.x, age_label.x),
            ):
                continue
            top = min(
                top_label.y - top_label.height / 2,
                age_label.y - age_label.height / 2,
            )
            bottom = max(
                top_label.y + top_label.height / 2,
                age_label.y + age_label.height / 2,
            )
            candidates.append(
                VisionObservation(
                    x=top_label.x,
                    y=(top + bottom) / 2,
                    width=max(top_label.width, age_label.width),
                    # Treat the two visible labels as one physical header
                    # box.  Using only a single-line height let the adjacent
                    # horizontal age axis leak into numeric data as a fake
                    # ``-1`` row on very dense tables.
                    height=bottom - top,
                    confidence=min(top_label.confidence, age_label.confidence),
                    # Preserve the observed annual-label wording.  The
                    # generic matrix reconstruction only requires the
                    # year/age semantics plus an adjacent `1..N` axis, so a
                    # plain ``保单年度`` table is not a weaker inference than
                    # the ``保单年度末`` variant.
                    text=f"{re.sub(r'\\s+', '', top_label.text)}\\投保年龄",
                )
            )
    # Some dense scans print only ``投保年龄`` over a horizontal age axis.
    # RapidOCR can additionally split the diagonal corner into ``投保年`` and
    # a small ``龄`` box on the following baseline.  Recover that label only
    # when no explicit policy-year label exists on the page and a run of at
    # least five consecutive integer headings is visible beside it.  Those
    # two independent constraints keep ordinary prose containing 投保年龄 out
    # of the matrix route.
    if not top_labels:
        for age_label in _standalone_age_label_candidates(observations):
            if not _has_adjacent_integer_axis(
                observations,
                label=age_label,
                minimum_x=age_label.x,
            ):
                continue
            candidates.append(age_label)
    if not candidates:
        return []
    height = statistics.median(item.height for item in candidates)
    groups = _cluster_observations(
        candidates,
        axis="y",
        tolerance=max(6.0, height * 0.50),
    )
    return sorted(
        (max(group, key=lambda item: item.confidence) for group in groups),
        key=lambda item: item.y,
    )


def _standalone_age_label_candidates(
    observations: list[VisionObservation],
) -> list[VisionObservation]:
    labels = [
        item
        for item in observations
        if re.fullmatch(
            r"投保年龄(?:[（(]周岁[）)])?",
            re.sub(r"\s+", "", item.text),
        )
    ]
    suffixes = [
        item
        for item in observations
        if re.sub(r"\s+", "", item.text) == "龄"
    ]
    for prefix in observations:
        if re.sub(r"\s+", "", prefix.text) != "投保年":
            continue
        nearby = [
            suffix
            for suffix in suffixes
            if 0 < suffix.y - prefix.y
            <= max(32.0, (prefix.height + suffix.height) * 1.5)
            and abs(suffix.x - prefix.x)
            <= max(64.0, prefix.width + suffix.width)
        ]
        if not nearby:
            continue
        suffix = min(
            nearby,
            key=lambda item: (item.y - prefix.y, abs(item.x - prefix.x)),
        )
        left = min(
            prefix.x - prefix.width / 2,
            suffix.x - suffix.width / 2,
        )
        right = max(
            prefix.x + prefix.width / 2,
            suffix.x + suffix.width / 2,
        )
        top = min(
            prefix.y - prefix.height / 2,
            suffix.y - suffix.height / 2,
        )
        bottom = max(
            prefix.y + prefix.height / 2,
            suffix.y + suffix.height / 2,
        )
        labels.append(
            VisionObservation(
                # The small final character sits at the row-axis corner and
                # is the more stable x anchor for the first numeric column.
                x=suffix.x,
                y=prefix.y,
                width=right - left,
                # Preserve the complete two-line extent.  Besides representing
                # the observed glyphs accurately, this keeps the adjacent age
                # headings in the header band rather than misclassifying them
                # as the first numeric data row.
                height=bottom - top,
                confidence=min(prefix.confidence, suffix.confidence),
                text="投保年龄",
            )
        )
    return labels


def _has_adjacent_integer_axis(
    observations: list[VisionObservation],
    *,
    label: VisionObservation,
    minimum_x: float,
    minimum_run: int = 5,
) -> bool:
    band = max(16.0, label.height * 1.5)
    values = sorted(
        {
            value
            for item in observations
            if abs(item.y - label.y) <= band and item.x > minimum_x
            if (value := _integer_value(item.text)) is not None
            and 0 <= value <= 254
        }
    )
    longest = 0
    previous: int | None = None
    for value in values:
        longest = longest + 1 if previous is not None and value == previous + 1 else 1
        if longest >= minimum_run:
            return True
        previous = value
    return False


def _numeric_header_axis_observations(
    observations: list[VisionObservation],
    *,
    semantic_header: VisionObservation,
    minimum_run: int = 5,
) -> list[VisionObservation]:
    """Return one nearby horizontal integer axis with consecutive evidence."""

    band = max(100.0, semantic_header.height * 2.0)
    candidates = [
        item
        for item in observations
        if abs(item.y - semantic_header.y) <= band
        and (value := _integer_value(item.text)) is not None
        and 0 <= value <= 254
    ]
    if not candidates:
        return []
    groups = _cluster_observations(
        candidates,
        axis="y",
        tolerance=max(5.0, semantic_header.height * 0.25),
    )

    def consecutive_run(group: list[VisionObservation]) -> int:
        values = sorted(
            {
                value
                for item in group
                if (value := _integer_value(item.text)) is not None
            }
        )
        longest = 0
        current = 0
        previous: int | None = None
        for value in values:
            current = current + 1 if previous is not None and value == previous + 1 else 1
            longest = max(longest, current)
            previous = value
        return longest

    supported = [
        (consecutive_run(group), group)
        for group in groups
        if consecutive_run(group) >= minimum_run
    ]
    if not supported:
        return []
    _, best = max(
        supported,
        key=lambda item: (
            item[0],
            len(item[1]),
            -abs(_axis_mean(item[1], "y") - semantic_header.y),
        ),
    )
    return best


def _has_adjacent_annual_sequence(
    observations: list[VisionObservation],
    *,
    annual_label: VisionObservation,
    minimum_x: float,
) -> bool:
    """Prove a diagonally offset age/year corner with visible annual labels.

    The ordinary stacked-corner form keeps both labels in one narrow x band.
    Some source tables instead print ``投保年龄`` in the next small corner
    cell.  We accept that wider geometry only when the ``保单年度末`` row also
    contains at least five consecutive annual values to its right.  Requiring
    the literal `1..5` prefix made one missed OCR glyph (for example `4`)
    discard an otherwise complete `5..75` axis.
    """
    return _has_adjacent_integer_axis(
        observations,
        label=annual_label,
        minimum_x=minimum_x,
    )


def _is_matrix_header_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if (
        "年龄" not in compact
        or len(compact) > 20
        or any(marker in compact for marker in ("说明", "代表", "下表"))
    ):
        return False
    if "保单年度末" in compact:
        return True
    return "年度" in compact and any(separator in compact for separator in ("/", "\\", "／"))


def _context_markdown(
    observations: list[VisionObservation],
    *,
    lower: float,
    upper: float,
) -> str:
    candidates = [
        item
        for item in observations
        if lower < item.y < upper
        and item.text.strip()
        and not _is_matrix_header_text(item.text)
        and _numeric_value(item.text) is None
    ]
    if not candidates:
        return ""
    line_tolerance = max(
        4.0,
        statistics.median(item.height for item in candidates) * 0.45,
    )
    groups = _cluster_observations(candidates, axis="y", tolerance=line_tolerance)
    lines: list[str] = []
    for group in groups:
        texts: list[str] = []
        for item in sorted(group, key=lambda value: value.x):
            text = item.text.strip()
            if text and (not texts or text != texts[-1]):
                texts.append(text)
        line = " ".join(texts)
        if line and (not lines or line != lines[-1]):
            lines.append(line)
    return "\n".join(lines)


def vision_ocr_available() -> bool:
    return (
        platform.system() == "Darwin"
        and shutil.which("swiftc") is not None
        and _vision_helper_source().exists()
    )


def _compile_vision_helper(cache_dir: Path) -> Path:
    source = _vision_helper_source()
    digest = hashlib.sha1(source.read_bytes()).hexdigest()[:12]
    binary_dir = cache_dir / "bin"
    binary_dir.mkdir(parents=True, exist_ok=True)
    binary = binary_dir / f"vision_ocr_{digest}"
    if binary.exists():
        return binary
    completed = subprocess.run(
        ["swiftc", str(source), "-o", str(binary)],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise VisionOCRError(
            completed.stderr.strip()
            or f"swiftc exited with {completed.returncode}"
        )
    return binary


def _vision_helper_source() -> Path:
    return Path(__file__).resolve().parents[2] / "tools" / "vision_ocr.swift"


def _parse_vision_tsv(text: str, image_slice: ImageSlice) -> list[VisionObservation]:
    observations: list[VisionObservation] = []
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
                x=image_slice.x0 + (x + width / 2) * image_slice.width,
                y=image_slice.y0 + (1 - (y + height / 2)) * image_slice.height,
                width=width * image_slice.width,
                height=height * image_slice.height,
                confidence=confidence,
                text=fields[5].strip(),
            )
        )
    return observations


def _cluster_observations(
    observations: list[VisionObservation],
    *,
    axis: str,
    tolerance: float,
) -> list[list[VisionObservation]]:
    groups: list[list[VisionObservation]] = []
    for item in sorted(observations, key=lambda value: getattr(value, axis)):
        coordinate = getattr(item, axis)
        if not groups or coordinate - _axis_mean(groups[-1], axis) > tolerance:
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


def _axis_mean(observations: list[VisionObservation], axis: str) -> float:
    return sum(getattr(item, axis) for item in observations) / len(observations)


def _nearest_index(centers: list[float], value: float) -> int:
    return min(range(len(centers)), key=lambda index: abs(centers[index] - value))


def _axis_assignment_limit(centers: list[float], index: int) -> float:
    gaps: list[float] = []
    if index > 0:
        gaps.append(centers[index] - centers[index - 1])
    if index + 1 < len(centers):
        gaps.append(centers[index + 1] - centers[index])
    return max(20.0, min(gaps, default=80.0) * 0.42)


def _edge_allowance(centers: list[float], index: int) -> float:
    return _axis_assignment_limit(centers, index)


def _central_gap_sample(gaps: list[float]) -> list[float]:
    positive = [gap for gap in gaps if gap > 0]
    if not positive:
        return []
    median = statistics.median(positive)
    return [gap for gap in positive if median * 0.70 <= gap <= median * 1.30]


def _contiguous_row_centers(
    *,
    candidates: list[tuple[float, int]],
    first: float,
    step: float,
) -> list[float]:
    """Follow the regular data-row lattice and stop before distant footers."""

    centers = [first]
    for _ in range(1, 10_000):
        # Track bounded local drift instead of projecting every row from the
        # first one. A harmless 0.03 px step error accumulates beyond the
        # assignment window on 400+ row actuarial tables and used to truncate
        # their tail even when every physical row was detected.
        expected = centers[-1] + step
        nearby = [
            (center, evidence)
            for center, evidence in candidates
            if abs(center - expected) <= step * 0.38
        ]
        if not nearby:
            break
        # Prefer an actual row with broader column evidence, then proximity.
        center, _ = max(
            nearby,
            key=lambda item: (item[1], -abs(item[0] - expected)),
        )
        centers.append(center)
    return centers


def _normalize_numeric_text(text: str) -> str:
    # OCR frequently inserts a visual word gap after a decimal separator
    # (``176. 61``) or a thousands comma (``1, 234``).  These are still one
    # numeric cell, whereas a bare digit-to-digit gap remains ambiguous and
    # must not be concatenated.
    return re.sub(r"(?<=\d)[,.]\s+(?=\d)", lambda match: match.group(0)[0], text.strip())


def _numeric_value(text: str) -> str | None:
    normalized = _normalize_numeric_text(text)
    return normalized if _NUMERIC_CELL.fullmatch(normalized) else None


def _integer_value(text: str) -> int | None:
    normalized = _normalize_numeric_text(text)
    if not re.fullmatch(r"[+-]?\d+", normalized):
        return None
    return int(normalized)
