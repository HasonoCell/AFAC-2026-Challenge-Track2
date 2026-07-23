"""Conservative Markdown table parsing and sliced-table reconstruction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape
from html.parser import HTMLParser

from .images import ImageSlice


@dataclass(frozen=True)
class MarkdownTable:
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    header_is_explicit: bool = True
    source_html: str | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class _HTMLCell:
    text: str
    rowspan: int
    colspan: int
    is_header: bool


def parse_markdown_pipe_table(text: str) -> MarkdownTable | None:
    """Parse a simple GitHub-style pipe table.

    The parser is intentionally conservative: if the text contains non-table
    content or inconsistent cell counts, it returns ``None`` instead of trying
    to guess.  This keeps the production pipeline safe: uncertain slice outputs
    can fall back to annotated concatenation.
    """

    lines = [
        line.strip()
        for line in text.strip().splitlines()
        if line.strip() and not _is_html_comment(line.strip())
    ]
    if len(lines) < 2:
        return None
    if any("|" not in line for line in lines):
        return None
    if not _is_separator_line(lines[1]):
        return None

    header = tuple(_split_pipe_row(lines[0]))
    separator = _split_pipe_row(lines[1])
    if not header or len(separator) != len(header):
        return None

    rows: list[tuple[str, ...]] = []
    for line in lines[2:]:
        cells = tuple(_split_pipe_row(line))
        if len(cells) != len(header):
            return None
        rows.append(cells)

    return MarkdownTable(header=header, rows=tuple(rows))


def parse_html_table(text: str) -> MarkdownTable | None:
    """Parse an HTML table and expand ``rowspan``/``colspan`` into a grid."""

    if "<table" not in text.lower():
        return None

    parser = _SimpleHTMLTableParser()
    parser.feed(text)
    if parser.unsupported or parser.table_count != 1 or not parser.rows:
        return None

    grid = _expand_html_rows(parser.rows, parser.row_groups)
    if grid is None or not grid:
        return None

    header_row_count = _html_header_row_count(parser.rows)
    header = _flatten_header_rows(grid[:header_row_count])
    rows = tuple(tuple(row) for row in grid[header_row_count:])
    return MarkdownTable(
        header=header,
        rows=rows,
        header_is_explicit=_has_explicit_html_header(parser.rows),
        source_html=_single_html_table_markup(text),
    )


def parse_table(text: str) -> MarkdownTable | None:
    return parse_markdown_pipe_table(text) or parse_html_table(text)


def parse_sliced_table(text: str) -> MarkdownTable | None:
    """Parse one tile, accepting consecutive same-width HTML table blocks.

    Finix can split a continuous visual table within a crop into several HTML
    ``<table>`` blocks. This relaxed parser is deliberately used only by grid
    reconstruction; normal document parsing remains conservative about
    multiple independent tables.
    """

    pipe_table = parse_markdown_pipe_table(text)
    if pipe_table is not None:
        return pipe_table
    if "<table" not in text.lower():
        return None

    parser = _SimpleHTMLTableParser()
    parser.feed(text)
    if parser.unsupported or not parser.rows:
        return None
    grid = _expand_html_rows(parser.rows, parser.row_groups)
    if grid is None or not grid:
        return None
    header_row_count = _html_header_row_count(parser.rows)
    header = _flatten_header_rows(grid[:header_row_count])
    rows = tuple(tuple(row) for row in grid[header_row_count:])
    return MarkdownTable(
        header=header,
        rows=rows,
        header_is_explicit=_has_explicit_html_header(parser.rows),
        source_html=(
            _single_html_table_markup(text) if parser.table_count == 1 else None
        ),
    )


def table_to_markdown(table: MarkdownTable) -> str:
    lines = [
        _format_pipe_row(table.header),
        _format_pipe_row(tuple("---" for _ in table.header)),
    ]
    lines.extend(_format_pipe_row(row) for row in table.rows)
    return "\n".join(lines).strip() + "\n"


def retain_complete_pipe_table_rows(text: str, *, max_bytes: int) -> str:
    """Fit one trailing pipe table into a UTF-8 budget without malformed rows."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    lines = text.splitlines(keepends=True)
    table_start = next(
        (
            index
            for index in range(len(lines) - 1)
            if "|" in lines[index] and _is_separator_line(lines[index + 1].strip())
        ),
        None,
    )
    if table_start is None:
        raise ValueError("oversized output has no pipe table")
    prefix = "".join(lines[:table_start])
    table = parse_markdown_pipe_table("".join(lines[table_start:]))
    if table is None:
        raise ValueError("oversized output is not one complete trailing pipe table")
    retained: list[tuple[str, ...]] = []
    for row in table.rows:
        candidate = prefix + table_to_markdown(
            MarkdownTable(header=table.header, rows=tuple((*retained, row)))
        )
        if len(candidate.encode("utf-8")) > max_bytes:
            break
        retained.append(row)
    compact = prefix + table_to_markdown(
        MarkdownTable(header=table.header, rows=tuple(retained))
    )
    if len(compact.encode("utf-8")) > max_bytes or not retained:
        raise ValueError("table header cannot fit the requested byte budget")
    return compact


def table_to_html(table: MarkdownTable) -> str:
    if table.source_html is not None:
        return table.source_html
    lines = ["<table>"]
    if table.header:
        lines.append(
            _format_html_row(table.header, header=table.header_is_explicit)
        )
    lines.extend(_format_html_row(row, header=False) for row in table.rows)
    lines.append("</table>")
    return "\n".join(lines).strip() + "\n"


_HTML_TABLE_BLOCK = re.compile(
    r"<table\b[^>]*>.*?</table\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)


def html_tables_to_markdown(text: str) -> str:
    """Replace complete HTML tables by cell-equivalent pipe Markdown.

    The B-list evaluator has empirically accepted pipe Markdown and rejected
    otherwise valid HTML-table submissions.  Keep this transformation next to
    the table parser instead of treating it as a last-minute CSV workaround:
    every replacement is parsed again and must preserve the expanded header
    and cell grid exactly.  Tables containing a literal pipe are rejected
    rather than silently splitting a cell in Markdown.
    """

    matches = list(_HTML_TABLE_BLOCK.finditer(text))
    if not matches:
        return text

    output: list[str] = []
    cursor = 0
    for match in matches:
        source = match.group(0)
        table = parse_html_table(source)
        if table is None:
            raise ValueError("HTML table cannot be converted to pipe Markdown")
        cells = (*table.header, *(cell for row in table.rows for cell in row))
        if any("|" in cell for cell in cells):
            raise ValueError(
                "HTML table contains a literal pipe and cannot be safely "
                "converted to pipe Markdown"
            )
        rewritten = table_to_markdown(table)
        reparsed = parse_markdown_pipe_table(rewritten)
        if (
            reparsed is None
            or reparsed.header != table.header
            or reparsed.rows != table.rows
        ):
            raise ValueError("HTML-to-Markdown conversion changed table cells")
        output.append(text[cursor : match.start()])
        output.append(rewritten)
        cursor = match.end()
    output.append(text[cursor:])
    return "".join(output)


def try_reconstruct_grid_tables(
    slices: list[ImageSlice],
    parts: list[str],
) -> str | None:
    """Try to reconstruct a complete Markdown table from sliced table outputs.

    Returns Markdown when every slice can be parsed and merged deterministically.
    Returns ``None`` when the content is not a simple table or when rows/columns
    cannot be aligned safely.
    """

    if not slices or len(slices) != len(parts):
        return None

    parsed_items = _parse_nonempty_sliced_tables(slices, parts)
    if parsed_items is None:
        return None

    row_ids = sorted({image_slice.row for image_slice in slices})
    merged_row_tables: list[MarkdownTable] = []

    for row_id in row_ids:
        row_items = sorted(
            (
                (image_slice, table)
                for image_slice, table in parsed_items
                if image_slice.row == row_id
            ),
            key=lambda item: item[0].col,
        )
        if not row_items:
            continue

        merged = _merge_horizontal_tables([table for _, table in row_items])
        if merged is None:
            return None
        merged_row_tables.append(merged)

    final_table = _merge_vertical_tables(merged_row_tables)
    if final_table is None:
        return None
    return table_to_markdown(final_table)


def try_reconstruct_grid_tables_html(
    slices: list[ImageSlice],
    parts: list[str],
) -> str | None:
    """Reconstruct sliced tables and emit HTML instead of pipe Markdown."""

    if not slices or len(slices) != len(parts):
        return None

    parsed_items = _parse_nonempty_sliced_tables(slices, parts)
    if parsed_items is None:
        return try_reconstruct_grid_table_bands_html(slices, parts)

    row_ids = sorted({image_slice.row for image_slice in slices})
    merged_row_tables: list[MarkdownTable] = []

    for row_id in row_ids:
        row_items = sorted(
            (
                (image_slice, table)
                for image_slice, table in parsed_items
                if image_slice.row == row_id
            ),
            key=lambda item: item[0].col,
        )
        if not row_items:
            continue

        merged = _merge_horizontal_tables([table for _, table in row_items])
        if merged is None:
            return try_reconstruct_grid_table_bands_html(slices, parts)
        merged_row_tables.append(merged)

    final_table = _merge_vertical_tables(merged_row_tables)
    if final_table is None:
        return try_reconstruct_grid_table_bands_html(slices, parts)
    return _reconstructed_table_to_html(final_table, slices=slices, parts=parts)


def try_reconstruct_grid_table_bands_html(
    slices: list[ImageSlice],
    parts: list[str],
) -> str | None:
    """Reconstruct horizontal table bands when vertical headers do not align.

    Large financial tables often have a triangular blank tail: within one
    image-row band, a right-hand crop may contain fewer visible data rows than
    a left-hand crop.  The normal full-grid reconstruction correctly refuses
    to guess in that situation.  This fallback preserves the image-row bands,
    pads only trailing missing rows, removes repeated context rows, and joins
    the coordinate-contiguous bands into one HTML table.  Rows may retain
    different widths when the visual table itself has a blank triangular tail.
    """

    if not slices or len(slices) != len(parts):
        return None

    parsed_items: list[tuple[ImageSlice, MarkdownTable]] = []
    for image_slice, part in zip(slices, parts, strict=True):
        if not part.strip():
            continue
        # This fallback only accepts a strict single-table result. Fragmented
        # tile output is refined upstream and must not be guessed here.
        table = parse_table(part)
        if table is None or not table.header:
            continue
        parsed_items.append((image_slice, table))

    reconstructed_band_tables: list[MarkdownTable] = []
    for row_id in sorted({image_slice.row for image_slice in slices}):
        row_items = sorted(
            (
                (image_slice, table)
                for image_slice, table in parsed_items
                if image_slice.row == row_id
            ),
            key=lambda item: item[0].col,
        )
        if not row_items:
            continue
        max_rows = max(len(table.rows) for _, table in row_items)
        padded = [_pad_table_rows(table, max_rows) for _, table in row_items]
        merged = _merge_horizontal_tables(padded)
        if merged is not None:
            # A crop containing consecutive HTML table fragments is excluded
            # from the strict band cells, but it still carries useful th/td
            # convention evidence. Recover only that boolean signal; never
            # import cells from a tile rejected by the strict fallback.
            header_evidence = [
                evidence
                for image_slice, part in zip(slices, parts, strict=True)
                if image_slice.row == row_id and part.strip()
                if (evidence := parse_sliced_table(part)) is not None
            ]
            if header_evidence:
                merged = MarkdownTable(
                    header=merged.header,
                    rows=merged.rows,
                    header_is_explicit=_horizontal_header_is_explicit(
                        header_evidence
                    ),
                )
            merged = _trim_trailing_blank_rows(merged)
            reconstructed_band_tables.append(
                _trim_trailing_unkeyed_fragments(merged)
            )

    if len(reconstructed_band_tables) < 2:
        return None
    reconstructed_band_tables = _remove_repeated_top_context_from_bands(
        reconstructed_band_tables,
        max_rows=5,
    )
    if _bands_have_numeric_row_continuity(reconstructed_band_tables):
        return _reconstructed_table_to_html(
            _join_coordinate_table_bands(reconstructed_band_tables),
            slices=slices,
            parts=parts,
        )
    reconstructed_band_tables = _join_contextually_continuous_band_groups(
        reconstructed_band_tables
    )
    reconstructed_bands = [table_to_html(table) for table in reconstructed_band_tables]
    return "\n\n".join(band.strip() for band in reconstructed_bands).strip() + "\n"


def _reconstructed_table_to_html(
    table: MarkdownTable,
    *,
    slices: list[ImageSlice],
    parts: list[str],
) -> str:
    table, collapsed_seams = _collapse_numeric_header_seam_columns(table)
    grouped_header = _try_grouped_numeric_header_html(
        table,
        slices,
        parts,
        allow_heading_inference=collapsed_seams >= 2,
    )
    return grouped_header if grouped_header is not None else table_to_html(table)


def _try_grouped_numeric_header_html(
    table: MarkdownTable,
    slices: list[ImageSlice],
    parts: list[str],
    *,
    allow_heading_inference: bool = False,
) -> str | None:
    """Restore a two-row grouped numeric header from independent crop evidence.

    Financial age/year matrices often have one row-spanned key cell and one
    group label spanning a long consecutive numeric leaf header. A horizontal
    crop can retain the rowspan in one tile and the colspan in another. The
    flattened grid otherwise repeats ``Group / leaf`` and loses both spans.

    Reconstruction is intentionally narrow: exactly one semantic rowspan key,
    exactly one semantic colspan label, a matching numeric leaf subsequence,
    and a globally consecutive leaf sequence are all required. Data rows are
    never modified.
    """

    if not slices or len(slices) != len(parts) or len(table.header) < 5:
        return None
    first_row_id = min(image_slice.row for image_slice in slices)
    key_labels: dict[str, str] = {}
    group_evidence: dict[str, tuple[str, list[tuple[str, ...]]]] = {}
    heading_evidence: list[tuple[int, str]] = []

    for image_slice, part in zip(slices, parts, strict=True):
        if image_slice.row != first_row_id or not part.strip():
            continue
        heading_evidence.extend(_markdown_headings_before_table(part))
        parser = _SimpleHTMLTableParser()
        parser.feed(part)
        if parser.unsupported or parser.table_count != 1 or not parser.rows:
            continue
        for row_index, row in enumerate(parser.rows):
            for cell in row:
                if cell.rowspan >= 2 and _is_semantic_header_label(cell.text):
                    key_labels.setdefault(_header_label_key(cell.text), cell.text)
                if (
                    cell.colspan < 2
                    or not cell.is_header
                    or not _is_semantic_header_label(cell.text)
                    or row_index + 1 >= len(parser.rows)
                    or parser.row_groups[row_index + 1]
                    != parser.row_groups[row_index]
                ):
                    continue
                leaf_row = parser.rows[row_index + 1]
                if not leaf_row or not all(leaf.is_header for leaf in leaf_row):
                    continue
                leaf_values = tuple(
                    leaf.text
                    for leaf in leaf_row
                    for _ in range(leaf.colspan)
                )
                if len(leaf_values) < 2:
                    continue
                label_key = _header_label_key(cell.text)
                if label_key not in group_evidence:
                    group_evidence[label_key] = (cell.text, [])
                group_evidence[label_key][1].append(leaf_values)

    inferred_from_heading = False
    if len(key_labels) == 1 and len(group_evidence) == 1:
        key_label = next(iter(key_labels.values()))
        group_label, evidence_leaf_rows = next(iter(group_evidence.values()))
    elif (
        allow_heading_inference
        and len(key_labels) <= 1
        and not group_evidence
        and _is_semantic_header_label(table.header[0])
    ):
        key_label = next(iter(key_labels.values()), table.header[0])
        group_label = _unique_deepest_heading(heading_evidence)
        if group_label is None:
            return None
        evidence_leaf_rows = []
        inferred_from_heading = True
    else:
        return None
    if _normalize_cell(table.header[0]) != _normalize_cell(key_label):
        return None

    group_key = _header_label_key(group_label)
    leaves: list[str] = []
    for cell in table.header[1:]:
        normalized = _normalize_cell(cell)
        if " / " in normalized:
            pieces = [piece.strip() for piece in normalized.split(" / ")]
            if pieces and _header_label_key(pieces[0]) == group_key:
                normalized = pieces[-1]
        leaves.append(normalized)

    nonempty_leaves = [leaf for leaf in leaves if leaf]
    if len(nonempty_leaves) < 4 or len(leaves) - len(nonempty_leaves) > 2:
        return None
    if leaves[: len(nonempty_leaves)] != nonempty_leaves:
        return None
    try:
        numeric_leaves = [int(leaf) for leaf in nonempty_leaves]
    except ValueError:
        return None
    if any(
        right - left != 1
        for left, right in zip(numeric_leaves, numeric_leaves[1:])
    ):
        return None

    if not inferred_from_heading:
        normalized_evidence = [
            tuple(_normalize_cell(value) for value in row)
            for row in evidence_leaf_rows
        ]
        if not any(
            _tuple_is_contiguous_subsequence(tuple(nonempty_leaves), evidence)
            for evidence in normalized_evidence
        ):
            return None

    tag = "th" if inferred_from_heading or table.header_is_explicit else "td"
    lines = [
        "<table>",
        "<tr>"
        f'<{tag} rowspan="2">{escape(key_label)}</{tag}>'
        f'<{tag} colspan="{len(leaves)}">{escape(group_label)}</{tag}>'
        "</tr>",
        _format_html_row(tuple(leaves), header=table.header_is_explicit),
    ]
    lines.extend(_format_html_row(row, header=False) for row in table.rows)
    lines.append("</table>")
    return "\n".join(lines).strip() + "\n"


def _collapse_numeric_header_seam_columns(
    table: MarkdownTable,
) -> tuple[MarkdownTable, int]:
    """Collapse crop-edge columns split out of a consecutive numeric header.

    A hard vertical crop can turn one numeric column into an unlabeled prefix
    column followed by a labeled suffix column. Multiple regularly occurring
    blank labels inside an otherwise consecutive age/year header are strong
    coordinate evidence of that artifact. This rule never applies to a single
    gap, a skipped number, or a non-numeric schema.
    """

    header = tuple(_normalize_cell(cell) for cell in table.header)
    if len(header) < 8 or not _is_semantic_header_label(header[0]):
        return table, 0
    leaves = header[1:]
    blank_indexes = [index + 1 for index, cell in enumerate(leaves) if not cell]
    nonempty = [cell for cell in leaves if cell]
    if len(blank_indexes) < 2 or len(blank_indexes) > 10:
        return table, 0
    try:
        numbers = [int(cell) for cell in nonempty]
    except ValueError:
        return table, 0
    if len(numbers) < 5 or any(
        right - left != 1 for left, right in zip(numbers, numbers[1:])
    ):
        return table, 0
    if any(
        index <= 1
        or index + 1 >= len(header)
        or not re.fullmatch(r"\d+", header[index - 1])
        or not re.fullmatch(r"\d+", header[index + 1])
        for index in blank_indexes
    ):
        return table, 0

    width = len(header)
    padded_rows = [
        row + ("",) * (width - len(row))
        for row in table.rows
    ]
    for index in blank_indexes:
        paired_numeric_rows = sum(
            _is_numeric_table_cell(row[index])
            and _is_numeric_table_cell(row[index + 1])
            for row in padded_rows
        )
        if paired_numeric_rows < 3:
            return table, 0

    blank_index_set = set(blank_indexes)
    collapsed_header: list[str] = []
    collapsed_rows: list[list[str]] = [[] for _ in padded_rows]
    index = 0
    while index < width:
        if index in blank_index_set:
            collapsed_header.append(header[index + 1])
            for output_row, row in zip(collapsed_rows, padded_rows, strict=True):
                output_row.append(
                    _merge_numeric_seam_cells(row[index], row[index + 1])
                )
            index += 2
            continue
        collapsed_header.append(table.header[index])
        for output_row, row in zip(collapsed_rows, padded_rows, strict=True):
            output_row.append(row[index])
        index += 1

    return (
        MarkdownTable(
            header=tuple(collapsed_header),
            rows=tuple(tuple(row) for row in collapsed_rows),
            header_is_explicit=table.header_is_explicit,
        ),
        len(blank_indexes),
    )


def _is_numeric_table_cell(value: str) -> bool:
    return bool(re.fullmatch(r"[+-]?[\d,]+(?:\.\d+)?", _normalize_cell(value)))


def _merge_numeric_seam_cells(left: str, right: str) -> str:
    left = _normalize_cell(left)
    right = _normalize_cell(right)
    if not left:
        return right
    if not right or left == right:
        return left

    suffix = re.fullmatch(r"\d{0,2}\.(\d+)", right)
    if re.fullmatch(r"\d{2,}", left) and suffix:
        return f"{left}.{suffix.group(1)}"
    if (
        left.isdigit()
        and right.isdigit()
        and len(left) <= 2
        and len(left) + len(right) <= 5
    ):
        return left + right
    return left if len(left) >= len(right) else right


def _markdown_headings_before_table(text: str) -> list[tuple[int, str]]:
    preamble = re.split(r"<table\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    headings: list[tuple[int, str]] = []
    for match in re.finditer(r"(?m)^\s*(#{1,6})\s+(.+?)\s*$", preamble):
        label = _normalize_cell(match.group(2))
        if _is_semantic_header_label(label):
            headings.append((len(match.group(1)), label))
    return headings


def _unique_deepest_heading(
    headings: list[tuple[int, str]],
) -> str | None:
    if not headings:
        return None
    deepest = max(level for level, _ in headings)
    labels: dict[str, str] = {}
    for level, label in headings:
        if level == deepest:
            labels.setdefault(_header_label_key(label), label)
    return next(iter(labels.values())) if len(labels) == 1 else None


def _is_semantic_header_label(value: str) -> bool:
    return bool(re.search(r"[A-Za-z\u3400-\u9fff]", _normalize_cell(value)))


def _header_label_key(value: str) -> str:
    return re.sub(r"[\s()（）]", "", _normalize_cell(value)).casefold()


def _tuple_is_contiguous_subsequence(
    values: tuple[str, ...],
    candidate: tuple[str, ...],
) -> bool:
    if not candidate or len(candidate) > len(values):
        return False
    return any(
        values[index : index + len(candidate)] == candidate
        for index in range(len(values) - len(candidate) + 1)
    )


def _join_coordinate_table_bands(tables: list[MarkdownTable]) -> MarkdownTable:
    """Join ordered row bands while treating later OCR headers as data rows."""

    first = tables[0]
    rows = list(first.rows)
    previous = first
    for table in tables[1:]:
        if (
            _normalize_row(table.header) != _normalize_row(first.header)
            and not _band_header_is_boundary_artifact(previous, table)
            and not _boundary_has_monotonic_duplicate_key(previous, table)
        ):
            rows.append(table.header)
        rows.extend(table.rows)
        previous = table
    width = max(len(row) for row in (first.header, *rows))

    def pad(row: tuple[str, ...]) -> tuple[str, ...]:
        return row + ("",) * (width - len(row))

    return MarkdownTable(
        header=pad(first.header),
        rows=tuple(pad(row) for row in rows),
        header_is_explicit=first.header_is_explicit,
    )


def _band_header_is_boundary_artifact(
    previous: MarkdownTable,
    current: MarkdownTable,
    *,
    max_cells: int = 8,
) -> bool:
    """Reject a repeated or reversed OCR row at an image-band boundary."""

    if not previous.rows or not current.rows:
        return False
    previous_row = previous.rows[-1]
    first_row = current.rows[0]
    for index in range(
        min(max_cells, len(previous_row), len(current.header), len(first_row))
    ):
        previous_key = _small_integer_cell(previous_row, index)
        header_key = _small_integer_cell(current.header, index)
        first_key = _small_integer_cell(first_row, index)
        if None in (previous_key, header_key, first_key):
            continue
        assert previous_key is not None
        assert header_key is not None
        assert first_key is not None
        if (
            header_key == previous_key
            and first_key - header_key in {1, 2}
        ):
            return True
        if (
            first_key - previous_key in {1, 2}
            and header_key >= first_key
        ):
            return True
    return False


def _bands_have_numeric_row_continuity(tables: list[MarkdownTable]) -> bool:
    """Require content evidence before treating image bands as one table.

    Financial tables commonly use a monotonically increasing year or age in
    one of their leading metadata columns.  The first column can remain fixed
    (for example, retirement age) while a later age/year column increments.
    Independent logical tables can occupy adjacent image bands as well, so
    coordinates alone are insufficient. Join the entire grid only when every
    adjacent boundary contains a small leading integer that continues by one
    (or by two, allowing one failed OCR row). Partially continuous grids are
    grouped separately so a failed boundary remains a logical table break.
    """

    forward_continuity = [
        _boundary_has_numeric_row_continuity(left, right)
        for left, right in zip(tables, tables[1:])
    ]
    duplicate_overlap = [
        _boundary_has_monotonic_duplicate_key(left, right)
        and len(right.header) < len(left.header)
        for left, right in zip(tables, tables[1:])
    ]
    continuity = [
        forward or duplicate
        for forward, duplicate in zip(
            forward_continuity,
            duplicate_overlap,
            strict=True,
        )
    ]
    if all(continuity):
        # A same-key boundary is weaker than a forward 1/2 step: two logical
        # tables can independently contain the same age range. Accept it only
        # inside a grid of at least three bands, with another independently
        # continuous boundary, and when the later band is narrower as expected
        # for a triangular table tail.
        return (
            not any(duplicate_overlap)
            or (
                len(tables) >= 3
                and any(forward_continuity)
                and all(
                    forward or duplicate
                    for forward, duplicate in zip(
                        forward_continuity,
                        duplicate_overlap,
                        strict=True,
                    )
                )
            )
        )
    # The first image band can contain only a wide, explicit column header,
    # leaving no comparable row key at its lower edge. If every subsequent
    # data-band boundary is continuous, attach that header band as well. At
    # least three bands are required so this cannot merge an arbitrary pair.
    return (
        len(tables) >= 3
        and tables[0].header_is_explicit
        and all(continuity[1:])
    )


def _join_contextually_continuous_band_groups(
    tables: list[MarkdownTable],
) -> list[MarkdownTable]:
    if len(tables) < 2:
        return tables

    groups: list[list[MarkdownTable]] = [[tables[0]]]
    for left, right in zip(tables, tables[1:]):
        if _boundary_has_contextual_numeric_continuity(left, right):
            groups[-1].append(right)
        else:
            groups.append([right])
    return [
        _join_coordinate_table_bands(group) if len(group) > 1 else group[0]
        for group in groups
    ]


def _boundary_has_contextual_numeric_continuity(
    left: MarkdownTable,
    right: MarkdownTable,
) -> bool:
    left_rows = ((left.header,) + left.rows)[-4:]
    right_rows = ((right.header,) + right.rows)[:4]
    return any(
        _rows_have_contextual_numeric_continuity(left_row, right_row)
        for left_row in left_rows
        for right_row in right_rows
    )


def _boundary_has_numeric_row_continuity(
    left: MarkdownTable,
    right: MarkdownTable,
) -> bool:
    left_values = _last_leading_small_integers(left)
    right_values = _first_leading_small_integers(right)
    if any(
        right_value - left_value in {1, 2}
        for left_value in left_values
        for right_value in right_values
    ):
        return True

    # Adjacent image bands can both contain the boundary row. OCR values in
    # that row often differ slightly across crops, so an exact row comparison
    # is too strict. A shared small-integer key is considered a one-row
    # overlap only when each side independently continues the same increasing
    # sequence. This distinguishes ``54,55 | 55,56`` from two unrelated
    # tables that merely happen to contain the same number.
    # A single badly aligned horizontal tile can leave a stray row at the edge
    # of an otherwise continuous image band. Search a narrow boundary window,
    # but require multiple matching metadata cells so unrelated logical tables
    # with coincidentally adjacent numbers remain separate.
    return _boundary_has_contextual_numeric_continuity(left, right)


def _boundary_has_monotonic_duplicate_key(
    left: MarkdownTable,
    right: MarkdownTable,
    *,
    max_cells: int = 8,
) -> bool:
    left_rows = [
        row
        for row in (left.header,) + left.rows
        if any(_normalize_cell(cell) for cell in row)
    ][-4:]
    right_rows = [
        row
        for row in (right.header,) + right.rows
        if any(_normalize_cell(cell) for cell in row)
    ][:4]
    if len(left_rows) < 2 or len(right_rows) < 2:
        return False

    for cell_index in range(
        min(max_cells, *(len(row) for row in (*left_rows, *right_rows)))
    ):
        left_previous = _small_integer_cell(left_rows[-2], cell_index)
        left_boundary = _small_integer_cell(left_rows[-1], cell_index)
        right_boundary = _small_integer_cell(right_rows[0], cell_index)
        right_next = _small_integer_cell(right_rows[1], cell_index)
        if None in (left_previous, left_boundary, right_boundary, right_next):
            continue
        assert left_previous is not None
        assert left_boundary is not None
        assert right_boundary is not None
        assert right_next is not None
        if (
            left_boundary == right_boundary
            and left_boundary - left_previous in {1, 2}
            and right_next - right_boundary in {1, 2}
        ):
            return True
    return False


def _small_integer_cell(row: tuple[str, ...], index: int) -> int | None:
    normalized = _normalize_cell(row[index])
    if not re.fullmatch(r"[+-]?\d+", normalized):
        return None
    value = int(normalized)
    return value if 0 <= value <= 130 else None


def _rows_have_contextual_numeric_continuity(
    left: tuple[str, ...],
    right: tuple[str, ...],
    *,
    max_cells: int = 8,
) -> bool:
    left_integers = _leading_small_integer_cells(left, max_cells=max_cells)
    right_integers = _leading_small_integer_cells(right, max_cells=max_cells)
    for left_index, left_value in left_integers:
        for right_index, right_value in right_integers:
            if left_index != right_index:
                continue
            difference = right_value - left_value
            if difference not in {0, 1, 2}:
                continue
            matching_context = sum(
                1
                for index, (left_cell, right_cell) in enumerate(
                    zip(left[:max_cells], right[:max_cells])
                )
                if index != left_index
                and _normalize_cell(left_cell)
                and _normalize_cell(left_cell) == _normalize_cell(right_cell)
            )
            required_context = 3 if difference == 0 else 2
            if matching_context >= required_context:
                return True
    return False


def _first_leading_small_integers(table: MarkdownTable) -> tuple[int, ...]:
    for row in (table.header,) + table.rows:
        values = _leading_small_integers(row)
        if values:
            return values
    return ()


def _last_leading_small_integers(table: MarkdownTable) -> tuple[int, ...]:
    for row in reversed((table.header,) + table.rows):
        values = _leading_small_integers(row)
        if values:
            return values
    return ()


def _leading_small_integers(row: tuple[str, ...], *, max_cells: int = 8) -> tuple[int, ...]:
    return tuple(
        value
        for _, value in _leading_small_integer_cells(row, max_cells=max_cells)
    )


def _leading_small_integer_cells(
    row: tuple[str, ...],
    *,
    max_cells: int = 8,
) -> tuple[tuple[int, int], ...]:
    values: list[tuple[int, int]] = []
    for index, cell in enumerate(row[:max_cells]):
        normalized = _normalize_cell(cell)
        if re.fullmatch(r"[+-]?\d+", normalized):
            value = int(normalized)
            if 0 <= value <= 130:
                values.append((index, value))
    return tuple(values)


def _first_integer_row_key(table: MarkdownTable) -> int | None:
    for row in (table.header,) + table.rows:
        value = _integer_row_key(row)
        if value is not None:
            return value
    return None


def _last_integer_row_key(table: MarkdownTable) -> int | None:
    for row in reversed((table.header,) + table.rows):
        value = _integer_row_key(row)
        if value is not None:
            return value
    return None


def _integer_row_key(row: tuple[str, ...]) -> int | None:
    if not row:
        return None
    normalized = _normalize_cell(row[0])
    if not re.fullmatch(r"[+-]?\d+", normalized):
        return None
    return int(normalized)


def _remove_repeated_top_context_from_bands(
    tables: list[MarkdownTable],
    *,
    max_rows: int,
) -> list[MarkdownTable]:
    """Drop repeated top context before emitting independent row bands.

    Context-aware image slices prepend the first visual rows to every lower
    band.  When band widths differ, full vertical reconstruction cannot align
    them and this fallback emits separate HTML tables.  Without an explicit
    cleanup, those prepended rows reappear between every pair of bands and
    severely damage reading order.  Compare the leading row key (the stable
    left context column) with the first band and promote the first real row to
    the next band's header after removing the repeated prefix.
    """

    if len(tables) < 2 or max_rows <= 0:
        return tables

    reference_rows = (tables[0].header,) + tables[0].rows
    reference_keys = tuple(
        _normalize_cell(row[0])
        for row in reference_rows[:max_rows]
        if row and _normalize_cell(row[0])
    )
    if not reference_keys:
        return tables

    cleaned = [tables[0]]
    for table in tables[1:]:
        candidate_rows = (table.header,) + table.rows
        repeated = 0
        for index, row in enumerate(candidate_rows[: len(reference_keys)]):
            if not row or _normalize_cell(row[0]) != reference_keys[index]:
                break
            repeated += 1
        remaining = candidate_rows[repeated:]
        # A single repeated row is commonly the legitimate table schema
        # header and should remain on each independent HTML table.  Context
        # crops repeat multiple visual data rows, which is the case we remove.
        if repeated >= 2 and remaining:
            table = MarkdownTable(
                header=remaining[0],
                rows=tuple(remaining[1:]),
                header_is_explicit=False,
            )
        cleaned.append(table)
    return cleaned


def _pad_table_rows(table: MarkdownTable, target_rows: int) -> MarkdownTable:
    if len(table.rows) >= target_rows:
        return table
    blank_row = tuple("" for _ in table.header)
    return MarkdownTable(
        header=table.header,
        rows=table.rows + (blank_row,) * (target_rows - len(table.rows)),
        header_is_explicit=table.header_is_explicit,
    )


def _trim_trailing_blank_rows(table: MarkdownTable) -> MarkdownTable:
    rows = list(table.rows)
    while rows and not any(_normalize_cell(cell) for cell in rows[-1]):
        rows.pop()
    if len(rows) == len(table.rows):
        return table
    return MarkdownTable(
        header=table.header,
        rows=tuple(rows),
        header_is_explicit=table.header_is_explicit,
    )


def _trim_trailing_unkeyed_fragments(
    table: MarkdownTable,
    *,
    max_key_cells: int = 8,
) -> MarkdownTable:
    """Remove a crop-edge row fragment that has no leading row metadata.

    Different horizontal crops of the same image band can expose one extra
    visual row near a slanted boundary. After top-aligned padding this appears
    as a final row with content only in later columns, while the preceding two
    rows carry a monotonically increasing age/year key. The next vertical band
    reads that visual row in full. Keeping the suffix fragment creates a fake
    row and prevents band continuity; dropping it loses no keyed table row.
    """

    rows = list(table.rows)
    while len(rows) >= 2:
        last = rows[-1]
        leading_width = min(max_key_cells, len(last))
        if (
            not any(_normalize_cell(cell) for cell in last)
            or any(_normalize_cell(cell) for cell in last[:leading_width])
        ):
            break
        previous = rows[-2]
        sequence_found = False
        if len(rows) >= 3:
            before_previous = rows[-3]
            for cell_index in range(
                min(max_key_cells, len(previous), len(before_previous))
            ):
                older = _small_integer_cell(before_previous, cell_index)
                newer = _small_integer_cell(previous, cell_index)
                if older is not None and newer is not None and newer - older in {1, 2}:
                    sequence_found = True
                    break
        elif (
            _is_semantic_header_label(table.header[0])
            and sum(
                bool(re.fullmatch(r"\d+", _normalize_cell(cell)))
                for cell in table.header[1:]
            )
            >= 4
            and any(
                _small_integer_cell(previous, cell_index) is not None
                for cell_index in range(min(max_key_cells, len(previous)))
            )
        ):
            # The first image band can contain a schema row, one keyed data
            # row, and a suffix-only continuation emitted as a second OCR row.
            # The suffix has no row key and the next vertical band will read
            # that visual row in full, so it is the same crop-edge artifact as
            # the monotonic three-row case above.
            sequence_found = True
        if not sequence_found:
            break
        rows.pop()

    if len(rows) == len(table.rows):
        return table
    return MarkdownTable(
        header=table.header,
        rows=tuple(rows),
        header_is_explicit=table.header_is_explicit,
    )


def _parse_nonempty_sliced_tables(
    slices: list[ImageSlice],
    parts: list[str],
) -> list[tuple[ImageSlice, MarkdownTable]] | None:
    """Parse useful tile output while allowing intentionally skipped blank tiles.

    Content-density routing can omit an entirely blank tile. Such an omission
    must not make the whole grid fall back to concatenation, but a nonempty
    response that cannot be parsed remains unsafe and rejects reconstruction.
    """

    parsed_items: list[tuple[ImageSlice, MarkdownTable]] = []
    for image_slice, part in zip(slices, parts, strict=True):
        if not part.strip():
            continue
        table = parse_sliced_table(part)
        if table is None:
            return None
        parsed_items.append((image_slice, table))
    return parsed_items


def _merge_horizontal_tables(tables: list[MarkdownTable]) -> MarkdownTable | None:
    if not tables:
        return None
    if len(tables) == 1:
        return tables[0]

    row_count = len(tables[0].rows)
    if any(len(table.rows) != row_count for table in tables):
        return None

    merged_header = list(tables[0].header)
    merged_rows = [list(row) for row in tables[0].rows]

    for table in tables[1:]:
        prefix_context = _repeated_prefix_columns(
            tuple(merged_header),
            tuple(tuple(row) for row in merged_rows),
            table,
            max_cells=8,
        )
        right_header = table.header[prefix_context:]
        right_rows = [row[prefix_context:] for row in table.rows]
        if not right_header:
            continue

        overlap = _table_column_overlap(
            tuple(merged_header),
            tuple(tuple(row) for row in merged_rows),
            right_header,
            tuple(right_rows),
            max_cells=5,
        )
        merged_header.extend(right_header[overlap:])
        for index, row in enumerate(table.rows):
            merged_rows[index].extend(right_rows[index][overlap:])

    return MarkdownTable(
        header=tuple(merged_header),
        rows=tuple(tuple(row) for row in merged_rows),
        # One crop can spuriously emit ``th`` while neighbouring crops use
        # ``td`` for the same visual row. Let the horizontal band vote instead
        # of promoting the entire reconstructed width from a single outlier.
        # A semantic schema label provides supporting evidence when exactly a
        # third of the crops retain ``th``; pure numeric matrix rows still need
        # a majority. Ties stay explicit so clipped real headers are retained.
        header_is_explicit=_horizontal_header_is_explicit(tables),
    )


def _horizontal_header_is_explicit(tables: list[MarkdownTable]) -> bool:
    explicit_headers = sum(table.header_is_explicit for table in tables)
    has_semantic_header = any(
        re.search(r"[A-Za-z\u3400-\u9fff]", _normalize_cell(cell))
        for table in tables
        for cell in table.header
    )
    return (
        explicit_headers * 2 >= len(tables)
        or (
            has_semantic_header
            and explicit_headers > 0
            and explicit_headers * 3 >= len(tables)
        )
    )


def _merge_vertical_tables(tables: list[MarkdownTable]) -> MarkdownTable | None:
    if not tables:
        return None
    if len(tables) == 1:
        return tables[0]

    header = tables[0].header
    merged_rows = list(tables[0].rows)

    for table in tables[1:]:
        if table.header != header:
            return None
        prefix_context = _repeated_prefix_rows(
            tuple(merged_rows),
            table.rows,
            max_rows=5,
        )
        candidate_rows = table.rows[prefix_context:]
        overlap = _row_overlap(tuple(merged_rows), candidate_rows, max_rows=10)
        merged_rows.extend(candidate_rows[overlap:])

    return MarkdownTable(
        header=header,
        rows=tuple(merged_rows),
        header_is_explicit=tables[0].header_is_explicit,
    )


def _table_column_overlap(
    left_header: tuple[str, ...],
    left_rows: tuple[tuple[str, ...], ...],
    right_header: tuple[str, ...],
    right_rows: tuple[tuple[str, ...], ...],
    *,
    max_cells: int,
) -> int:
    """Find exact duplicate boundary columns using full column vectors.

    Header-only overlap is unsafe because blank crop-edge cells compare equal
    and can silently delete distinct columns. Require every header/body value
    in the candidate vectors to agree and at least two nonempty matched values.
    """

    if len(left_rows) != len(right_rows):
        return 0
    limit = min(max_cells, len(left_header), len(right_header))
    for size in range(limit, 0, -1):
        left_vectors = tuple(
            (
                _normalize_cell(left_header[-size + offset]),
                *(
                    _normalize_cell(row[-size + offset])
                    for row in left_rows
                ),
            )
            for offset in range(size)
        )
        right_vectors = tuple(
            (
                _normalize_cell(right_header[offset]),
                *(
                    _normalize_cell(row[offset])
                    for row in right_rows
                ),
            )
            for offset in range(size)
        )
        nonempty_evidence = sum(
            bool(value)
            for vector in left_vectors
            for value in vector
        )
        if left_vectors == right_vectors and nonempty_evidence >= 2:
            return size
    return 0


def _repeated_prefix_columns(
    left_header: tuple[str, ...],
    left_rows: tuple[tuple[str, ...], ...],
    right_table: MarkdownTable,
    max_cells: int,
) -> int:
    limit = min(max_cells, len(left_header), len(right_table.header))
    for size in range(limit, 0, -1):
        if tuple(_normalize_cell(cell) for cell in left_header[:size]) != tuple(
            _normalize_cell(cell) for cell in right_table.header[:size]
        ):
            continue
        if all(
            _normalize_row(left_rows[row_index][:size])
            == _normalize_row(right_table.rows[row_index][:size])
            for row_index in range(len(left_rows))
        ):
            return size
    return 0


def _row_overlap(
    left: tuple[tuple[str, ...], ...],
    right: tuple[tuple[str, ...], ...],
    max_rows: int,
) -> int:
    limit = min(max_rows, len(left), len(right))
    for size in range(limit, 0, -1):
        if tuple(_normalize_row(row) for row in left[-size:]) == tuple(
            _normalize_row(row) for row in right[:size]
        ):
            return size
    return 0


def _repeated_prefix_rows(
    left: tuple[tuple[str, ...], ...],
    right: tuple[tuple[str, ...], ...],
    max_rows: int,
) -> int:
    limit = min(max_rows, len(left), len(right))
    for size in range(limit, 0, -1):
        if tuple(_normalize_row(row) for row in left[:size]) == tuple(
            _normalize_row(row) for row in right[:size]
        ):
            return size
    return 0


def _normalize_row(row: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_normalize_cell(cell) for cell in row)


def _normalize_cell(cell: str) -> str:
    return re.sub(r"\s+", " ", cell).strip()


def _split_pipe_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _format_pipe_row(cells: tuple[str, ...]) -> str:
    return "| " + " | ".join(cells) + " |"


def _format_html_row(cells: tuple[str, ...], *, header: bool) -> str:
    tag = "th" if header else "td"
    return "<tr>" + "".join(f"<{tag}>{escape(cell)}</{tag}>" for cell in cells) + "</tr>"


def _is_separator_line(line: str) -> bool:
    cells = _split_pipe_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def _is_html_comment(line: str) -> bool:
    return line.startswith("<!--") and line.endswith("-->")


class _SimpleHTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.table_depth = 0
        self.table_count = 0
        self.in_row = False
        self.in_cell = False
        self.current_cell_text: list[str] = []
        self.current_cell_rowspan = 1
        self.current_cell_colspan = 1
        self.current_cell_is_header = False
        self.current_row: list[_HTMLCell] = []
        self.rows: list[list[_HTMLCell]] = []
        self.row_groups: list[int] = []
        self.current_row_group = 0
        self.next_row_group = 1
        self.unsupported = False

    def _start_row_group(self) -> None:
        self.current_row_group = self.next_row_group
        self.next_row_group += 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self.table_depth += 1
            self.table_count += 1
            if self.table_depth > 1:
                self.unsupported = True
            elif self.table_depth == 1:
                # Consecutive table blocks are distinct implicit row groups;
                # a rowspan from one block must never occupy the next block.
                self._start_row_group()
            return

        if self.table_depth == 0:
            return

        if tag in {"thead", "tbody", "tfoot"}:
            self._start_row_group()
        elif tag == "tr":
            if self.in_row:
                self.unsupported = True
            self.in_row = True
            self.current_row = []
        elif tag in {"td", "th"}:
            if self.in_cell:
                self.unsupported = True
            span = _read_span_attrs(attrs)
            if span is None:
                self.unsupported = True
                span = (1, 1)
            self.in_cell = True
            self.current_cell_text = []
            self.current_cell_rowspan = span[0]
            self.current_cell_colspan = span[1]
            self.current_cell_is_header = tag == "th"
        elif tag == "br" and self.in_cell:
            self.current_cell_text.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self.in_cell:
            self.current_row.append(
                _HTMLCell(
                    text=_normalize_cell("".join(self.current_cell_text)),
                    rowspan=self.current_cell_rowspan,
                    colspan=self.current_cell_colspan,
                    is_header=self.current_cell_is_header,
                )
            )
            self.current_cell_text = []
            self.current_cell_rowspan = 1
            self.current_cell_colspan = 1
            self.current_cell_is_header = False
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if self.current_row:
                self.rows.append(self.current_row)
                self.row_groups.append(self.current_row_group)
            self.current_row = []
            self.in_row = False
        elif tag in {"thead", "tbody", "tfoot"}:
            # Any rows outside the just-closed explicit group form a fresh
            # implicit group rather than inheriting its rowspan occupancy.
            self._start_row_group()
        elif tag == "table" and self.table_depth > 0:
            self.table_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_cell_text.append(data)


def _read_span_attrs(attrs: list[tuple[str, str | None]]) -> tuple[int, int] | None:
    rowspan = 1
    colspan = 1
    for name, value in attrs:
        normalized_name = name.lower()
        if normalized_name not in {"rowspan", "colspan"}:
            continue
        if value is None:
            return None
        try:
            parsed = int(value)
        except ValueError:
            return None
        if parsed < 1:
            return None
        if normalized_name == "rowspan":
            rowspan = parsed
        else:
            colspan = parsed
    return rowspan, colspan


def _expand_html_rows(
    rows: list[list[_HTMLCell]],
    row_groups: list[int] | None = None,
) -> tuple[tuple[str, ...], ...] | None:
    if row_groups is None:
        row_groups = [0] * len(rows)
    if len(row_groups) != len(rows):
        return None
    grid: list[list[str | None]] = []

    for row_index, row in enumerate(rows):
        _ensure_grid_row(grid, row_index)
        col_index = 0
        for cell in row:
            while col_index < len(grid[row_index]) and grid[row_index][col_index] is not None:
                col_index += 1

            # HTML rowspans are confined to their row group (thead/tbody/tfoot)
            # and cannot create rows beyond the group's actual <tr> elements.
            available_rowspan = 0
            for target_row in range(row_index, len(rows)):
                if row_groups[target_row] != row_groups[row_index]:
                    break
                available_rowspan += 1
            for dr in range(min(cell.rowspan, available_rowspan)):
                target_row = row_index + dr
                _ensure_grid_width(grid, target_row, col_index + cell.colspan)
                for dc in range(cell.colspan):
                    target_col = col_index + dc
                    if grid[target_row][target_col] is not None:
                        return None
                    grid[target_row][target_col] = cell.text
            col_index += cell.colspan

    if not grid:
        return None

    width = max(len(row) for row in grid)
    if width == 0:
        return None

    expanded: list[tuple[str, ...]] = []
    for row in grid:
        padded = row + [None] * (width - len(row))
        expanded.append(tuple("" if cell is None else cell for cell in padded))
    return tuple(expanded)


def _ensure_grid_row(grid: list[list[str | None]], row_index: int) -> None:
    while len(grid) <= row_index:
        grid.append([])


def _ensure_grid_width(grid: list[list[str | None]], row_index: int, width: int) -> None:
    _ensure_grid_row(grid, row_index)
    while len(grid[row_index]) < width:
        grid[row_index].append(None)


def _html_header_row_count(rows: list[list[_HTMLCell]]) -> int:
    count = 0
    for row in rows:
        if row and all(cell.is_header for cell in row):
            count += 1
            continue
        break
    return max(1, count)


def _has_explicit_html_header(rows: list[list[_HTMLCell]]) -> bool:
    return bool(rows and rows[0] and all(cell.is_header for cell in rows[0]))


def _single_html_table_markup(text: str) -> str | None:
    matches = re.findall(
        r"<table\b[^>]*>.*?</table\s*>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if len(matches) != 1:
        return None
    return matches[0].strip() + "\n"


def _flatten_header_rows(rows: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    if not rows:
        return ()

    width = len(rows[0])
    header: list[str] = []
    for col in range(width):
        parts: list[str] = []
        for row in rows:
            cell = _normalize_cell(row[col])
            if cell and cell not in parts:
                parts.append(cell)
        # Empty cells are structural information, not missing schema names.
        # Inventing ``Column N`` here leaks a local parser placeholder into the
        # submitted OCR text at every crop boundary and directly hurts both
        # text edit distance and TEDS. Preserve the source cell as empty.
        header.append(" / ".join(parts) if parts else "")
    return tuple(header)
