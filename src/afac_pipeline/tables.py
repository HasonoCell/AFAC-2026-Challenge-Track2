"""Conservative Markdown table parsing and sliced-table reconstruction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

from .images import ImageSlice


@dataclass(frozen=True)
class MarkdownTable:
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


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

    grid = _expand_html_rows(parser.rows)
    if grid is None or not grid:
        return None

    header_row_count = _html_header_row_count(parser.rows)
    header = _flatten_header_rows(grid[:header_row_count])
    rows = tuple(tuple(row) for row in grid[header_row_count:])
    return MarkdownTable(header=header, rows=rows)


def parse_table(text: str) -> MarkdownTable | None:
    return parse_markdown_pipe_table(text) or parse_html_table(text)


def table_to_markdown(table: MarkdownTable) -> str:
    lines = [
        _format_pipe_row(table.header),
        _format_pipe_row(tuple("---" for _ in table.header)),
    ]
    lines.extend(_format_pipe_row(row) for row in table.rows)
    return "\n".join(lines).strip() + "\n"


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

    parsed = [parse_table(part) for part in parts]
    if any(table is None for table in parsed):
        return None
    tables = [table for table in parsed if table is not None]

    row_ids = sorted({image_slice.row for image_slice in slices})
    merged_row_tables: list[MarkdownTable] = []

    for row_id in row_ids:
        row_items = sorted(
            (
                (image_slice, table)
                for image_slice, table in zip(slices, tables, strict=True)
                if image_slice.row == row_id
            ),
            key=lambda item: item[0].col,
        )
        if not row_items:
            return None

        expected_cols = row_items[0][0].cols
        if len(row_items) != expected_cols:
            return None

        merged = _merge_horizontal_tables([table for _, table in row_items])
        if merged is None:
            return None
        merged_row_tables.append(merged)

    final_table = _merge_vertical_tables(merged_row_tables)
    if final_table is None:
        return None
    return table_to_markdown(final_table)


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
        overlap = _cell_overlap(tuple(merged_header), table.header, max_cells=5)
        merged_header.extend(table.header[overlap:])
        for index, row in enumerate(table.rows):
            merged_rows[index].extend(row[overlap:])

    return MarkdownTable(
        header=tuple(merged_header),
        rows=tuple(tuple(row) for row in merged_rows),
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
        overlap = _row_overlap(tuple(merged_rows), table.rows, max_rows=10)
        merged_rows.extend(table.rows[overlap:])

    return MarkdownTable(header=header, rows=tuple(merged_rows))


def _cell_overlap(left: tuple[str, ...], right: tuple[str, ...], max_cells: int) -> int:
    limit = min(max_cells, len(left), len(right))
    for size in range(limit, 0, -1):
        if tuple(_normalize_cell(cell) for cell in left[-size:]) == tuple(
            _normalize_cell(cell) for cell in right[:size]
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
        self.unsupported = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self.table_depth += 1
            self.table_count += 1
            if self.table_depth > 1:
                self.unsupported = True
            return

        if self.table_depth == 0:
            return

        if tag == "tr":
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
            self.current_row = []
            self.in_row = False
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


def _expand_html_rows(rows: list[list[_HTMLCell]]) -> tuple[tuple[str, ...], ...] | None:
    grid: list[list[str | None]] = []

    for row_index, row in enumerate(rows):
        _ensure_grid_row(grid, row_index)
        col_index = 0
        for cell in row:
            while col_index < len(grid[row_index]) and grid[row_index][col_index] is not None:
                col_index += 1

            for dr in range(cell.rowspan):
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
        header.append(" / ".join(parts) if parts else f"Column {col + 1}")
    return tuple(header)
