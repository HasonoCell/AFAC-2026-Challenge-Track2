"""Shared reconstruction for sliced OCR outputs.

This module deliberately contains no API, cache, dataset, or CSV orchestration.
Prediction, baseline repair, and offline cache remerge all call the same public
entry point so structural fixes cannot silently diverge between workflows.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re

from .images import ImageSlice
from .tables import (
    parse_html_table,
    try_reconstruct_grid_tables,
    try_reconstruct_grid_tables_html,
)


_HTML_TABLE_BLOCK = re.compile(
    r"<table\b[^>]*>.*?</table\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
_HTML_TABLE_OPEN = re.compile(r"<table\b[^>]*>", flags=re.IGNORECASE)
_HTML_TABLE_CLOSE = re.compile(r"</table\s*>", flags=re.IGNORECASE)


def merge_sliced_markdown(slices: list[ImageSlice], parts: list[str]) -> str:
    """Merge OCR parts according to their image coordinates and content type."""

    if not slices or not parts:
        return ""
    if any("<table" in part.lower() for part in parts):
        reconstructed_html = try_reconstruct_grid_tables_html(slices, parts)
        if reconstructed_html is not None:
            # A long document slice may contain only prose before the first
            # table (or after the last one).  The table reconstructor quite
            # intentionally consumes only table-bearing parts, so blindly
            # taking preambles from those parts would drop an entire leading
            # slice.  Preserve the plain edge slices around the reconstructed
            # table; this is content recovery, not a document-specific rule.
            table_indices = [
                index for index, part in enumerate(parts) if _has_table_markup(part)
            ]
            first_table = min(table_indices)
            last_table = max(table_indices)
            plain_preamble = merge_markdown_parts_legacy(
                [part for part in parts[:first_table] if part.strip()]
            )
            plain_postamble = merge_markdown_parts_legacy(
                [part for part in parts[last_table + 1 :] if part.strip()]
            )
            preamble = merge_markdown_parts_legacy(
                [
                    section
                    for section in (plain_preamble, _merge_table_preambles(parts))
                    if section.strip()
                ]
            )
            postamble = merge_markdown_parts_legacy(
                [
                    section
                    for section in (_merge_table_postambles(parts), plain_postamble)
                    if section.strip()
                ]
            )
            sections = [preamble, reconstructed_html, postamble]
            return "\n\n".join(
                section.strip() for section in sections if section.strip()
            ).strip() + "\n"
    reconstructed = try_reconstruct_grid_tables(slices, parts)
    if reconstructed is not None:
        return reconstructed
    if len(slices) == 1 or all(image_slice.cols == 1 for image_slice in slices):
        return coalesce_adjacent_html_tables(merge_markdown_parts_legacy(parts))

    nonempty_parts = [part.strip() for part in parts if part.strip()]
    return "\n\n".join(nonempty_parts).strip() + "\n"


def merge_markdown_parts(parts: list[str]) -> str:
    """Merge sequential Markdown blocks with fuzzy block-overlap recovery."""

    if not parts:
        return ""

    merged_blocks = _split_markdown_blocks(parts[0])
    for part in parts[1:]:
        next_blocks = _split_markdown_blocks(part)
        block_overlap = _block_overlap(merged_blocks, next_blocks, max_blocks=8)
        if block_overlap:
            merged_blocks.extend(next_blocks[block_overlap:])
            continue

        merged_lines = _strip_edge_blank_lines("\n\n".join(merged_blocks).splitlines())
        next_lines = _strip_edge_blank_lines(part.splitlines())
        line_overlap = _line_overlap(merged_lines, next_lines, max_lines=30)
        merged_blocks = _split_markdown_blocks(
            "\n".join([*merged_lines, *next_lines[line_overlap:]])
        )
    return "\n\n".join(merged_blocks).strip() + "\n"


def merge_markdown_parts_legacy(parts: list[str]) -> str:
    """Use v033's line-overlap merge for ordinary vertical document slices.

    The block merge remains useful for table-aware and two-dimensional routes,
    but the frozen long-document comparison found line-level overlap more
    faithful for sequential prose pages.
    """

    if not parts:
        return ""

    merged_lines = _strip_edge_blank_lines(parts[0].splitlines())
    for part in parts[1:]:
        next_lines = _strip_edge_blank_lines(part.splitlines())
        overlap = _line_overlap(merged_lines, next_lines, max_lines=30)
        merged_lines.extend(next_lines[overlap:])
    return "\n".join(merged_lines).strip() + "\n"


def coalesce_adjacent_html_tables(text: str) -> str:
    """Join numbered table groups split into adjacent HTML blocks.

    Finix can close and reopen one visual table at a numbered subgroup boundary.
    Adjacency and equal width alone are insufficient: the labelled Ground Truth
    contains genuinely independent tables with exactly those properties. This
    post-process therefore also requires consecutive ``第N组`` headings. Raw
    row/cell markup is kept byte-for-byte; only the interior boundary is removed.
    """

    matches = list(_HTML_TABLE_BLOCK.finditer(text))
    if len(matches) < 2:
        return text

    output = [text[: matches[0].start()]]
    current = matches[0].group(0)
    previous_end = matches[0].end()
    for match in matches[1:]:
        separator = text[previous_end : match.start()]
        following = match.group(0)
        if (
            not separator.strip()
            and _html_tables_have_same_width(current, following)
            and _html_tables_continue_numbered_groups(current, following)
        ):
            current = _join_raw_html_tables(current, following)
        else:
            output.extend((current, separator))
            current = following
        previous_end = match.end()
    output.extend((current, text[previous_end:]))
    return "".join(output)


def _html_tables_have_same_width(left: str, right: str) -> bool:
    left_table = parse_html_table(left)
    right_table = parse_html_table(right)
    if left_table is None or right_table is None:
        return False
    left_width = len(left_table.header)
    right_width = len(right_table.header)
    return left_width >= 2 and left_width == right_width


def _html_tables_continue_numbered_groups(left: str, right: str) -> bool:
    left_groups = [int(value) for value in re.findall(r"第\s*(\d+)\s*组", left)]
    right_groups = [int(value) for value in re.findall(r"第\s*(\d+)\s*组", right)]
    return bool(
        left_groups
        and right_groups
        and min(right_groups) == max(left_groups) + 1
    )


def _join_raw_html_tables(left: str, right: str) -> str:
    left_closes = list(_HTML_TABLE_CLOSE.finditer(left))
    right_open = _HTML_TABLE_OPEN.search(right)
    if not left_closes or right_open is None:
        return left + right
    return (
        left[: left_closes[-1].start()].rstrip()
        + "\n"
        + right[right_open.end() :].lstrip()
    )


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


def _merge_table_preambles(parts: list[str]) -> str:
    seen: set[str] = set()
    merged: list[str] = []
    for part in parts:
        preamble = _table_preamble(part)
        for line in preamble.splitlines():
            normalized = " ".join(line.split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(line.strip())
    return "\n".join(merged).strip()


def _has_table_markup(text: str) -> bool:
    lowered = text.lower()
    return "<table" in lowered or (
        any(
            line.strip().startswith("|")
            and line.strip().endswith("|")
            and line.strip().count("|") >= 2
            for line in text.splitlines()
        )
    )


def _merge_table_postambles(parts: list[str]) -> str:
    seen: set[str] = set()
    merged: list[str] = []
    for part in parts:
        postamble = _table_postamble(part)
        for line in postamble.splitlines():
            normalized = " ".join(line.split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(line.strip())
    return "\n".join(merged).strip()


def _table_preamble(text: str) -> str:
    lowered = text.lower()
    positions = [position for position in [lowered.find("<table")] if position >= 0]
    for offset, line in _line_offsets(text):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
            positions.append(offset)
            break
    if not positions:
        return ""
    return text[: min(positions)].strip()


def _table_postamble(text: str) -> str:
    lowered = text.lower()
    table_end = lowered.rfind("</table>")
    if table_end >= 0:
        return text[table_end + len("</table>") :].strip()

    lines = text.splitlines()
    last_table_line = -1
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
            last_table_line = index
    if last_table_line < 0:
        return ""
    return "\n".join(lines[last_table_line + 1 :]).strip()


def _split_markdown_blocks(text: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text.strip()) if block.strip()]
    return blocks or ([text.strip()] if text.strip() else [])


def _block_overlap(left: list[str], right: list[str], max_blocks: int) -> int:
    limit = min(max_blocks, len(left), len(right))
    for size in range(limit, 1, -1):
        pairs = zip(left[-size:], right[:size], strict=True)
        if all(_blocks_match(left_block, right_block) for left_block, right_block in pairs):
            return size
    return 0


def _blocks_match(left: str, right: str) -> bool:
    normalized_left = _normalize_block(left)
    normalized_right = _normalize_block(right)
    if normalized_left == normalized_right:
        return True
    if not normalized_left or not normalized_right:
        return False
    if min(len(normalized_left), len(normalized_right)) < 80:
        return False
    return SequenceMatcher(None, normalized_left, normalized_right).ratio() >= 0.94


def _normalize_block(block: str) -> str:
    return " ".join(block.split())


def _line_offsets(text: str) -> list[tuple[int, str]]:
    offsets: list[tuple[int, str]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        offsets.append((cursor, line))
        cursor += len(line)
    return offsets
