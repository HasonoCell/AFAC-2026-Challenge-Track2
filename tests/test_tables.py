from __future__ import annotations

import unittest

from afac_pipeline.images import ImageSlice
from afac_pipeline.tables import (
    parse_html_table,
    parse_markdown_pipe_table,
    table_to_markdown,
    try_reconstruct_grid_tables,
)


def _slice(row: int, col: int, rows: int, cols: int) -> ImageSlice:
    return ImageSlice(
        file_name=f"slice_r{row}_c{col}.jpg",
        image_bytes=b"",
        x0=(col - 1) * 100,
        x1=col * 100,
        y0=(row - 1) * 100,
        y1=row * 100,
        width=100,
        height=100,
        row=row,
        col=col,
        rows=rows,
        cols=cols,
    )


class MarkdownTableTest(unittest.TestCase):
    def test_parse_and_format_simple_pipe_table(self) -> None:
        table = parse_markdown_pipe_table(
            """
            | A | B |
            | --- | --- |
            | 1 | 2 |
            """
        )

        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.header, ("A", "B"))
        self.assertEqual(table.rows, (("1", "2"),))
        self.assertEqual(
            table_to_markdown(table),
            "| A | B |\n| --- | --- |\n| 1 | 2 |\n",
        )

    def test_reconstructs_horizontal_table_slices(self) -> None:
        reconstructed = try_reconstruct_grid_tables(
            slices=[_slice(1, 1, 1, 2), _slice(1, 2, 1, 2)],
            parts=[
                "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n",
                "| C | D |\n| --- | --- |\n| 5 | 6 |\n| 7 | 8 |\n",
            ],
        )

        self.assertEqual(
            reconstructed,
            "| A | B | C | D |\n"
            "| --- | --- | --- | --- |\n"
            "| 1 | 2 | 5 | 6 |\n"
            "| 3 | 4 | 7 | 8 |\n",
        )

    def test_parse_simple_html_table(self) -> None:
        table = parse_html_table(
            """
            <table>
              <tr><th>A</th><th>B</th></tr>
              <tr><td>1</td><td>2</td></tr>
            </table>
            """
        )

        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.header, ("A", "B"))
        self.assertEqual(table.rows, (("1", "2"),))

    def test_reconstructs_horizontal_html_table_slices(self) -> None:
        reconstructed = try_reconstruct_grid_tables(
            slices=[_slice(1, 1, 1, 2), _slice(1, 2, 1, 2)],
            parts=[
                "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>",
                "<table><tr><th>C</th><th>D</th></tr><tr><td>3</td><td>4</td></tr></table>",
            ],
        )

        self.assertEqual(
            reconstructed,
            "| A | B | C | D |\n"
            "| --- | --- | --- | --- |\n"
            "| 1 | 2 | 3 | 4 |\n",
        )

    def test_expands_html_table_with_colspan(self) -> None:
        table = parse_html_table(
            "<table><tr><th colspan='2'>AB</th></tr><tr><td>1</td><td>2</td></tr></table>"
        )

        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.header, ("AB", "AB"))
        self.assertEqual(table.rows, (("1", "2"),))

    def test_expands_html_table_with_rowspan(self) -> None:
        table = parse_html_table(
            """
            <table>
              <tr><th>A</th><th>B</th></tr>
              <tr><td rowspan="2">X</td><td>1</td></tr>
              <tr><td>2</td></tr>
            </table>
            """
        )

        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.header, ("A", "B"))
        self.assertEqual(table.rows, (("X", "1"), ("X", "2")))

    def test_flattens_multi_row_html_headers_with_spans(self) -> None:
        table = parse_html_table(
            """
            <table>
              <tr><th rowspan="2">Fund</th><th colspan="2">Rate</th></tr>
              <tr><th>A</th><th>B</th></tr>
              <tr><td>Alpha</td><td>1%</td><td>2%</td></tr>
            </table>
            """
        )

        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.header, ("Fund", "Rate / A", "Rate / B"))
        self.assertEqual(table.rows, (("Alpha", "1%", "2%"),))

    def test_removes_exact_horizontal_overlap_columns(self) -> None:
        reconstructed = try_reconstruct_grid_tables(
            slices=[_slice(1, 1, 1, 2), _slice(1, 2, 1, 2)],
            parts=[
                "| A | B |\n| --- | --- |\n| 1 | 2 |\n",
                "| B | C |\n| --- | --- |\n| 2 | 3 |\n",
            ],
        )

        self.assertEqual(
            reconstructed,
            "| A | B | C |\n"
            "| --- | --- | --- |\n"
            "| 1 | 2 | 3 |\n",
        )

    def test_reconstructs_vertical_table_slices_and_removes_duplicate_rows(self) -> None:
        reconstructed = try_reconstruct_grid_tables(
            slices=[_slice(1, 1, 2, 1), _slice(2, 1, 2, 1)],
            parts=[
                "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n",
                "| A | B |\n| --- | --- |\n| 3 | 4 |\n| 5 | 6 |\n",
            ],
        )

        self.assertEqual(
            reconstructed,
            "| A | B |\n"
            "| --- | --- |\n"
            "| 1 | 2 |\n"
            "| 3 | 4 |\n"
            "| 5 | 6 |\n",
        )

    def test_returns_none_for_non_table_content(self) -> None:
        reconstructed = try_reconstruct_grid_tables(
            slices=[_slice(1, 1, 1, 1)],
            parts=["# Not a table\n"],
        )

        self.assertIsNone(reconstructed)


if __name__ == "__main__":
    unittest.main()
