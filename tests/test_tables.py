from __future__ import annotations

import unittest

from afac_pipeline.images import ImageSlice
from afac_pipeline.tables import (
    html_tables_to_markdown,
    parse_html_table,
    parse_markdown_pipe_table,
    parse_sliced_table,
    repair_split_numeric_pipe_cells,
    table_to_html,
    table_to_markdown,
    retain_complete_pipe_table_rows,
    try_reconstruct_grid_table_bands_html,
    try_reconstruct_grid_tables,
    try_reconstruct_grid_tables_html,
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
    def test_repairs_decimal_fragments_without_changing_row_width(self) -> None:
        source = (
            "| A | B | C | D | E | F |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| 1 | 441 | .02 | 5677. |  | 84 |\n"
        )

        repaired = repair_split_numeric_pipe_cells(source)

        table = parse_markdown_pipe_table(repaired)
        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(
            table.rows,
            (("1", "441.02", "5677.84", "", "", ""),),
        )

    def test_row_budget_keeps_header_and_complete_rows(self) -> None:
        source = "标题\n| A | B |\n| --- | --- |\n| 一 | 二 |\n| 三 | 四 |\n"
        compact = retain_complete_pipe_table_rows(
            source,
            max_bytes=len("标题\n| A | B |\n| --- | --- |\n| 一 | 二 |\n".encode()),
        )
        self.assertEqual(compact, "标题\n| A | B |\n| --- | --- |\n| 一 | 二 |\n")

    def test_row_budget_keeps_complete_prefix_across_multiple_tables(self) -> None:
        source = (
            "标题\n\n"
            "| A | B |\n| --- | --- |\n| 一 | 二 |\n\n"
            "小标题\n\n"
            "| C | D |\n| --- | --- |\n| 三 | 四 |\n| 五 | 六 |\n"
        )
        budget = len(
            (
                "标题\n\n"
                "| A | B |\n| --- | --- |\n| 一 | 二 |\n\n"
                "小标题\n\n"
                "| C | D |\n| --- | --- |\n| 三 | 四 |\n"
            ).encode()
        )

        compact = retain_complete_pipe_table_rows(source, max_bytes=budget)

        self.assertEqual(len(compact.encode()), budget)
        self.assertIn("| 一 | 二 |", compact)
        self.assertIn("| 三 | 四 |", compact)
        self.assertNotIn("| 五 | 六 |", compact)
    def test_html_conversion_preserves_expanded_grid(self) -> None:
        converted = html_tables_to_markdown(
            '<table><tr><th rowspan="2">A</th><th colspan="2">B</th></tr>'
            '<tr><th>1</th><th>2</th></tr>'
            '<tr><td>x</td><td>y</td><td>z</td></tr></table>'
        )

        self.assertNotIn("<table", converted)
        parsed = parse_markdown_pipe_table(converted)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.header, ("A", "B / 1", "B / 2"))
        self.assertEqual(parsed.rows, (("x", "y", "z"),))

    def test_html_conversion_flattens_td_annual_leaf_axis_as_header(self) -> None:
        converted = html_tables_to_markdown(
            "<table><tr><th rowspan=\"2\">年龄</th><th colspan=\"5\">保单年度末</th></tr>"
            "<tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr>"
            "<tr><td>20</td><td>1.1</td><td>2.2</td><td>3.3</td><td>4.4</td><td>5.5</td></tr></table>"
        )
        parsed = parse_markdown_pipe_table(converted)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(
            parsed.header,
            ("年龄", "保单年度末 / 1", "保单年度末 / 2", "保单年度末 / 3", "保单年度末 / 4", "保单年度末 / 5"),
        )
        self.assertEqual(parsed.rows, (("20", "1.1", "2.2", "3.3", "4.4", "5.5"),))

    def test_html_parser_recovers_a_single_implicit_cell_closure(self) -> None:
        table = parse_html_table(
            "<table><tr><th>A</th><th>B</th></tr>"
            "<tr><td>first/td><td>second</td></tr></table>"
        )
        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.rows, (("first/td>", "second"),))

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
        self.assertTrue(table.header_is_explicit)

    def test_html_roundtrip_preserves_td_only_first_row_convention(self) -> None:
        table = parse_html_table(
            "<table><tr><td>A</td><td>B</td></tr>"
            "<tr><td>1</td><td>2</td></tr></table>"
        )

        self.assertIsNotNone(table)
        assert table is not None
        self.assertFalse(table.header_is_explicit)
        rebuilt = table_to_html(table)
        self.assertNotIn("<th>", rebuilt)
        self.assertIn("<tr><td>A</td><td>B</td></tr>", rebuilt)

    def test_html_parser_does_not_invent_names_for_empty_header_cells(self) -> None:
        table = parse_html_table(
            "<table><tr><td>A</td><td></td></tr>"
            "<tr><td>1</td><td>2</td></tr></table>"
        )

        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.header, ("A", ""))
        self.assertNotIn("Column", table_to_html(table))

    def test_grid_reconstruction_preserves_td_only_header_convention(self) -> None:
        reconstructed = try_reconstruct_grid_tables_html(
            slices=[_slice(1, 1, 1, 2), _slice(1, 2, 1, 2)],
            parts=[
                "<table><tr><td>A</td><td>B</td></tr>"
                "<tr><td>1</td><td>2</td></tr></table>",
                "<table><tr><td>C</td><td>D</td></tr>"
                "<tr><td>3</td><td>4</td></tr></table>",
            ],
        )

        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        self.assertNotIn("<th>", reconstructed)
        self.assertIn("<tr><td>A</td><td>B</td><td>C</td><td>D</td></tr>", reconstructed)

    def test_horizontal_header_vote_ignores_one_th_outlier(self) -> None:
        reconstructed = try_reconstruct_grid_tables_html(
            slices=[
                _slice(1, 1, 1, 3),
                _slice(1, 2, 1, 3),
                _slice(1, 3, 1, 3),
            ],
            parts=[
                "<table><tr><th>10</th></tr><tr><td>1</td></tr></table>",
                "<table><tr><td>20</td></tr><tr><td>2</td></tr></table>",
                "<table><tr><td>30</td></tr><tr><td>3</td></tr></table>",
            ],
        )

        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        self.assertNotIn("<th>", reconstructed)
        self.assertIn("<tr><td>10</td><td>20</td><td>30</td></tr>", reconstructed)

    def test_horizontal_header_vote_keeps_semantic_one_third_header(self) -> None:
        reconstructed = try_reconstruct_grid_tables_html(
            slices=[
                _slice(1, 1, 1, 3),
                _slice(1, 2, 1, 3),
                _slice(1, 3, 1, 3),
            ],
            parts=[
                "<table><tr><td>投保年龄</td></tr><tr><td>20</td></tr></table>",
                "<table><tr><th>0</th></tr><tr><td>100</td></tr></table>",
                "<table><tr><td>1</td></tr><tr><td>200</td></tr></table>",
            ],
        )

        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        self.assertIn(
            "<tr><th>投保年龄</th><th>0</th><th>1</th></tr>",
            reconstructed,
        )

    def test_single_tile_html_reconstruction_preserves_span_topology(self) -> None:
        source = (
            '<table class="source"><thead><tr>'
            '<th rowspan="2">Fund</th><th colspan="2">Rate</th>'
            '</tr><tr><th>A</th><th>B</th></tr></thead>'
            '<tbody><tr><td>Alpha</td><td>1%</td><td>2%</td></tr>'
            '</tbody></table>'
        )

        reconstructed = try_reconstruct_grid_tables_html(
            slices=[_slice(1, 1, 1, 1)],
            parts=[source],
        )

        self.assertEqual(reconstructed, source + "\n")
        assert reconstructed is not None
        self.assertIn('rowspan="2"', reconstructed)
        self.assertIn('colspan="2"', reconstructed)

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

    def test_rowspan_does_not_cross_from_thead_into_tbody(self) -> None:
        table = parse_html_table(
            "<table><thead><tr><th rowspan='2'>Key</th><th>Value</th></tr></thead>"
            "<tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
        )

        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.header, ("Key", "Value"))
        self.assertEqual(table.rows, (("1", "2"),))

    def test_parses_consecutive_html_table_blocks_for_one_slice(self) -> None:
        table = parse_sliced_table(
            "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
            "<table><tr><td>3</td><td>4</td></tr></table>"
        )

        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.header, ("A", "B"))
        self.assertEqual(table.rows, (("1", "2"), ("3", "4")))

    def test_rowspan_does_not_leak_into_a_consecutive_table_block(self) -> None:
        table = parse_sliced_table(
            "<table><tr><th rowspan='2'>A</th></tr></table>"
            "<table><tr><td>B</td></tr></table>"
        )

        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.header, ("A",))
        self.assertEqual(table.rows, (("B",),))

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

    def test_horizontal_reconstruction_does_not_overlap_blank_headers_alone(self) -> None:
        reconstructed = try_reconstruct_grid_tables_html(
            slices=[_slice(1, 1, 1, 2), _slice(1, 2, 1, 2)],
            parts=[
                "<table><tr><td>A</td><td></td></tr>"
                "<tr><td>1</td><td>X</td></tr></table>",
                "<table><tr><td></td><td>B</td></tr>"
                "<tr><td>Y</td><td>2</td></tr></table>",
            ],
        )

        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        table = parse_html_table(reconstructed)
        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(len(table.header), 4)
        self.assertEqual(table.rows, (("1", "X", "Y", "2"),))

    def test_removes_repeated_left_context_columns(self) -> None:
        reconstructed = try_reconstruct_grid_tables(
            slices=[_slice(1, 1, 1, 2), _slice(1, 2, 1, 2)],
            parts=[
                "| Product | Age | Premium |\n"
                "| --- | --- | --- |\n"
                "| Alpha | 30 | 100 |\n"
                "| Beta | 31 | 110 |\n",
                "| Product | Cash Value | Dividend |\n"
                "| --- | --- | --- |\n"
                "| Alpha | 1000 | 20 |\n"
                "| Beta | 1100 | 21 |\n",
            ],
        )

        self.assertEqual(
            reconstructed,
            "| Product | Age | Premium | Cash Value | Dividend |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| Alpha | 30 | 100 | 1000 | 20 |\n"
            "| Beta | 31 | 110 | 1100 | 21 |\n",
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

    def test_removes_repeated_top_context_rows(self) -> None:
        reconstructed = try_reconstruct_grid_tables(
            slices=[_slice(1, 1, 2, 1), _slice(2, 1, 2, 1)],
            parts=[
                "| Age | Cash Value |\n"
                "| --- | --- |\n"
                "| 30 | 1000 |\n"
                "| 31 | 1100 |\n",
                "| Age | Cash Value |\n"
                "| --- | --- |\n"
                "| 30 | 1000 |\n"
                "| 32 | 1200 |\n",
            ],
        )

        self.assertEqual(
            reconstructed,
            "| Age | Cash Value |\n"
            "| --- | --- |\n"
            "| 30 | 1000 |\n"
            "| 31 | 1100 |\n"
            "| 32 | 1200 |\n",
        )

    def test_returns_none_for_non_table_content(self) -> None:
        reconstructed = try_reconstruct_grid_tables(
            slices=[_slice(1, 1, 1, 1)],
            parts=["# Not a table\n"],
        )

        self.assertIsNone(reconstructed)

    def test_reconstructs_when_a_blank_grid_tile_is_skipped(self) -> None:
        reconstructed = try_reconstruct_grid_tables(
            slices=[_slice(1, 1, 1, 2), _slice(1, 2, 1, 2)],
            parts=[
                "| A | B |\n| --- | --- |\n| 1 | 2 |\n",
                "",
            ],
        )

        self.assertEqual(
            reconstructed,
            "| A | B |\n| --- | --- |\n| 1 | 2 |\n",
        )

    def test_reconstructs_coordinate_bands_with_trailing_blank_rows(self) -> None:
        reconstructed = try_reconstruct_grid_table_bands_html(
            slices=[
                _slice(1, 1, 2, 2),
                _slice(1, 2, 2, 2),
                _slice(2, 1, 2, 2),
                _slice(2, 2, 2, 2),
            ],
            parts=[
                "| A |\n| --- |\n| 1 |\n| 2 |\n",
                "| B |\n| --- |\n| x |\n",
                "| A |\n| --- |\n| 3 |\n",
                "| B |\n| --- |\n| y |\n",
            ],
        )

        self.assertEqual(
            reconstructed,
            "<table>\n"
            "<tr><th>A</th><th>B</th></tr>\n"
            "<tr><td>1</td><td>x</td></tr>\n"
            "<tr><td>2</td><td></td></tr>\n"
            "<tr><td>3</td><td>y</td></tr>\n"
            "</table>\n",
        )

    def test_band_fallback_recovers_grouped_numeric_header_spans(self) -> None:
        reconstructed = try_reconstruct_grid_table_bands_html(
            slices=[
                _slice(1, 1, 3, 3),
                _slice(1, 2, 3, 3),
                _slice(1, 3, 3, 3),
                _slice(2, 1, 3, 3),
                _slice(2, 2, 3, 3),
                _slice(2, 3, 3, 3),
                _slice(3, 1, 3, 3),
                _slice(3, 2, 3, 3),
                _slice(3, 3, 3, 3),
            ],
            parts=[
                "<table><thead><tr><th rowspan='2'>Year</th><th>0</th><th>1</th></tr>"
                "</thead><tbody><tr><td>1</td><td>10</td><td>11</td></tr></tbody></table>",
                "<table><thead><tr><th colspan='2'>Age</th></tr>"
                "<tr><th>2</th><th>3</th></tr></thead>"
                "<tbody><tr><td>12</td><td>13</td></tr></tbody></table>",
                "<table><tr><td>4</td><td>5</td></tr>"
                "<tr><td>14</td><td>15</td></tr></table>",
                "<table><tr><td>2</td><td>20</td><td>21</td></tr>"
                "<tr><td>3</td><td>30</td><td>31</td></tr></table>",
                "<table><tr><td>22</td><td>23</td></tr>"
                "<tr><td>32</td><td>33</td></tr></table>",
                "<table><tr><td>24</td><td>25</td></tr>"
                "<tr><td>34</td><td>35</td></tr></table>",
                "<table><tr><td>4</td><td>40</td><td>41</td></tr>"
                "<tr><td>5</td><td>50</td><td>51</td></tr></table>",
                "<table><tr><td>42</td><td>43</td></tr>"
                "<tr><td>52</td><td>53</td></tr></table>",
                "<table><tr><td>44</td><td>45</td></tr>"
                "<tr><td>54</td><td>55</td></tr></table>",
            ],
        )

        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        self.assertIn(
            '<tr><th rowspan="2">Year</th><th colspan="6">Age</th></tr>',
            reconstructed,
        )
        self.assertIn(
            "<tr><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th></tr>",
            reconstructed,
        )

    def test_band_fallback_collapses_numeric_crop_seams_and_infers_group_heading(self) -> None:
        reconstructed = try_reconstruct_grid_table_bands_html(
            slices=[
                _slice(1, 1, 2, 3),
                _slice(1, 2, 2, 3),
                _slice(1, 3, 2, 3),
                _slice(2, 1, 2, 3),
                _slice(2, 2, 2, 3),
                _slice(2, 3, 2, 3),
            ],
            parts=[
                "# Policy\n## Cash Value\n### Contract Year\n"
                "<table><tr><td>Insured Age</td><td>26</td><td></td></tr>"
                "<tr><td>0</td><td>100</td><td>101</td></tr></table>",
                "<table><tr><td>27</td><td>28</td><td>29</td><td>30</td><td>31</td><td></td></tr>"
                "<tr><td>1.50</td><td>200</td><td>250</td><td>275</td><td>300</td><td>301</td></tr></table>",
                "<table><tr><td>32</td><td>33</td></tr>"
                "<tr><td>1.25</td><td>400</td></tr>"
                "<tr><td>ORPHAN</td><td>999</td></tr></table>",
                "<table><tr><td>1</td><td>110</td><td>111</td></tr>"
                "<tr><td>2</td><td>120</td><td>121</td></tr></table>",
                "<table><tr><td>1.60</td><td>210</td><td>260</td><td>285</td><td>310</td><td>311</td></tr>"
                "<tr><td>1.70</td><td>220</td><td>270</td><td>295</td><td>320</td><td>321</td></tr></table>",
                "<table><tr><td>1.35</td><td>410</td></tr>"
                "<tr><td>1.45</td><td>420</td></tr></table>",
            ],
        )

        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        self.assertEqual(reconstructed.count("<table>"), 1)
        self.assertIn(
            '<tr><th rowspan="2">Insured Age</th><th colspan="8">Contract Year</th></tr>',
            reconstructed,
        )
        self.assertIn("<td>101.50</td>", reconstructed)
        self.assertIn("<td>301.25</td>", reconstructed)
        self.assertNotIn("ORPHAN", reconstructed)

    def test_grouped_header_requires_a_consecutive_global_leaf_sequence(self) -> None:
        reconstructed = try_reconstruct_grid_table_bands_html(
            slices=[
                _slice(1, 1, 2, 2),
                _slice(1, 2, 2, 2),
                _slice(2, 1, 2, 2),
                _slice(2, 2, 2, 2),
            ],
            parts=[
                "<table><thead><tr><th rowspan='2'>Year</th><th>0</th></tr></thead>"
                "<tbody><tr><td>1</td><td>10</td></tr></tbody></table>",
                "<table><thead><tr><th colspan='3'>Age</th></tr>"
                "<tr><th>2</th><th>4</th><th>5</th></tr></thead>"
                "<tbody><tr><td>12</td><td>14</td><td>15</td></tr></tbody></table>",
                "<table><tr><td>2</td><td>20</td></tr>"
                "<tr><td>3</td><td>30</td></tr></table>",
                "<table><tr><td>22</td><td>24</td><td>25</td></tr>"
                "<tr><td>32</td><td>34</td><td>35</td></tr></table>",
            ],
        )

        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        self.assertNotIn('rowspan="2"', reconstructed)

    def test_band_fallback_removes_repeated_top_context_rows(self) -> None:
        reconstructed = try_reconstruct_grid_table_bands_html(
            slices=[
                _slice(1, 1, 2, 2),
                _slice(1, 2, 2, 2),
                _slice(2, 1, 2, 2),
                _slice(2, 2, 2, 2),
            ],
            parts=[
                "<table><tr><th>6</th><th>A</th></tr>"
                "<tr><td>7</td><td>B</td></tr>"
                "<tr><td>24</td><td>C</td></tr></table>",
                "<table><tr><th>6</th><th>X</th></tr>"
                "<tr><td>7</td><td>Y</td></tr>"
                "<tr><td>24</td><td>Z</td></tr></table>",
                "<table><tr><th>6</th><th>A</th></tr>"
                "<tr><td>7</td><td>B</td></tr>"
                "<tr><td>25</td><td>D</td></tr>"
                "<tr><td>26</td><td>E</td></tr></table>",
                "<table><tr><th>6</th><th>X</th></tr>"
                "<tr><td>7</td><td>Y</td></tr>"
                "<tr><td>25</td><td>W</td></tr>"
                "<tr><td>26</td><td>V</td></tr></table>",
            ],
        )

        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        self.assertEqual(reconstructed.count("<th>6</th>"), 1)
        self.assertIn("<tr><td>25</td><td>D</td><td>W</td></tr>", reconstructed)

    def test_band_fallback_keeps_discontinuous_logical_tables_separate(self) -> None:
        reconstructed = try_reconstruct_grid_table_bands_html(
            slices=[_slice(1, 1, 2, 1), _slice(2, 1, 2, 1)],
            parts=[
                "<table><tr><th>1</th><th>A</th></tr>"
                "<tr><td>20</td><td>B</td></tr></table>",
                "<table><tr><th>80</th><th>C</th></tr>"
                "<tr><td>90</td><td>D</td></tr></table>",
            ],
        )

        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        self.assertEqual(reconstructed.count("<table"), 2)

    def test_band_fallback_joins_on_a_later_age_column_and_pads_width(self) -> None:
        reconstructed = try_reconstruct_grid_table_bands_html(
            slices=[
                _slice(1, 1, 3, 1),
                _slice(2, 1, 3, 1),
                _slice(3, 1, 3, 1),
            ],
            parts=[
                "<table><tr><th>Plan</th><th>Term</th><th>Pay</th><th>Sex</th><th>Age</th></tr>"
                "<tr><td>55</td><td>105</td><td>3</td><td>F</td><td>20</td></tr></table>",
                "<table><tr><th>55</th><th>105</th><th>3</th><th>F</th><th>21</th><th>100</th></tr>"
                "<tr><td>55</td><td>105</td><td>3</td><td>F</td><td>22</td><td>110</td></tr></table>",
                "<table><tr><th>55</th><th>105</th><th>3</th><th>F</th><th>23</th></tr>"
                "<tr><td>55</td><td>105</td><td>3</td><td>F</td><td>24</td></tr></table>",
            ],
        )

        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        self.assertEqual(reconstructed.count("<table"), 1)
        self.assertIn(
            "<tr><td>55</td><td>105</td><td>3</td><td>F</td><td>20</td><td></td></tr>",
            reconstructed,
        )
        self.assertIn(
            "<tr><td>55</td><td>105</td><td>3</td><td>F</td><td>24</td><td></td></tr>",
            reconstructed,
        )

    def test_band_fallback_looks_past_one_noisy_boundary_row(self) -> None:
        reconstructed = try_reconstruct_grid_table_bands_html(
            slices=[_slice(1, 1, 2, 1), _slice(2, 1, 2, 1)],
            parts=[
                "<table><tr><th>Age</th><th>Sex</th><th>Term</th><th>Plan</th></tr>"
                "<tr><td>37</td><td>F</td><td>10</td><td>A</td></tr>"
                "<tr><td>38</td><td>F</td><td>10</td><td>A</td></tr>"
                "<tr><td>30</td><td>F</td><td>10</td><td>A</td></tr></table>",
                "<table><tr><td>39</td><td>F</td><td>10</td><td>A</td></tr>"
                "<tr><td>40</td><td>F</td><td>10</td><td>A</td></tr></table>",
            ],
        )

        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        self.assertEqual(reconstructed.count("<table"), 1)

    def test_band_fallback_joins_a_monotonic_duplicate_boundary_row(self) -> None:
        reconstructed = try_reconstruct_grid_table_bands_html(
            slices=[
                _slice(1, 1, 3, 2),
                _slice(1, 2, 3, 2),
                _slice(2, 1, 3, 2),
                _slice(2, 2, 3, 2),
                _slice(3, 1, 3, 2),
                _slice(3, 2, 3, 2),
            ],
            parts=[
                "<table><tr><th></th></tr>"
                "<tr><td>23</td></tr><tr><td>24</td></tr></table>",
                "<table><tr><th>A</th></tr>"
                "<tr><td>100</td></tr><tr><td>200</td></tr></table>",
                "<table><tr><td>25</td></tr>"
                "<tr><td>54</td></tr><tr><td>55</td></tr></table>",
                "<table><tr><td>300</td><td>301</td></tr>"
                "<tr><td>400</td><td>401</td></tr>"
                "<tr><td>500</td><td>501</td></tr>"
                "<tr><td></td><td></td></tr></table>",
                "<table><tr><td>55</td></tr><tr><td>56</td></tr></table>",
                "<table><tr><td>501</td></tr><tr><td>600</td></tr></table>",
            ],
        )

        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        self.assertEqual(reconstructed.count("<table"), 1)
        self.assertEqual(reconstructed.count("<td>55</td>"), 1)
        self.assertNotIn("<tr><td></td><td></td></tr>", reconstructed)

    def test_band_fallback_drops_duplicate_and_reversed_band_headers(self) -> None:
        reconstructed = try_reconstruct_grid_table_bands_html(
            slices=[
                _slice(1, 1, 4, 1),
                _slice(2, 1, 4, 1),
                _slice(3, 1, 4, 1),
                _slice(4, 1, 4, 1),
            ],
            parts=[
                "<table><tr><th>Age</th><th>Value</th></tr>"
                "<tr><td>0</td><td>A</td></tr></table>",
                "<table><tr><td>0</td><td>A2</td></tr>"
                "<tr><td>1</td><td>B</td></tr>"
                "<tr><td>2</td><td>C</td></tr></table>",
                "<table><tr><td>4</td><td>NOISY</td></tr>"
                "<tr><td>3</td><td>D</td></tr>"
                "<tr><td>4</td><td>E</td></tr></table>",
                "<table><tr><td>5</td><td>F</td></tr>"
                "<tr><td>6</td><td>G</td></tr></table>",
            ],
        )

        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        self.assertEqual(reconstructed.count("<table>"), 1)
        self.assertEqual(reconstructed.count("<td>0</td>"), 1)
        self.assertNotIn("NOISY", reconstructed)
        self.assertIn("<tr><td>6</td><td>G</td></tr>", reconstructed)

    def test_duplicate_boundary_alone_does_not_join_two_equal_width_tables(self) -> None:
        reconstructed = try_reconstruct_grid_table_bands_html(
            slices=[_slice(1, 1, 2, 1), _slice(2, 1, 2, 1)],
            parts=[
                "<table><tr><td>54</td><td>A</td></tr>"
                "<tr><td>55</td><td>B</td></tr></table>",
                "<table><tr><td>55</td><td>C</td></tr>"
                "<tr><td>56</td><td>D</td></tr></table>",
            ],
        )

        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        self.assertEqual(reconstructed.count("<table"), 2)

    def test_band_fallback_drops_an_unkeyed_trailing_crop_fragment(self) -> None:
        def wide_row(first: str) -> str:
            cells = (first,) + ("",) * 7
            return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"

        reconstructed = try_reconstruct_grid_table_bands_html(
            slices=[
                _slice(1, 1, 2, 2),
                _slice(1, 2, 2, 2),
                _slice(2, 1, 2, 2),
                _slice(2, 2, 2, 2),
            ],
            parts=[
                f"<table>{wide_row('53')}{wide_row('54')}{wide_row('55')}</table>",
                "<table><tr><td>100</td></tr>"
                "<tr><td>150</td></tr><tr><td>200</td></tr>"
                "<tr><td>ORPHAN</td></tr></table>",
                f"<table>{wide_row('55')}{wide_row('56')}</table>",
                "<table><tr><td>201</td></tr><tr><td>300</td></tr></table>",
            ],
        )

        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        self.assertEqual(reconstructed.count("<table"), 2)
        self.assertNotIn("ORPHAN", reconstructed)

    def test_band_fallback_rejects_windowed_numbers_without_stable_context(self) -> None:
        reconstructed = try_reconstruct_grid_table_bands_html(
            slices=[_slice(1, 1, 2, 1), _slice(2, 1, 2, 1)],
            parts=[
                "<table><tr><th>Age</th><th>Kind</th><th>Term</th></tr>"
                "<tr><td>20</td><td>A</td><td>X</td></tr>"
                "<tr><td>99</td><td>A</td><td>X</td></tr></table>",
                "<table><tr><td>21</td><td>B</td><td>Y</td></tr>"
                "<tr><td>22</td><td>B</td><td>Y</td></tr></table>",
            ],
        )

        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        self.assertEqual(reconstructed.count("<table"), 2)

    def test_band_fallback_keeps_a_failed_boundary_as_a_group_break(self) -> None:
        reconstructed = try_reconstruct_grid_table_bands_html(
            slices=[
                _slice(1, 1, 3, 1),
                _slice(2, 1, 3, 1),
                _slice(3, 1, 3, 1),
            ],
            parts=[
                "<table><tr><th>Age</th><th>Sex</th><th>Term</th><th>Plan</th></tr>"
                "<tr><td>20</td><td>F</td><td>10</td><td>A</td></tr></table>",
                "<table><tr><td>21</td><td>F</td><td>10</td><td>A</td></tr>"
                "<tr><td>22</td><td>F</td><td>10</td><td>A</td></tr></table>",
                "<table><tr><th>80</th><th>M</th><th>20</th><th>B</th></tr>"
                "<tr><td>90</td><td>M</td><td>20</td><td>B</td></tr></table>",
            ],
        )

        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        self.assertEqual(reconstructed.count("<table"), 2)

    def test_band_fallback_attaches_an_explicit_header_only_first_band(self) -> None:
        reconstructed = try_reconstruct_grid_table_bands_html(
            slices=[
                _slice(1, 1, 3, 1),
                _slice(2, 1, 3, 1),
                _slice(3, 1, 3, 1),
            ],
            parts=[
                "<table><tr><th>Year</th><th>68</th><th>69</th></tr>"
                "<tr><td>Rate</td><td>1.0</td><td>2.0</td></tr></table>",
                "<table><tr><td>6</td><td>950</td></tr>"
                "<tr><td>17</td><td>970</td></tr></table>",
                "<table><tr><td>18</td><td>971</td></tr>"
                "<tr><td>29</td><td>990</td></tr></table>",
            ],
        )

        self.assertIsNotNone(reconstructed)
        assert reconstructed is not None
        self.assertEqual(reconstructed.count("<table"), 1)


if __name__ == "__main__":
    unittest.main()
