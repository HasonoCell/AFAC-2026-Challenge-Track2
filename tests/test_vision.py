from __future__ import annotations

import unittest

from afac_pipeline.tables import parse_table
from afac_pipeline.vision import (
    VisionObservation,
    _contiguous_row_centers,
    _merge_split_numeric_fragments,
    _numeric_value,
    _normalize_numeric_text,
    render_observations_in_reading_order,
    reconstruct_numeric_matrix_from_observations,
)


class VisionMatrixTest(unittest.TestCase):
    def test_renders_prose_boxes_by_geometry_and_deduplicates_overlap(self) -> None:
        observations = [
            VisionObservation(80, 100, 40, 12, 0.99, "正文"),
            # The same box seen in the next overlapping image slice.
            VisionObservation(80.5, 101, 40, 12, 0.99, "正文"),
            # A neighbouring tile re-recognized the same long line with a
            # punctuation variant; it must replace rather than concatenate.
            VisionObservation(100, 120, 120, 12, 0.80, "旧的第二行"),
            VisionObservation(101, 121, 120, 12, 0.99, "第二行"),
            VisionObservation(80, 220, 80, 12, 0.99, "新段落"),
        ]

        self.assertEqual(
            render_observations_in_reading_order(observations),
            "正文\n\n第二行\n\n新段落",
        )

    def test_normalizes_ocr_spacing_after_numeric_separators(self) -> None:
        self.assertEqual(_normalize_numeric_text(" 176. 61 "), "176.61")
        self.assertEqual(_normalize_numeric_text("1, 234. 50"), "1,234.50")
        self.assertEqual(_numeric_value("176. 61"), "176.61")
        self.assertIsNone(_numeric_value("176 61"))

    def test_merges_many_split_numeric_fragments_without_cross_pair_reuse(self) -> None:
        # Large local-OCR tables can place thousands of observations in two
        # neighbouring columns. The matcher must use a narrow y window rather
        # than scanning the complete right column for every left fragment.
        left = [
            VisionObservation(10, float(index * 10), 4, 8, 0.9, "12")
            for index in range(400)
        ]
        right = [
            VisionObservation(20, float(index * 10), 4, 8, 0.9, ".34")
            for index in range(400)
        ]
        # Two distant columns establish the normal table pitch. The first
        # close pair is then an eligible split-cell seam.
        anchor = [
            VisionObservation(110, 0, 4, 8, 0.9, "7"),
            VisionObservation(200, 0, 4, 8, 0.9, "8"),
        ]

        merged = _merge_split_numeric_fragments(left + right + anchor)

        self.assertEqual(len(merged), 402)
        self.assertEqual(sum(item.text == "12.34" for item in merged), 400)

    def test_row_lattice_tracks_bounded_local_drift_without_reaching_footer(self) -> None:
        candidates = [(100 + index * 10.1, 20) for index in range(500)]
        candidates.append((5_500, 20))

        centers = _contiguous_row_centers(
            candidates=candidates,
            first=100,
            step=10,
        )

        self.assertEqual(len(centers), 500)
        self.assertAlmostEqual(centers[-1], 100 + 499 * 10.1)

    def test_reconstructs_semantic_numeric_matrix_from_coordinates(self) -> None:
        observations = [
            _observation(100, 40, "现金价值表"),
            _observation(100, 100, "保单年度末\\年龄"),
            _observation(200, 100, "10"),
            _observation(300, 100, "11"),
            _observation(400, 100, "12"),
        ]
        expected_rows = []
        for row in range(5):
            values = (
                str(row),
                f"{1_000 + row:,}",
                f"{2_000 + row:,}",
                f"{3_000 + row:,}",
            )
            expected_rows.append(values)
            for column, value in enumerate(values):
                observations.append(
                    _observation(100 + column * 100, 200 + row * 50, value)
                )

        result = reconstruct_numeric_matrix_from_observations(observations)

        self.assertIsNotNone(result)
        assert result is not None
        table = parse_table(result.markdown)
        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.header, ("保单年度末\\年龄", "10", "11", "12"))
        self.assertEqual(table.rows, tuple(expected_rows))
        self.assertEqual(result.coverage, 1.0)
        self.assertTrue(result.markdown.startswith("现金价值表\n\n"))

    def test_reconstructs_matrix_with_stacked_corner_header(self) -> None:
        observations = [
            _observation(100, 100, "保单年度末"),
            _observation(100, 130, "投保年龄"),
        ]
        observations.extend(
            _observation(200 + column * 100, 100, str(10 + column))
            for column in range(4)
        )
        expected_rows = []
        for row in range(5):
            values = (str(row), *(str(1_000 + row * 10 + column) for column in range(4)))
            expected_rows.append(values)
            observations.extend(
                _observation(100 + column * 100, 200 + row * 50, value)
                for column, value in enumerate(values)
            )

        result = reconstruct_numeric_matrix_from_observations(observations)

        self.assertIsNotNone(result)
        assert result is not None
        table = parse_table(result.markdown)
        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.header, ("保单年度末\\投保年龄", "10", "11", "12", "13"))
        self.assertEqual(table.rows, tuple(expected_rows))

    def test_reconstructs_matrix_with_diagonal_corner_and_annual_axis(self) -> None:
        """A wider corner is valid only when its annual sequence is visible."""

        observations = [
            _observation(180, 130, "投保年龄"),
            _observation(100, 100, "保单年度末"),
        ]
        observations.extend(
            _observation(200 + column * 100, 100, str(column + 1))
            for column in range(5)
        )
        expected_rows = []
        for row in range(8):
            values = (str(row), *(str(1_000 + row * 10 + column) for column in range(5)))
            expected_rows.append(values)
            observations.extend(
                _observation(100 + column * 100, 200 + row * 50, value)
                for column, value in enumerate(values)
            )

        result = reconstruct_numeric_matrix_from_observations(observations)

        self.assertIsNotNone(result)
        assert result is not None
        table = parse_table(result.markdown)
        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.header, ("保单年度末\\投保年龄", "1", "2", "3", "4", "5"))
        self.assertEqual(table.rows, tuple(expected_rows))

    def test_reconstructs_right_aligned_numeric_columns(self) -> None:
        """Varying value widths must not split one right-aligned column."""

        observations = [
            VisionObservation(95, 100, 10, 12, 1.0, "保单年度\\投保年龄"),
        ]
        for column in range(5):
            right = 200 + column * 100
            observations.append(
                VisionObservation(right - 6, 100, 12, 12, 1.0, str(column)))
        expected_rows = []
        for row in range(8):
            values = (str(row + 1), *(str(1000 + row * 10 + column) for column in range(5)))
            expected_rows.append(values)
            observations.append(VisionObservation(95, 200 + row * 40, 10, 12, 1.0, values[0]))
            for column, value in enumerate(values[1:]):
                right = 200 + column * 100
                width = 18 if (row + column) % 2 == 0 else 46
                observations.append(
                    VisionObservation(
                        right - width / 2,
                        200 + row * 40,
                        width,
                        12,
                        1.0,
                        value,
                    )
                )

        result = reconstruct_numeric_matrix_from_observations(observations)

        self.assertIsNotNone(result)
        assert result is not None
        table = parse_table(result.markdown)
        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.header, ("保单年度\\投保年龄", "0", "1", "2", "3", "4"))
        self.assertEqual(table.rows, tuple(expected_rows))

    def test_rejects_matrix_without_semantic_header(self) -> None:
        observations = [
            _observation(100 + column * 100, 100 + row * 50, str(row * 4 + column))
            for row in range(5)
            for column in range(4)
        ]

        self.assertIsNone(reconstruct_numeric_matrix_from_observations(observations))

    def test_reconstructs_repeated_matrices_with_shared_row_lattice(self) -> None:
        observations = [
            _observation(100, 20, "现金价值表"),
            _observation(100, 60, "男性"),
            _observation(180, 60, "一次性交付"),
        ]
        for table_index, header_y in enumerate((100, 450, 800)):
            if table_index > 0:
                observations.extend(
                    [
                        _observation(100, header_y - 40, "女性"),
                        _observation(180, header_y - 40, f"{table_index + 2}年交"),
                    ]
                )
            observations.extend(
                [
                    _observation(100, header_y, "年度/年龄"),
                    _observation(200, header_y, "55"),
                    _observation(300, header_y, "56"),
                    _observation(400, header_y, "57"),
                ]
            )
            for row in range(5):
                values = (str(row + 1), str(100 + row), str(200 + row), str(300 + row))
                for column, value in enumerate(values):
                    # The second matrix deliberately loses two row labels. Its
                    # row lattice is recovered only from cross-table consensus.
                    if table_index == 2 and column == 0 and row >= 3:
                        continue
                    observations.append(
                        _observation(
                            100 + column * 100,
                            header_y + 50 + row * 50,
                            value,
                        )
                    )

        result = reconstruct_numeric_matrix_from_observations(observations)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.table_count, 3)
        self.assertEqual(result.row_starts, (1, 1, 1))
        self.assertEqual(result.markdown.count("<table>"), 3)
        self.assertIn("现金价值表\n男性 一次性交付", result.markdown)
        self.assertIn("女性 3年交", result.markdown)
        self.assertIn("女性 4年交", result.markdown)

    def test_rejects_descriptive_sentence_as_matrix_header(self) -> None:
        observations = [
            _observation(100, 100, "说明：下表年龄代表投保年龄、年度代表保单年度末")
        ] + [
            _observation(100 + column * 100, 200 + row * 50, str(row * 4 + column))
            for row in range(5)
            for column in range(4)
        ]

        self.assertIsNone(reconstruct_numeric_matrix_from_observations(observations))

    def test_reconstructs_wide_year_matrix_from_triangular_resets(self) -> None:
        year_count = 12
        observations = [
            _observation(80, 100, "保险期间 交费期间 投保年龄"),
            _observation(260, 100, "性别"),
            _observation(700, 100, " ".join(f"第{index}保单年度" for index in range(1, 6))),
        ]
        expected_rows = []
        group_specs = ((1, "男"), (1, "女"), (3, "男"), (3, "女"))
        row_index = 0
        for payment, gender in group_specs:
            for age in range(8):
                y = 150 + row_index * 50
                observations.extend(
                    [
                        _observation(50, y, "终身"),
                        _observation(100, y, str(payment)),
                        _observation(200, y, str(age)),
                        _observation(300, y, gender),
                    ]
                )
                values = []
                for year in range(1, year_count + 1):
                    value = str(payment * 10_000 + age * 100 + year)
                    if year <= year_count - age:
                        observations.append(_observation(300 + year * 100, y, value))
                        values.append(value)
                    else:
                        values.append("")
                expected_rows.append(
                    ("终身", str(payment), str(age), gender, *values)
                )
                row_index += 1

        # A sparse OCR artifact close to a real year column must not create an
        # extra output column.  Its support is much weaker than the adjacent
        # triangular lattice column.
        for artifact_row in range(5):
            observations.append(
                _observation(825, 150 + artifact_row * 50, "8")
            )

        result = reconstruct_numeric_matrix_from_observations(observations)

        self.assertIsNotNone(result)
        assert result is not None
        table = parse_table(result.markdown)
        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(result.table_count, 1)
        self.assertEqual(result.rows, 32)
        self.assertEqual(result.cols, 16)
        self.assertEqual(table.header[:4], ("保险期间", "交费期间", "投保年龄", "性别"))
        self.assertEqual(table.header[-1], "第12保单年度")
        self.assertEqual(table.rows, tuple(expected_rows))

    def test_reconstructs_headerless_wide_matrix_with_strong_body_evidence(self) -> None:
        """A missed title alone must not discard an otherwise proven lattice."""

        year_count = 12
        observations = []
        expected_rows = []
        row_index = 0
        for payment, gender in ((1, "男"), (1, "女"), (3, "男"), (3, "女")):
            for age in range(8):
                y = 150 + row_index * 50
                observations.extend(
                    [
                        _observation(50, y, "终身"),
                        _observation(100, y, str(payment)),
                        _observation(200, y, str(age)),
                        _observation(300, y, gender),
                    ]
                )
                values = []
                for year in range(1, year_count + 1):
                    value = str(payment * 10_000 + age * 100 + year)
                    if year <= year_count - age:
                        observations.append(_observation(300 + year * 100, y, value))
                        values.append(value)
                    else:
                        values.append("")
                expected_rows.append(
                    ("终身", str(payment), str(age), gender, *values)
                )
                row_index += 1

        result = reconstruct_numeric_matrix_from_observations(observations)

        self.assertIsNotNone(result)
        assert result is not None
        table = parse_table(result.markdown)
        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(result.header_sequence_inliers, 0)
        self.assertEqual(table.header[:4], ("保险期间", "交费期间", "投保年龄", "性别"))
        self.assertEqual(table.rows, tuple(expected_rows))

    def test_reconstructs_headerless_wide_matrix_with_interleaved_genders(self) -> None:
        """Age may advance once per male/female pair rather than per row."""

        year_count = 12
        observations = []
        expected_rows = []
        row_index = 0
        for payment in (5, 10, 15, 20):
            for age in range(8):
                for gender in ("女", "男"):
                    y = 150 + row_index * 50
                    observations.extend(
                        [
                            _observation(50, y, "终身"),
                            _observation(100, y, str(payment)),
                            _observation(200, y, str(age)),
                            _observation(300, y, gender),
                        ]
                    )
                    values = []
                    for year in range(1, year_count + 1):
                        value = str(payment * 10_000 + age * 100 + year)
                        if year <= year_count - age:
                            observations.append(
                                _observation(300 + year * 100, y, value)
                            )
                            values.append(value)
                        else:
                            values.append("")
                    expected_rows.append(
                        ("终身", str(payment), str(age), gender, *values)
                    )
                    row_index += 1

        result = reconstruct_numeric_matrix_from_observations(observations)

        self.assertIsNotNone(result)
        assert result is not None
        table = parse_table(result.markdown)
        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.rows, tuple(expected_rows))

    def test_reconstructs_wide_matrix_when_each_group_has_its_own_age_start(self) -> None:
        year_count = 12
        observations = [
            _observation(80, 100, "保险期间 交费期间 投保年龄"),
            _observation(260, 100, "性别"),
            _observation(700, 100, " ".join(f"第{index}保单年度" for index in range(1, 6))),
        ]
        expected_rows = []
        # A document crop can begin after the first age row.  The old route
        # hard-coded age 0 at every triangular reset and rejected this valid
        # geometry even though the local age sequence is exact.
        group_specs = (
            (10, "男", 2, 8),
            (10, "女", 0, 8),
            (20, "男", 0, 8),
            (20, "女", 0, 8),
        )
        row_index = 0
        for payment, gender, age_start, row_count in group_specs:
            for local_row in range(row_count):
                age = age_start + local_row
                y = 150 + row_index * 50
                observations.extend(
                    [
                        _observation(50, y, "终身"),
                        _observation(100, y, str(payment)),
                        _observation(200, y, str(age)),
                        _observation(300, y, gender),
                    ]
                )
                values = []
                for year in range(1, year_count + 1):
                    value = str(payment * 10_000 + age * 100 + year)
                    if year <= year_count - age:
                        observations.append(_observation(300 + year * 100, y, value))
                        values.append(value)
                    else:
                        values.append("")
                expected_rows.append(("终身", str(payment), str(age), gender, *values))
                row_index += 1

        result = reconstruct_numeric_matrix_from_observations(observations)

        self.assertIsNotNone(result)
        assert result is not None
        table = parse_table(result.markdown)
        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.rows, tuple(expected_rows))

    def test_keeps_first_standard_wide_row_close_to_single_row_header(self) -> None:
        """A one-row annual header must not consume the first age-0 row."""

        year_count = 12
        observations = [
            _observation(80, 100, "保险期间 交费期间 投保年龄"),
            _observation(260, 100, "性别"),
            _observation(700, 100, " ".join(f"第{index}保单年度" for index in range(1, 6))),
        ]
        expected_rows = []
        row_index = 0
        for payment, gender in ((1, "男"), (1, "女"), (3, "男"), (3, "女")):
            for age in range(8):
                # The first body row is only 30 px below a 10 px header.  A
                # double-header exclusion band would incorrectly discard it.
                y = 130 + row_index * 30
                observations.extend(
                    [
                        _observation(50, y, "终身"),
                        _observation(100, y, str(payment)),
                        _observation(200, y, str(age)),
                        _observation(300, y, gender),
                    ]
                )
                values = []
                for year in range(1, year_count + 1):
                    value = str(payment * 10_000 + age * 100 + year)
                    if year <= year_count - age:
                        observations.append(_observation(300 + year * 100, y, value))
                        values.append(value)
                    else:
                        values.append("")
                expected_rows.append(("终身", str(payment), str(age), gender, *values))
                row_index += 1

        result = reconstruct_numeric_matrix_from_observations(observations)

        self.assertIsNotNone(result)
        assert result is not None
        table = parse_table(result.markdown)
        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.rows, tuple(expected_rows))

    def test_preserves_header_proven_short_annual_tail_without_values(self) -> None:
        """Do not reject a valid table merely because its sparse tail vanished."""

        observations = [
            _observation(80, 100, "保险期间 交费期间 投保年龄"),
            _observation(260, 100, "性别"),
            _observation(
                700,
                100,
                " ".join(f"第{index}保单年度" for index in range(1, 13)),
            ),
        ]
        row_index = 0
        for payment, gender in ((1, "男"), (1, "女"), (3, "男"), (3, "女")):
            for age in range(8):
                y = 150 + row_index * 50
                observations.extend(
                    [
                        _observation(50, y, "终身"),
                        _observation(100, y, str(payment)),
                        _observation(200, y, str(age)),
                        _observation(300, y, gender),
                    ]
                )
                # The last two years have no OCR body observations at all,
                # but the header proves that they exist as blank tail cells.
                for year in range(1, min(10, 12 - age) + 1):
                    observations.append(
                        _observation(300 + year * 100, y, str(payment * 10_000 + age * 100 + year))
                    )
                row_index += 1

        result = reconstruct_numeric_matrix_from_observations(observations)

        self.assertIsNotNone(result)
        assert result is not None
        table = parse_table(result.markdown)
        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.header[-1], "第12保单年度")
        self.assertEqual(len(table.header), 16)
        self.assertTrue(all(not row[-1] for row in table.rows))

    def test_reconstructs_three_metadata_column_wide_matrix(self) -> None:
        """Do not require an insurance-period column for annual cash tables."""

        year_count = 12
        observations = [
            _observation(300, 100, "投保年龄 保单年度末"),
            _observation(100, 120, "交费期间"),
            _observation(200, 120, "性别"),
        ]
        observations.extend(
            _observation(300 + (year + 1) * 100, 120, str(year))
            for year in range(1, year_count + 1)
        )
        expected_rows = []
        for age in range(20):
            y = 180 + age * 50
            observations.extend(
                [
                    _observation(100, y, "5年"),
                    _observation(200, y, "男"),
                    _observation(300, y, str(age)),
                ]
            )
            values = tuple(str(1_000 + age * 100 + year) for year in range(1, year_count + 1))
            for year, value in enumerate(values, start=1):
                x = 300 + (year + 1) * 100
                if year == 6:
                    # Simulate a currency token split by a vertical tile seam.
                    observations.extend(
                        [
                            _observation(x - 15, y, value[:2]),
                            _observation(x + 15, y, value[2:]),
                        ]
                    )
                else:
                    observations.append(_observation(x, y, value))
            expected_rows.append(("5年", "男", str(age), *values))

        result = reconstruct_numeric_matrix_from_observations(observations)

        self.assertIsNotNone(result)
        assert result is not None
        table = parse_table(result.markdown)
        self.assertIsNotNone(table)
        assert table is not None
        self.assertIn('<th rowspan="2">交费期间</th>', result.markdown)
        self.assertIn('<th colspan="12">保单年度末</th>', result.markdown)
        self.assertIn("<tr><td>1</td><td>2</td>", result.markdown)
        self.assertEqual(table.rows[0][:3], ("交费期间", "性别", "投保年龄（周岁）"))
        self.assertEqual(table.rows[1:], tuple(expected_rows))

    def test_reconstructs_wide_matrix_with_two_text_metadata_columns(self) -> None:
        """Insurance and payment periods can both be textual body columns."""

        year_count = 12
        observations = [
            _observation(80, 100, "保险期间 交费期间 年龄"),
            _observation(300, 100, "性别"),
            _observation(700, 100, " ".join(f"第{index}保单年度" for index in range(1, 6))),
        ]
        expected_rows = []
        for age in range(20):
            y = 150 + age * 50
            observations.extend(
                [
                    _observation(50, y, "终身"),
                    _observation(120, y, "趸交"),
                    _observation(220, y, str(age)),
                    _observation(300, y, "男"),
                ]
            )
            values = tuple(str(1_000 + age * 100 + year) for year in range(1, year_count + 1))
            observations.extend(
                _observation(300 + (year + 1) * 100, y, value)
                for year, value in enumerate(values, start=1)
            )
            expected_rows.append(("终身", "趸交", str(age), "男", *values))

        result = reconstruct_numeric_matrix_from_observations(observations)

        self.assertIsNotNone(result)
        assert result is not None
        table = parse_table(result.markdown)
        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.header[:4], ("保险期间", "交费期间", "投保年龄", "性别"))
        self.assertEqual(table.rows, tuple(expected_rows))

    def test_adapts_text_metadata_columns_to_visible_header_order(self) -> None:
        """The four metadata fields are not printed in one universal order."""

        observations = [
            _observation(100, 100, "交费期间"),
            _observation(200, 100, "保险期间"),
            _observation(300, 100, "性别"),
            _observation(400, 100, "投保年龄"),
            _observation(700, 100, "第1保单年度 第2保单年度 第3保单年度 第4保单年度 第5保单年度"),
        ]
        expected_rows = []
        for age in range(20):
            y = 150 + age * 50
            values = tuple(str(1_000 + age * 10 + year) for year in range(1, 6))
            observations.extend(
                [
                    _observation(100, y, "5年"),
                    _observation(200, y, "至50周岁"),
                    _observation(300, y, "男"),
                    _observation(400, y, str(age)),
                ]
            )
            observations.extend(
                _observation(400 + (year + 1) * 100, y, value)
                for year, value in enumerate(values, start=1)
            )
            expected_rows.append(("5年", "至50周岁", "男", str(age), *values))

        result = reconstruct_numeric_matrix_from_observations(observations)

        self.assertIsNotNone(result)
        assert result is not None
        table = parse_table(result.markdown)
        self.assertIsNotNone(table)
        assert table is not None
        self.assertEqual(table.header[:4], ("交费期间", "保险期间", "性别", "投保年龄"))
        self.assertEqual(table.rows, tuple(expected_rows))

    def test_reconstructs_headerless_text_metadata_continuation(self) -> None:
        """A continuation may restart ages when visible metadata changes."""

        observations = []
        expected = []
        for group_index, (term, gender, age_start) in enumerate(
            (("5年", "女", 57), ("6年", "男", 0))
        ):
            for offset in range(10):
                row = group_index * 10 + offset
                y = 100 + row * 50
                age = age_start + offset
                values = tuple(str(1_000 + group_index * 100 + age * 10 + year) for year in range(1, 7))
                observations.extend(
                    [
                        _observation(100, y, term),
                        _observation(200, y, "趸交"),
                        _observation(300, y, str(age)),
                        _observation(400, y, gender),
                    ]
                )
                observations.extend(
                    _observation(400 + (year + 1) * 100, y, value)
                    for year, value in enumerate(values, start=1)
                )
                expected.append((term, "趸交", str(age), gender, *values))

        result = reconstruct_numeric_matrix_from_observations(observations)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.row_starts, (0, 10))
        table = parse_table(result.markdown)
        self.assertIsNotNone(table)
        assert table is not None
        self.assertFalse(table.header_is_explicit)
        self.assertEqual(table.header, expected[0])
        self.assertEqual(table.rows, tuple(expected[1:]))


def _observation(x: float, y: float, text: str) -> VisionObservation:
    return VisionObservation(
        x=x,
        y=y,
        width=30,
        height=10,
        confidence=1.0,
        text=text,
    )


if __name__ == "__main__":
    unittest.main()
