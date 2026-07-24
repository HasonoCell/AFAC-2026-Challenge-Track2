from __future__ import annotations

import csv
import io
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from afac_pipeline.baseline import (
    BaselineConfig,
    _RouteFailure,
    _call_coverage_tile,
    _call_table_content_grid_record,
    _baseline_cache_namespace,
    _cache_path,
    _call_table_repair_slice,
    _coverage_candidate_issue,
    _is_refinable_table_tile_failure,
    _is_truncation_error,
    _maybe_local_long_text_repair,
    _maybe_local_matrix_repair,
    _maybe_repair_short_table,
    _local_matrix_repair_is_better,
    _largest_pipe_numeric_sequence_gap,
    _merge_partial_sequence_row,
    _merge_previous_sequence_table_cells,
    _preserve_pipe_table_context,
    _repair_is_better,
    _fit_grid_to_call_budget,
    _local_matrix_geometry_config,
    _table_repair_grid,
    _table_repair_trigger_chars,
    _table_repair_failed_part_budget,
    _should_use_table_coverage,
    _should_attempt_local_matrix_repair,
    _table_candidate_issue,
    _try_long_slice_fallback,
    rebuild_cached_local_matrix_repairs,
    rebuild_local_matrix_repairs,
    run_baseline_submission,
)
from afac_pipeline.api import FinixDocError
from afac_pipeline.datasets import ImageRecord
from afac_pipeline.images import make_content_grid_slices, make_grid_slices, profile_image
from afac_pipeline.vision import VisionMatrixResult, VisionObservation


class _FakeClient:
    def __init__(
        self,
        fail_first: bool = False,
        malformed_first: bool = False,
        structurally_malformed_first: bool = False,
        short_first: bool = False,
        tabular_text_first: bool = False,
    ) -> None:
        self.calls: list[str] = []
        self.fail_first = fail_first
        self.malformed_first = malformed_first
        self.structurally_malformed_first = structurally_malformed_first
        self.short_first = short_first
        self.tabular_text_first = tabular_text_first

    def call_with_file(self, file_name: str, file_bytes: bytes, **kwargs: object) -> str:
        self.calls.append(file_name)
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("first crop failed")
        if self.malformed_first and len(self.calls) == 1:
            return "<table>"
        if self.structurally_malformed_first and len(self.calls) == 1:
            return "<table><tr><td>broken</td></table>"
        if self.short_first and len(self.calls) == 1:
            return "too short"
        if self.tabular_text_first and len(self.calls) == 1:
            return (
                "年度 年龄 现金价值\n"
                "1 59 880.50\n"
                "2 60 900.94\n"
                "3 61 921.48\n"
                "4 62 942.13\n"
            )
        if "_content_" in file_name:
            return "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
        if "_part" in file_name:
            return f"# {file_name}\n\nslice text\n"
        return "<table><tr><td>ok</td></tr></table>\n"


class BaselineSubmissionTest(unittest.TestCase):
    def test_local_matrix_rebuild_uses_submission_as_the_no_api_baseline(self) -> None:
        record = _inked_record(
            "sample.jpg",
            source="memory",
            size=(120, 120),
            ink_box=(10, 10, 110, 110),
        )
        recovered = "<table><tr><td>recovered</td></tr></table>\n"
        vision_result = VisionMatrixResult(
            markdown=recovered,
            rows=1,
            cols=1,
            populated_cells=1,
            total_cells=1,
            header_sequence_inliers=1,
            row_sequence_inliers=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_csv = root / "base.csv"
            output_csv = root / "repair.csv"
            with base_csv.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["file_name", "ground_truth"])
                writer.writeheader()
                writer.writerow({"file_name": record.file_name, "ground_truth": "short"})
            config = BaselineConfig(
                output_csv=output_csv,
                cache_dir=root / "fresh-local-cache",
                table_local_ocr_backend="rapidocr",
                table_local_ocr_min_pixels=0,
                table_repair_min_gain=0,
                table_repair_content_scale=1.0,
                table_repair_content_padding=0,
            )
            with patch(
                "afac_pipeline.baseline.run_local_numeric_matrix_ocr",
                return_value=vision_result,
            ) as run_local:
                result = rebuild_local_matrix_repairs(
                    records=[record],
                    base_csv=base_csv,
                    output_csv=output_csv,
                    config=config,
                )

            self.assertEqual(result.selected_file_names, ("sample.jpg",))
            run_local.assert_called_once()
            with output_csv.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows, [{"file_name": "sample.jpg", "ground_truth": recovered}])

    def test_cached_local_matrix_rebuild_uses_complete_tsv_cache_without_api(self) -> None:
        record = _inked_record(
            "sample.jpg",
            source="memory",
            size=(120, 120),
            ink_box=(10, 10, 110, 110),
        )
        markdown = "<table><tr><td>recovered</td></tr></table>\n"
        vision_result = VisionMatrixResult(
            markdown=markdown,
            rows=1,
            cols=1,
            populated_cells=1,
            total_cells=1,
            header_sequence_inliers=1,
            row_sequence_inliers=1,
        )
        config = BaselineConfig(
            output_csv=Path("unused.csv"),
            cache_dir=Path("unused-cache"),
            table_local_ocr_backend="rapidocr",
            table_local_ocr_min_pixels=0,
            table_repair_min_gain=0,
            table_repair_content_scale=1.0,
            table_repair_content_padding=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_csv = root / "base.csv"
            output_csv = root / "repair.csv"
            cache_root = root / "local-cache"
            with base_csv.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["file_name", "ground_truth"])
                writer.writeheader()
                writer.writerow({"file_name": record.file_name, "ground_truth": "short"})
            slices = make_content_grid_slices(
                file_name=record.file_name,
                image_bytes=record.read_bytes(),
                rows=4,
                cols=4,
                threshold=config.table_repair_content_threshold,
                sample_scale=config.table_repair_content_scale,
                padding=config.table_repair_content_padding,
                x_overlap=0,
                y_overlap=0,
                header_context_height=0,
                left_context_width=0,
                jpeg_quality=config.jpeg_quality,
            )
            tile_dir = cache_root / "sample" / "rapidocr-v4" / "tiles"
            tile_dir.mkdir(parents=True)
            for image_slice in slices:
                (tile_dir / f"{Path(image_slice.file_name).stem}.tsv").write_text(
                    "",
                    encoding="utf-8",
                )

            with patch(
                "afac_pipeline.baseline.run_local_numeric_matrix_ocr",
                return_value=vision_result,
            ) as run_local:
                result = rebuild_cached_local_matrix_repairs(
                    records=[record],
                    base_csv=base_csv,
                    output_csv=output_csv,
                    local_cache_root=cache_root,
                    config=config,
                )

            self.assertEqual(result.selected_file_names, ("sample.jpg",))
            run_local.assert_called_once()
            with output_csv.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows, [{"file_name": "sample.jpg", "ground_truth": markdown}])

    def test_local_matrix_route_is_disabled_by_default(self) -> None:
        with patch("afac_pipeline.baseline.run_local_numeric_matrix_ocr") as run_local:
            result = _maybe_local_matrix_repair(
                record=_record("sample.jpg"),
                config=BaselineConfig(
                    output_csv=Path("unused.csv"),
                    cache_dir=Path("unused-cache"),
                ),
                previous_markdown="short",
            )

        self.assertIsNone(result)
        run_local.assert_not_called()

    def test_local_matrix_route_respects_page_area_guard(self) -> None:
        with patch("afac_pipeline.baseline.run_local_numeric_matrix_ocr") as run_local:
            result = _maybe_local_matrix_repair(
                record=_record("sample.jpg"),
                config=BaselineConfig(
                    output_csv=Path("unused.csv"),
                    cache_dir=Path("unused-cache"),
                    table_local_ocr_backend="rapidocr",
                    table_local_ocr_min_pixels=1_000_000,
                ),
                previous_markdown="short",
            )

        self.assertIsNone(result)
        run_local.assert_not_called()

    def test_local_matrix_density_trigger_accepts_sparse_nontrivial_text(self) -> None:
        record = _inked_record(
            "sample.jpg",
            source="memory",
            size=(120, 120),
            ink_box=(10, 10, 110, 110),
        )
        config = BaselineConfig(
            output_csv=Path("unused.csv"),
            cache_dir=Path("unused-cache"),
            table_local_ocr_backend="rapidocr",
            table_local_ocr_min_pixels=0,
            table_local_ocr_trigger_max_chars=10,
            table_local_ocr_trigger_max_chars_per_content_pixel=1.0,
            table_repair_content_scale=1.0,
            table_repair_content_padding=0,
        )
        profile = profile_image(
            image_bytes=record.read_bytes(),
            threshold=config.table_repair_content_threshold,
            sample_scale=config.table_repair_content_scale,
        )

        self.assertTrue(
            _should_attempt_local_matrix_repair(
                profile=profile,
                previous_markdown="x" * 100,
                config=config,
            )
        )

    def test_huge_local_matrix_route_requires_a_material_sequence_gap(self) -> None:
        record = _inked_record(
            "sample.jpg",
            source="memory",
            size=(120, 120),
            ink_box=(10, 10, 110, 110),
        )
        profile = replace(
            profile_image(
                image_bytes=record.read_bytes(),
                sample_scale=1.0,
            ),
            width=10_000,
            height=10_000,
        )
        config = BaselineConfig(
            output_csv=Path("unused.csv"),
            cache_dir=Path("unused-cache"),
            table_local_ocr_backend="rapidocr",
            table_local_ocr_min_pixels=10_000_000,
            table_local_ocr_max_pixels=400_000_000,
            table_local_ocr_huge_page_min_pixels=100_000_000,
            table_local_ocr_huge_min_sequence_gap=8,
            table_local_ocr_trigger_max_chars_per_content_pixel=1.0,
        )
        complete = _numeric_pipe_table((*range(0, 30),))
        gapped = _numeric_pipe_table((*range(0, 20), *range(31, 40)))

        self.assertEqual(_largest_pipe_numeric_sequence_gap(complete), 0)
        self.assertEqual(_largest_pipe_numeric_sequence_gap(gapped), 11)
        self.assertFalse(
            _should_attempt_local_matrix_repair(
                profile=profile,
                previous_markdown=complete,
                config=config,
            )
        )
        self.assertTrue(
            _should_attempt_local_matrix_repair(
                profile=profile,
                previous_markdown=gapped,
                config=config,
            )
        )

    def test_local_table_repair_preserves_compatible_remote_context(self) -> None:
        previous = (
            "# Accurate remote title\n\n"
            "| Year | Value |\n"
            "| --- | --- |\n"
            "| 0 | 10 |\n"
            "\nRemote footnote\n"
        )
        repaired = (
            "# Noisy local tit1e\n\n"
            "| Year | Value |\n"
            "| --- | --- |\n"
            "| 0 | 10 |\n"
            "| 1 | 20 |\n"
        )

        self.assertEqual(
            _preserve_pipe_table_context(previous, repaired),
            (
                "# Accurate remote title\n\n"
                "| Year | Value |\n"
                "| --- | --- |\n"
                "| 0 | 10 |\n"
                "| 1 | 20 |\n\n"
                "Remote footnote\n"
            ),
        )

    def test_local_table_context_accepts_the_same_numeric_axis_without_blanks(self) -> None:
        previous = (
            "# Accurate remote title\n\n"
            "| Policy year | 1 | 2 |  | 3 | 4 |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| Age | 10 | 20 |  | 30 | 40 |\n"
        )
        repaired = (
            "# Noisy local title\n\n"
            "| Policy year\\Age | 1 | 2 | 3 | 4 |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| 0 | 10 | 20 | 30 | 40 |\n"
        )

        merged = _preserve_pipe_table_context(previous, repaired)

        self.assertTrue(merged.startswith("# Accurate remote title"))
        self.assertIn("| Policy year\\Age | 1 | 2 | 3 | 4 |", merged)
        self.assertNotIn("Noisy local title", merged)

    def test_sequence_gap_repair_preserves_existing_nonempty_cells(self) -> None:
        previous = (
            "# Trusted title\n\n"
            "| Year | A | B |\n"
            "| --- | --- | --- |\n"
            + "".join(f"| {key} | old-{key} |  |\n" for key in range(12))
            + "\n"
            + "".join(f"| {key} | old-{key} | kept-{key} |\n" for key in range(20, 24))
        )
        repaired = (
            "# Noisy title\n\n"
            "| Year | A | B |\n"
            "| --- | --- | --- |\n"
            + "".join(f"| {key} | new-{key} | fill-{key} |\n" for key in range(24))
        )

        merged = _merge_previous_sequence_table_cells(previous, repaired)

        self.assertTrue(merged.startswith("# Trusted title"))
        self.assertIn("| 5 | old-5 | fill-5 |", merged)
        self.assertIn("| 15 | new-15 | fill-15 |", merged)
        self.assertIn("| 22 | old-22 | kept-22 |", merged)

    def test_partial_sequence_row_aligns_around_a_missing_middle_band(self) -> None:
        self.assertEqual(
            _merge_partial_sequence_row(
                ("78", "4,325", "4,324", "4,272", "4,265"),
                ("78", "4.325", "4.324", "4.320", "4.319", "4.272", "4.265"),
            ),
            ("78", "4,325", "4,324", "4.320", "4.319", "4,272", "4,265"),
        )

    def test_local_long_text_route_requires_sparse_remote_and_material_coverage(self) -> None:
        record = _record("sample.jpg", size=(20, 120))
        config = BaselineConfig(
            output_csv=Path("unused.csv"),
            cache_dir=Path("unused-cache"),
            long_local_ocr_backend="rapidocr",
            long_local_ocr_min_pixels=1,
            long_local_ocr_max_width=30,
            long_local_ocr_trigger_char_density=0.10,
            long_local_ocr_min_char_density=0.01,
            long_local_ocr_min_gain=2,
        )
        observations = [
            VisionObservation(10, 20, 10, 8, 0.99, "完整正文"),
            VisionObservation(10, 40, 10, 8, 0.99, "第二行"),
        ]

        with patch(
            "afac_pipeline.baseline.run_rapidocr_observations",
            return_value=observations,
        ) as local:
            result = _maybe_local_long_text_repair(
                record=record,
                config=config,
                previous_markdown="短",
            )

        self.assertEqual(result, "完整正文\n\n第二行")
        local.assert_called_once()

    def test_local_long_text_route_does_not_use_table_shape_to_reject_coverage(self) -> None:
        record = _record("sample.jpg", size=(20, 120))
        config = BaselineConfig(
            output_csv=Path("unused.csv"),
            cache_dir=Path("unused-cache"),
            long_local_ocr_backend="rapidocr",
            long_local_ocr_min_pixels=1,
            long_local_ocr_max_width=30,
            long_local_ocr_trigger_char_density=1.0,
            long_local_ocr_min_char_density=0.01,
            long_local_ocr_min_gain=2,
        )
        remote_fragment = "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
        observations = [
            VisionObservation(
                10,
                20,
                10,
                8,
                0.99,
                "恢复的完整正文，覆盖此前遗漏的大段条款内容，并包含大量重要定义和责任说明。",
            )
        ]

        with patch(
            "afac_pipeline.baseline.run_rapidocr_observations",
            return_value=observations,
        ):
            result = _maybe_local_long_text_repair(
                record=record,
                config=config,
                previous_markdown=remote_fragment,
            )

        self.assertEqual(
            result,
            "恢复的完整正文，覆盖此前遗漏的大段条款内容，并包含大量重要定义和责任说明。",
        )

    def test_accepted_local_matrix_avoids_remote_repair_fanout(self) -> None:
        markdown = (
            "<table><tr><th>年度/年龄</th><th>55</th></tr>"
            "<tr><td>1</td><td>1000</td></tr></table>\n"
        )
        vision_result = VisionMatrixResult(
            markdown=markdown,
            rows=1,
            cols=2,
            populated_cells=2,
            total_cells=2,
            header_sequence_inliers=1,
            row_sequence_inliers=1,
        )
        config = BaselineConfig(
            output_csv=Path("unused.csv"),
            cache_dir=Path("unused-cache"),
            table_repair_min_chars=1,
            table_repair_min_gain=0,
            table_local_ocr_backend="rapidocr",
            table_local_ocr_min_pixels=0,
        )

        with (
            patch(
                "afac_pipeline.baseline.run_local_numeric_matrix_ocr",
                return_value=vision_result,
            ),
            patch("afac_pipeline.baseline._call_table_content_grid_record") as remote,
        ):
            repaired = _maybe_repair_short_table(
                client=_FakeClient(),
                record=_record("sample.jpg"),
                config=config,
                markdown="complete anchor",
            )

        self.assertEqual(repaired, (markdown, 0))
        remote.assert_not_called()

    def test_raw_tile_cache_namespaces_ignore_final_selection_knobs(self) -> None:
        config = BaselineConfig(
            output_csv=Path("submission.csv"),
            cache_dir=Path("cache"),
        )
        changed_selection = replace(
            config,
            table_repair_min_chars=config.table_repair_min_chars + 100,
            table_min_score=config.table_min_score + 1,
        )

        self.assertEqual(
            _baseline_cache_namespace(config, "repair_tiles"),
            _baseline_cache_namespace(changed_selection, "repair_tiles"),
        )
        self.assertEqual(
            _baseline_cache_namespace(config, "coverage_tiles"),
            _baseline_cache_namespace(changed_selection, "coverage_tiles"),
        )
        self.assertNotEqual(
            _baseline_cache_namespace(config, "repair_tiles"),
            _baseline_cache_namespace(
                replace(config, table_repair_snap_boundaries=True),
                "repair_tiles",
            ),
        )
        self.assertEqual(
            _baseline_cache_namespace(config, "repair_tiles"),
            _baseline_cache_namespace(
                replace(config, table_refine_max_depth=2),
                "repair_tiles",
            ),
        )
        self.assertEqual(
            _baseline_cache_namespace(config, "coverage_tiles"),
            _baseline_cache_namespace(
                replace(config, table_refine_max_depth=2),
                "coverage_tiles",
            ),
        )
        self.assertNotEqual(
            _baseline_cache_namespace(config, "repair_tiles"),
            _baseline_cache_namespace(
                replace(config, table_repair_snap_x_boundaries=True),
                "repair_tiles",
            ),
        )
        local_enabled = replace(config, table_local_ocr_backend="rapidocr")
        self.assertEqual(
            _baseline_cache_namespace(config, "repair_tiles"),
            _baseline_cache_namespace(local_enabled, "repair_tiles"),
        )
        self.assertNotEqual(
            _baseline_cache_namespace(config, "table"),
            _baseline_cache_namespace(local_enabled, "table"),
        )
        saturated_refinement = replace(
            local_enabled,
            table_local_ocr_refine_saturated=True,
        )
        self.assertEqual(
            _baseline_cache_namespace(local_enabled, "local_rapidocr_matrix"),
            _baseline_cache_namespace(
                saturated_refinement,
                "local_rapidocr_matrix",
            ),
        )
        self.assertEqual(
            _baseline_cache_namespace(local_enabled, "local_rapidocr_matrix"),
            _baseline_cache_namespace(
                replace(
                    local_enabled,
                    table_local_ocr_small_page_max_pixels=20_000_000,
                    table_local_ocr_small_target_tile_width=800,
                    table_local_ocr_small_target_tile_height=800,
                ),
                "local_rapidocr_matrix",
            ),
        )
        self.assertNotEqual(
            _baseline_cache_namespace(local_enabled, "table"),
            _baseline_cache_namespace(saturated_refinement, "table"),
        )

    def test_adaptive_repair_grid_fits_call_budget_without_large_aspect_distortion(self) -> None:
        self.assertEqual(_fit_grid_to_call_budget(12, 7, 48), (9, 5))
        self.assertEqual(_fit_grid_to_call_budget(8, 9, 48), (6, 8))
        self.assertEqual(_fit_grid_to_call_budget(5, 6, 48), (5, 6))

    def test_repair_grid_uses_content_box_target_tile_size(self) -> None:
        record = _inked_record(
            "content-box.jpg",
            source="train",
            size=(2000, 1600),
            ink_box=(200, 200, 1800, 1400),
        )
        config = BaselineConfig(
            output_csv=Path("unused.csv"),
            cache_dir=Path("unused-cache"),
            table_repair_target_tile_width=801,
            table_repair_target_tile_height=601,
            table_repair_content_scale=1.0,
            table_repair_content_padding=0,
        )

        self.assertEqual(_table_repair_grid(record.read_bytes(), config), (2, 2))

    def test_local_matrix_uses_small_page_geometry_without_changing_remote_grid(self) -> None:
        record = _inked_record(
            "small-local-table.jpg",
            source="train",
            size=(2_000, 1_600),
            ink_box=(200, 200, 1_800, 1_400),
        )
        config = BaselineConfig(
            output_csv=Path("unused.csv"),
            cache_dir=Path("unused-cache"),
            table_repair_target_tile_width=1_500,
            table_repair_target_tile_height=800,
            table_repair_content_scale=0.05,
            table_repair_content_padding=150,
            table_local_ocr_small_page_max_pixels=20_000_000,
            table_local_ocr_small_target_tile_width=800,
            table_local_ocr_small_target_tile_height=800,
            table_local_ocr_small_content_scale=0.04,
            table_local_ocr_small_content_padding=200,
        )

        local = _local_matrix_geometry_config(record.read_bytes(), config)

        self.assertEqual(
            (config.table_repair_target_tile_width, config.table_repair_content_scale),
            (1_500, 0.05),
        )
        self.assertEqual(
            (
                local.table_repair_target_tile_width,
                local.table_repair_target_tile_height,
                local.table_repair_content_scale,
                local.table_repair_content_padding,
            ),
            (800, 800, 0.04, 200),
        )

    def test_repair_grid_keeps_very_tall_content_in_one_column(self) -> None:
        record = _inked_record(
            "tall-table.jpg",
            source="train",
            size=(2000, 5000),
            ink_box=(200, 200, 1800, 4800),
        )
        config = BaselineConfig(
            output_csv=Path("unused.csv"),
            cache_dir=Path("unused-cache"),
            table_repair_target_tile_width=801,
            table_repair_target_tile_height=601,
            table_repair_content_scale=1.0,
            table_repair_content_padding=0,
        )

        rows, cols = _table_repair_grid(record.read_bytes(), config)
        self.assertGreater(rows, 1)
        self.assertEqual(cols, 1)

    def test_large_grid_failure_budget_can_scale_by_eligible_part_ratio(self) -> None:
        config = BaselineConfig(
            output_csv=Path("unused.csv"),
            cache_dir=Path("unused-cache"),
            table_repair_max_failed_parts=8,
            table_repair_max_failed_ratio=0.25,
        )

        self.assertEqual(_table_repair_failed_part_budget(config, [True] * 48), 12)
        self.assertEqual(_table_repair_failed_part_budget(config, [True] * 25), 8)

    def test_repair_rejects_runaway_duplicate_lines_before_structure_upgrade(self) -> None:
        repaired = "<table><tr><td>ok</td></tr></table>\n" + ("repeat\n" * 100)

        self.assertFalse(
            _repair_is_better(
                "plain anchor",
                repaired,
                min_gain=0,
                max_duplicate_line_ratio=0.30,
            )
        )

    def test_strong_local_matrix_can_override_generic_cell_count_score(self) -> None:
        config = BaselineConfig(
            output_csv=Path("unused.csv"),
            cache_dir=Path("unused-cache"),
            table_repair_min_gain=300,
            table_max_duplicate_line_ratio=0.30,
        )
        result = VisionMatrixResult(
            markdown="unused",
            rows=12,
            cols=72,
            populated_cells=860,
            total_cells=864,
            header_sequence_inliers=132,
            row_sequence_inliers=6,
            table_count=2,
            row_starts=(1, 1),
            header_starts=(0, 0),
        )

        with patch("afac_pipeline.baseline._repair_is_better", return_value=False):
            self.assertTrue(
                _local_matrix_repair_is_better(
                    "x" * 1_000,
                    "y" * 1_400,
                    result=result,
                    config=config,
                )
            )

    def test_local_matrix_rejects_negative_policy_age_axis(self) -> None:
        config = BaselineConfig(
            output_csv=Path("unused.csv"),
            cache_dir=Path("unused-cache"),
            table_repair_min_gain=0,
        )
        result = VisionMatrixResult(
            markdown="unused",
            rows=12,
            cols=8,
            populated_cells=96,
            total_cells=96,
            header_sequence_inliers=7,
            row_sequence_inliers=12,
            header_starts=(-2,),
        )
        repaired = (
            "| 年度/年龄 | -2 | -1 | 0 | 1 | 2 | 3 | 4 |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
            + "".join(
                f"| {year} | 1 | 1 | 1 | 1 | 1 | 1 | 1 |\n"
                for year in range(12)
            )
        )

        self.assertFalse(
            _local_matrix_repair_is_better(
                "short remote result",
                repaired,
                result=result,
                config=config,
            )
        )

    def test_local_matrix_can_replace_systematically_split_decimals(self) -> None:
        header = "| Age | A | B | C | D |\n| --- | --- | --- | --- | --- |\n"
        previous = "# Remote title\n\n" + header + "".join(
            f"| {age} | {900 + age} | .50 | {1000 + age}. | 00 |\n"
            for age in range(20)
        )
        repaired = "# Remote title\n\n" + header + "".join(
            f"| {age} | {900 + age}.50 | {1000 + age}.00 | 0.00 | 0.00 |\n"
            for age in range(20)
        )
        result = VisionMatrixResult(
            markdown="unused",
            rows=20,
            cols=5,
            populated_cells=100,
            total_cells=100,
            header_sequence_inliers=0,
            row_sequence_inliers=20,
        )
        config = BaselineConfig(
            output_csv=Path("unused.csv"),
            cache_dir=Path("unused-cache"),
            table_repair_min_gain=300,
        )

        with patch("afac_pipeline.baseline._repair_is_better", return_value=False):
            self.assertTrue(
                _local_matrix_repair_is_better(
                    previous,
                    repaired,
                    result=result,
                    config=config,
                )
            )

    def test_exact_local_axes_can_replace_a_blank_split_numeric_header(self) -> None:
        previous_header = (
            "| Age | 1 | 2 |  | 3 | 4 | 5 |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
        )
        previous = previous_header + "".join(
            f"| {age} | {age}.1 | {age}.2 |  | {age}.3 | {age}.4 | {age}.5 |\n"
            for age in range(20)
        )
        repaired_header = (
            "| Age | 1 | 2 | 3 | 4 | 5 |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
        )
        repaired = repaired_header + "".join(
            f"| {age} | {age}.1 | {age}.2 | {age}.3 | {age}.4 | {age}.5 |\n"
            for age in range(20)
        )
        result = VisionMatrixResult(
            markdown="unused",
            rows=20,
            cols=6,
            populated_cells=120,
            total_cells=120,
            header_sequence_inliers=5,
            row_sequence_inliers=20,
        )
        config = BaselineConfig(
            output_csv=Path("unused.csv"),
            cache_dir=Path("unused-cache"),
            table_repair_min_gain=300,
        )

        with patch("afac_pipeline.baseline._repair_is_better", return_value=False):
            self.assertTrue(
                _local_matrix_repair_is_better(
                    previous,
                    repaired,
                    result=result,
                    config=config,
                )
            )

    def test_weak_local_matrix_cannot_override_generic_score(self) -> None:
        config = BaselineConfig(
            output_csv=Path("unused.csv"),
            cache_dir=Path("unused-cache"),
            table_repair_min_gain=300,
        )
        result = VisionMatrixResult(
            markdown="unused",
            rows=12,
            cols=72,
            populated_cells=600,
            total_cells=864,
            header_sequence_inliers=132,
            row_sequence_inliers=6,
            table_count=2,
        )

        with patch("afac_pipeline.baseline._repair_is_better", return_value=False):
            self.assertFalse(
                _local_matrix_repair_is_better(
                    "x" * 1_000,
                    "y" * 1_400,
                    result=result,
                    config=config,
                )
            )

    def test_rejects_unstructured_content_grid_repair_before_caching(self) -> None:
        record = _record("sample.jpg")
        crop_markdown = "partial crop"
        invalid_repair = (
            "年度 年龄 现金价值\n"
            "1 59 880.50\n"
            "2 60 900.94\n"
            "3 61 921.48\n"
            "4 62 942.13\n"
        )
        config = BaselineConfig(
            output_csv=Path("unused.csv"),
            cache_dir=Path("unused-cache"),
            table_repair_min_chars=600,
        )

        with patch(
            "afac_pipeline.baseline._call_table_content_grid_record",
            return_value=(invalid_repair, 4),
        ):
            repaired = _maybe_repair_short_table(
                client=_FakeClient(),
                record=record,
                config=config,
                markdown=crop_markdown,
            )

        self.assertEqual(repaired, (crop_markdown, 4))

    def test_content_grid_repair_refines_a_truncated_tile(self) -> None:
        record = _record("sample.jpg")

        class _TruncatingClient:
            def __init__(self) -> None:
                self.calls = 0

            def call_with_file(self, file_name: str, file_bytes: bytes, **kwargs: object) -> str:
                self.calls += 1
                if self.calls == 1:
                    raise FinixDocError("candidate output appears truncated")
                return "<table><tr><td>ok</td></tr></table>\n"

        image_slice = make_grid_slices(
            file_name=record.file_name,
            image_bytes=record.read_bytes(),
        )[0]
        client = _TruncatingClient()
        markdown, calls = _call_table_repair_slice(
            client,
            image_slice,
            BaselineConfig(
                output_csv=Path("unused.csv"),
                cache_dir=Path("unused-cache"),
                table_refine_max_depth=1,
                table_refine_rows=2,
                table_refine_cols=2,
            ),
        )

        self.assertEqual(calls, 5)
        self.assertEqual(client.calls, 5)
        self.assertIn("<table>", markdown)

    def test_refined_repair_reuses_successful_child_tiles(self) -> None:
        record = _record("sample.jpg")

        class _ParentTruncatingClient:
            def __init__(self) -> None:
                self.calls = 0

            def call_with_file(
                self,
                file_name: str,
                file_bytes: bytes,
                **kwargs: object,
            ) -> str:
                self.calls += 1
                if file_name == "sample.jpg":
                    raise FinixDocError("candidate output appears truncated")
                return f"<table><tr><td>{file_name}</td></tr></table>\n"

        image_slice = make_grid_slices(
            file_name=record.file_name,
            image_bytes=record.read_bytes(),
        )[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            config = BaselineConfig(
                output_csv=Path(tmpdir) / "unused.csv",
                cache_dir=cache_dir,
                table_refine_max_depth=1,
                table_refine_rows=2,
                table_refine_cols=2,
            )
            first_client = _ParentTruncatingClient()
            first_markdown, first_calls = _call_table_repair_slice(
                first_client,
                image_slice,
                config,
                cache_dir=cache_dir,
            )
            second_client = _ParentTruncatingClient()
            second_markdown, second_calls = _call_table_repair_slice(
                second_client,
                image_slice,
                config,
                cache_dir=cache_dir,
            )

        self.assertEqual(first_calls, 5)
        self.assertEqual(first_client.calls, 5)
        self.assertEqual(second_calls, 1)
        self.assertEqual(second_client.calls, 1)
        self.assertEqual(second_markdown, first_markdown)

    def test_oversized_repair_tile_skips_unstable_parent_call(self) -> None:
        record = _record("sample.jpg")

        class _ChildOnlyClient:
            def __init__(self) -> None:
                self.file_names: list[str] = []

            def call_with_file(
                self,
                file_name: str,
                file_bytes: bytes,
                **kwargs: object,
            ) -> str:
                self.file_names.append(file_name)
                return f"<table><tr><td>{file_name}</td></tr></table>\n"

        image_slice = make_grid_slices(
            file_name=record.file_name,
            image_bytes=record.read_bytes(),
        )[0]
        client = _ChildOnlyClient()
        markdown, calls = _call_table_repair_slice(
            client,
            image_slice,
            BaselineConfig(
                output_csv=Path("unused.csv"),
                cache_dir=Path("unused-cache"),
                table_repair_target_tile_width=40,
                table_repair_target_tile_height=40,
                table_refine_max_depth=1,
                table_refine_rows=2,
                table_refine_cols=2,
            ),
        )

        self.assertEqual(calls, 4)
        self.assertEqual(len(client.file_names), 4)
        self.assertNotIn("sample.jpg", client.file_names)
        self.assertIn("<table>", markdown)

    def test_table_repair_stops_after_failed_tile_budget(self) -> None:
        record = _inked_record(
            "sample.jpg",
            source="memory",
            size=(100, 100),
            ink_box=(10, 10, 90, 90),
        )

        class _AlwaysFailingClient:
            def __init__(self) -> None:
                self.calls = 0

            def call_with_file(self, file_name: str, file_bytes: bytes, **kwargs: object) -> str:
                self.calls += 1
                raise FinixDocError("request timed out")

        client = _AlwaysFailingClient()
        with self.assertRaises(_RouteFailure) as context:
            _call_table_content_grid_record(
                client,
                record,
                BaselineConfig(
                    output_csv=Path("unused.csv"),
                    cache_dir=Path("unused-cache"),
                    table_repair_rows=2,
                    table_repair_cols=2,
                    table_repair_min_success_parts=1,
                    table_repair_max_failed_parts=2,
                    retries=0,
                ),
                previous_markdown="<table><tr><td>anchor</td></tr></table>\n",
            )

        self.assertIn("failed tile budget", str(context.exception))
        self.assertEqual(client.calls, 2)

    def test_table_repair_stops_after_three_identical_large_tile_outputs(self) -> None:
        record = _record("sample.jpg")
        slices = make_grid_slices(
            file_name=record.file_name,
            image_bytes=record.read_bytes(),
            slice_width=30,
        )

        class _RepeatingClient:
            def __init__(self) -> None:
                self.calls = 0

            def call_with_file(self, file_name: str, file_bytes: bytes, **kwargs: object) -> str:
                self.calls += 1
                return "<table><tr><td>" + ("same" * 20) + "</td></tr></table>\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            client = _RepeatingClient()
            config = BaselineConfig(
                output_csv=Path(tmpdir) / "unused.csv",
                cache_dir=Path(tmpdir) / "cache",
                table_repair_min_success_parts=1,
                table_repair_min_gain=0,
                table_repair_max_identical_parts=3,
                table_repair_identical_min_chars=10,
            )
            with patch(
                "afac_pipeline.baseline.make_content_grid_slices",
                return_value=slices,
            ):
                with self.assertRaises(_RouteFailure) as raised:
                    _call_table_content_grid_record(
                        client,
                        record,
                        config,
                        previous_markdown="",
                    )

            self.assertIn("same large OCR result", str(raised.exception))
            self.assertEqual(raised.exception.calls, 3)
            self.assertEqual(client.calls, 3)

    def test_table_repair_tile_rejects_repeated_unstructured_numeric_rows(self) -> None:
        record = _record("sample.jpg")
        image_slice = make_grid_slices(
            file_name=record.file_name,
            image_bytes=record.read_bytes(),
        )[0]

        class _RunawayNumericClient:
            def call_with_file(
                self,
                file_name: str,
                file_bytes: bytes,
                **kwargs: object,
            ) -> str:
                return "\n".join(["10905", "11124", "11346", "11573"] * 100)

        with self.assertRaises(_RouteFailure) as raised:
            _call_table_repair_slice(
                _RunawayNumericClient(),
                image_slice,
                BaselineConfig(
                    output_csv=Path("unused.csv"),
                    cache_dir=Path("unused-cache"),
                    table_refine_max_depth=0,
                ),
            )

        self.assertIn("incomplete unstructured table text", str(raised.exception))
        self.assertEqual(raised.exception.calls, 1)

    def test_timeout_is_retryable_but_not_a_refinable_table_tile_failure(self) -> None:
        self.assertTrue(_is_truncation_error(FinixDocError("request timed out")))
        self.assertFalse(
            _is_refinable_table_tile_failure(FinixDocError("request timed out"))
        )
        self.assertTrue(
            _is_refinable_table_tile_failure(
                FinixDocError(
                    "candidate output looks like incomplete unstructured table text"
                )
            )
        )
        self.assertTrue(
            _is_refinable_table_tile_failure(
                FinixDocError("candidate HTML table structure has unclosed <tr>")
            )
        )
        self.assertTrue(
            _is_refinable_table_tile_failure(
                FinixDocError(
                    "candidate HTML table structure closes </table> while <td> is open"
                )
            )
        )

    def test_table_repair_slice_stops_at_its_call_budget(self) -> None:
        record = _record("sample.jpg")

        class _TruncatingClient:
            def __init__(self) -> None:
                self.calls = 0

            def call_with_file(self, file_name: str, file_bytes: bytes, **kwargs: object) -> str:
                self.calls += 1
                if self.calls == 1:
                    raise FinixDocError("candidate output appears truncated")
                return "<table><tr><td>ok</td></tr></table>\n"

        image_slice = make_grid_slices(
            file_name=record.file_name,
            image_bytes=record.read_bytes(),
        )[0]
        client = _TruncatingClient()
        with self.assertRaises(_RouteFailure) as context:
            _call_table_repair_slice(
                client,
                image_slice,
                BaselineConfig(
                    output_csv=Path("unused.csv"),
                    cache_dir=Path("unused-cache"),
                    table_refine_max_depth=1,
                ),
                remaining_calls=2,
            )

        self.assertEqual(context.exception.calls, 2)
        self.assertEqual(client.calls, 2)

    def test_dense_table_raises_the_repair_trigger_without_filename_rules(self) -> None:
        record = _inked_record(
            "sample.jpg",
            source="memory",
            size=(400, 300),
            ink_box=(20, 20, 380, 280),
        )
        trigger = _table_repair_trigger_chars(
            record,
            BaselineConfig(
                output_csv=Path("unused.csv"),
                cache_dir=Path("unused-cache"),
                table_repair_min_chars=600,
                table_repair_min_chars_per_content_pixel=1.0,
                table_repair_content_threshold=250,
                table_repair_content_scale=1.0,
            ),
        )

        self.assertGreater(trigger, 600)

    def test_content_grid_repair_reuses_completed_tile_cache(self) -> None:
        record = _record("sample.jpg")
        image_slice = make_grid_slices(
            file_name=record.file_name,
            image_bytes=record.read_bytes(),
        )[0]

        class _TableClient:
            def __init__(self) -> None:
                self.calls = 0

            def call_with_file(self, file_name: str, file_bytes: bytes, **kwargs: object) -> str:
                self.calls += 1
                return "<table><tr><td>ok</td></tr></table>\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BaselineConfig(
                output_csv=Path(tmpdir) / "unused.csv",
                cache_dir=Path(tmpdir) / "cache",
                resume=False,
                table_repair_min_success_parts=1,
                table_repair_min_gain=0,
            )
            client = _TableClient()
            with patch(
                "afac_pipeline.baseline.make_content_grid_slices",
                return_value=[image_slice],
            ):
                first, first_calls = _call_table_content_grid_record(
                    client,
                    record,
                    config,
                    previous_markdown="",
                )
                second, second_calls = _call_table_content_grid_record(
                    client,
                    record,
                    config,
                    previous_markdown="",
                )

        self.assertIn("<table>", first)
        self.assertEqual(second, first)
        self.assertEqual(first_calls, 1)
        self.assertEqual(second_calls, 0)
        self.assertEqual(client.calls, 1)

    def test_content_grid_repair_can_read_tiles_concurrently_in_grid_order(self) -> None:
        record = _record("sample.jpg")
        slices = make_grid_slices(
            file_name=record.file_name,
            image_bytes=record.read_bytes(),
            slice_width=50,
        )[:2]
        barrier = threading.Barrier(2)

        def call_slice(client, image_slice, config, *, remaining_calls, cache_dir):
            barrier.wait(timeout=2)
            return (
                f"<table><tr><td>{image_slice.col}</td></tr></table>\n",
                1,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BaselineConfig(
                output_csv=Path(tmpdir) / "unused.csv",
                cache_dir=Path(tmpdir) / "cache",
                table_repair_min_success_parts=1,
                table_repair_min_gain=0,
                table_repair_max_calls=2,
                table_repair_workers=2,
                table_refine_max_depth=0,
                table_repair_min_content_pixels=0,
                table_repair_min_content_ratio=0.0,
            )
            with (
                patch(
                    "afac_pipeline.baseline.make_content_grid_slices",
                    return_value=slices,
                ),
                patch(
                    "afac_pipeline.baseline._call_table_repair_slice",
                    side_effect=call_slice,
                ),
            ):
                markdown, calls = _call_table_content_grid_record(
                    object(),
                    record,
                    config,
                    previous_markdown="",
                )

        self.assertEqual(calls, 2)
        self.assertLess(markdown.index(">1<"), markdown.index(">2<"))

    def test_parallel_content_grid_repair_bounds_recursive_refinement_per_tile(self) -> None:
        record = _record("sample.jpg")
        slices = make_grid_slices(
            file_name=record.file_name,
            image_bytes=record.read_bytes(),
            slice_width=50,
        )[:2]
        received_budgets: list[int | None] = []

        def call_slice(client, image_slice, config, *, remaining_calls, cache_dir):
            received_budgets.append(remaining_calls)
            return (
                f"<table><tr><td>{image_slice.col}</td></tr></table>\n",
                remaining_calls,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BaselineConfig(
                output_csv=Path(tmpdir) / "unused.csv",
                cache_dir=Path(tmpdir) / "cache",
                table_repair_min_success_parts=1,
                table_repair_min_gain=0,
                table_repair_max_calls=2,
                table_repair_workers=2,
                table_refine_max_depth=1,
                table_refine_rows=2,
                table_refine_cols=2,
                table_repair_min_content_pixels=0,
                table_repair_min_content_ratio=0.0,
            )
            with (
                patch(
                    "afac_pipeline.baseline.make_content_grid_slices",
                    return_value=slices,
                ),
                patch(
                    "afac_pipeline.baseline._call_table_repair_slice",
                    side_effect=call_slice,
                ),
            ):
                markdown, calls = _call_table_content_grid_record(
                    object(),
                    record,
                    config,
                    previous_markdown="",
                )

        self.assertEqual(calls, 10)
        self.assertEqual(received_budgets, [5, 5])
        self.assertLess(markdown.index(">1<"), markdown.index(">2<"))

    def test_rejects_candidate_with_excessive_table_blocks(self) -> None:
        config = BaselineConfig(
            output_csv=Path('unused.csv'),
            cache_dir=Path('unused-cache'),
            table_max_blocks=2,
        )
        markdown = '<table></table>\n' * 3

        self.assertEqual(
            _table_candidate_issue(markdown, config),
            'candidate contains too many separate table blocks (3 > 2)',
        )

    def test_rejects_pathologically_flattened_html_row(self) -> None:
        config = BaselineConfig(
            output_csv=Path('unused.csv'),
            cache_dir=Path('unused-cache'),
        )
        markdown = (
            '<table><tr>'
            + ''.join(f'<th>{index}</th>' for index in range(257))
            + '</tr></table>'
        )

        self.assertEqual(
            _table_candidate_issue(markdown, config),
            'candidate contains a pathologically wide HTML row (257 > 256 cells)',
        )

    def test_coverage_rejects_excessive_duplicate_lines(self) -> None:
        config = BaselineConfig(
            output_csv=Path('unused.csv'),
            cache_dir=Path('unused-cache'),
            table_max_duplicate_line_ratio=0.30,
        )
        markdown = '<table><tr><td>header</td></tr></table>\n' + ('repeat\n' * 8)

        self.assertEqual(
            _coverage_candidate_issue(markdown, config),
            'coverage candidate has too many duplicate lines (0.778 > 0.300)',
        )

    def test_hybrid_coverage_requires_dense_content(self) -> None:
        config = BaselineConfig(
            output_csv=Path('unused.csv'),
            cache_dir=Path('unused-cache'),
            table_hybrid_min_content_ratio=0.50,
            table_repair_content_scale=1.0,
        )
        sparse = _record('sparse.jpg')
        dense = _inked_record(
            'dense.jpg',
            source='memory',
            size=(100, 100),
            ink_box=(0, 0, 99, 99),
        )

        self.assertFalse(_should_use_table_coverage(sparse, config))
        self.assertTrue(_should_use_table_coverage(dense, config))

    def test_refines_a_truncated_coverage_tile(self) -> None:
        record = _record("sample.jpg")

        class _TruncatingClient:
            def __init__(self) -> None:
                self.calls = 0

            def call_with_file(self, file_name: str, file_bytes: bytes, **kwargs: object) -> str:
                self.calls += 1
                if self.calls == 1:
                    raise FinixDocError("response appears truncated")
                return "| A | B |\n| --- | --- |\n| 1 | 2 |\n"

        tile = make_grid_slices(
            file_name=record.file_name,
            image_bytes=record.read_bytes(),
        )[0]
        client = _TruncatingClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown, calls = _call_coverage_tile(
                client,
                tile,
                BaselineConfig(
                    output_csv=Path(tmpdir) / "submission.csv",
                    cache_dir=Path(tmpdir) / "cache",
                    min_chars=1,
                ),
            )

        self.assertEqual(calls, 5)
        self.assertEqual(client.calls, 5)
        self.assertIn("| A | B |", markdown)

    def test_refines_a_fragmented_coverage_tile_horizontally(self) -> None:
        record = _record("sample.jpg")

        class _FragmentingClient:
            def __init__(self) -> None:
                self.calls = 0

            def call_with_file(self, file_name: str, file_bytes: bytes, **kwargs: object) -> str:
                self.calls += 1
                if self.calls == 1:
                    return "<table><tr><td>1</td></tr></table><table><tr><td>2</td></tr></table>"
                return "| A | B |\n| --- | --- |\n| 1 | 2 |\n"

        tile = make_grid_slices(
            file_name=record.file_name,
            image_bytes=record.read_bytes(),
        )[0]
        client = _FragmentingClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown, calls = _call_coverage_tile(
                client,
                tile,
                BaselineConfig(
                    output_csv=Path(tmpdir) / "submission.csv",
                    cache_dir=Path(tmpdir) / "cache",
                    min_chars=1,
                ),
            )

        self.assertEqual(calls, 3)
        self.assertEqual(client.calls, 3)
        self.assertIn("| A | B |", markdown)

    def test_run_baseline_submission_writes_original_file_name(self) -> None:
        record = _record("sample.jpg")
        client = _FakeClient()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_csv = root / "submission.csv"
            stats = run_baseline_submission(
                records=[record],
                client=client,
                config=BaselineConfig(
                    output_csv=output_csv,
                    cache_dir=root / "cache",
                    crop_sizes=(80,),
                    anchors=("top_left",),
                    min_chars=1,
                ),
            )

            self.assertEqual(stats.processed, 1)
            self.assertEqual(stats.api_calls, 1)
            self.assertEqual(stats.template_missing, 0)
            self.assertEqual(client.calls, ["sample_crop_top_left_80.jpg"])
            with output_csv.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["file_name"], "sample.jpg")
            self.assertIn("<table>", rows[0]["ground_truth"])

    def test_baseline_can_emit_validated_pipe_tables_at_output_boundary(self) -> None:
        record = _record("sample.jpg")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_csv = root / "submission.csv"
            run_baseline_submission(
                records=[record],
                client=_FakeClient(),
                config=BaselineConfig(
                    output_csv=output_csv,
                    cache_dir=root / "cache",
                    crop_sizes=(80,),
                    anchors=("top_left",),
                    min_chars=1,
                    table_output_format="markdown",
                ),
            )
            with output_csv.open(newline="", encoding="utf-8") as file:
                markdown = next(csv.DictReader(file))["ground_truth"]

        self.assertNotIn("<table", markdown)
        self.assertIn("| ok |", markdown)

    def test_tries_next_crop_after_failure(self) -> None:
        record = _record("sample.jpg")
        client = _FakeClient(fail_first=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stats = run_baseline_submission(
                records=[record],
                client=client,
                config=BaselineConfig(
                    output_csv=root / "submission.csv",
                    cache_dir=root / "cache",
                    crop_sizes=(80, 60),
                    anchors=("top_left",),
                    min_chars=1,
                ),
            )

            self.assertEqual(stats.api_calls, 2)
            self.assertEqual(client.calls, [
                "sample_crop_top_left_80.jpg",
                "sample_crop_top_left_60.jpg",
            ])

    def test_tries_next_crop_after_malformed_table(self) -> None:
        record = _record("sample.jpg")
        client = _FakeClient(malformed_first=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stats = run_baseline_submission(
                records=[record],
                client=client,
                config=BaselineConfig(
                    output_csv=root / "submission.csv",
                    cache_dir=root / "cache",
                    crop_sizes=(80, 60),
                    anchors=("top_left",),
                    min_chars=1,
                ),
            )

            self.assertEqual(stats.api_calls, 2)
            self.assertEqual(client.calls, [
                "sample_crop_top_left_80.jpg",
                "sample_crop_top_left_60.jpg",
            ])

    def test_tries_next_crop_after_unclosed_html_row_or_cell(self) -> None:
        record = _record("sample.jpg")
        client = _FakeClient(structurally_malformed_first=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stats = run_baseline_submission(
                records=[record],
                client=client,
                config=BaselineConfig(
                    output_csv=root / "submission.csv",
                    cache_dir=root / "cache",
                    crop_sizes=(80, 60),
                    anchors=("top_left",),
                    min_chars=1,
                ),
            )

            self.assertEqual(stats.api_calls, 2)
            self.assertEqual(
                client.calls,
                ["sample_crop_top_left_80.jpg", "sample_crop_top_left_60.jpg"],
            )

    def test_tries_next_crop_after_short_output(self) -> None:
        record = _record("sample.jpg")
        client = _FakeClient(short_first=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stats = run_baseline_submission(
                records=[record],
                client=client,
                config=BaselineConfig(
                    output_csv=root / "submission.csv",
                    cache_dir=root / "cache",
                    crop_sizes=(80, 60),
                    anchors=("top_left",),
                    min_chars=20,
                ),
            )

            self.assertEqual(stats.api_calls, 2)

    def test_expands_to_submission_template_with_missing_placeholders(self) -> None:
        record = _record("sample.jpg")
        client = _FakeClient()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_csv = root / "submission.csv"
            stats = run_baseline_submission(
                records=[record],
                client=client,
                config=BaselineConfig(
                    output_csv=output_csv,
                    cache_dir=root / "cache",
                    crop_sizes=(80,),
                    anchors=("top_left",),
                    min_chars=1,
                    submission_file_names=("missing.jpg", "sample.jpg"),
                    missing_markdown="<table></table>\n",
                ),
            )

            self.assertEqual(stats.processed, 2)
            self.assertEqual(stats.template_missing, 1)
            with output_csv.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0], {
                "file_name": "missing.jpg",
                "ground_truth": "<table></table>\n",
            })
            self.assertEqual(rows[1]["file_name"], "sample.jpg")

    def test_long_records_use_full_page_vertical_slices(self) -> None:
        record = _record(
            "sample.jpg",
            source="data/raw/AFAC A榜评测数据集(2)/finix_huge_long_rest_A/images/sample.jpg",
            size=(16, 160),
        )
        client = _FakeClient()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_csv = root / "submission.csv"
            stats = run_baseline_submission(
                records=[record],
                client=client,
                config=BaselineConfig(
                    output_csv=output_csv,
                    cache_dir=root / "cache",
                    long_slice_height=60,
                    long_slice_overlap=10,
                    long_min_chars=1,
                ),
            )

            self.assertEqual(stats.api_calls, 3)
            self.assertEqual(client.calls, [
                "sample_part001.jpg",
                "sample_part002.jpg",
                "sample_part003.jpg",
            ])
            with output_csv.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertIn("sample_part003.jpg", rows[0]["ground_truth"])

    def test_long_record_with_a_failed_slice_is_not_cached_as_partial(self) -> None:
        record = _record(
            "sample.jpg",
            source="data/raw/AFAC A榜评测数据集(2)/finix_huge_long_rest_A/images/sample.jpg",
            size=(16, 160),
        )
        client = _FakeClient(fail_first=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            with self.assertRaises(FinixDocError):
                run_baseline_submission(
                    records=[record],
                    client=client,
                    config=BaselineConfig(
                        output_csv=root / "submission.csv",
                        cache_dir=cache_dir,
                        long_slice_height=60,
                        long_slice_overlap=10,
                        long_min_chars=1,
                    ),
                )
            self.assertFalse(any(cache_dir.rglob("sample.md")))
            self.assertEqual(len(list(cache_dir.rglob("sample_part*.md"))), 2)

    def test_partial_long_result_retries_only_missing_slice_before_record_cache(self) -> None:
        record = _record(
            "sample.jpg",
            source="data/raw/finix_huge_long_rest_B/images/sample.jpg",
            size=(16, 160),
        )
        client = _FakeClient(fail_first=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            config = BaselineConfig(
                output_csv=root / "first.csv",
                cache_dir=cache_dir,
                long_slice_height=60,
                long_slice_overlap=10,
                long_min_chars=1,
                long_min_success_ratio=0.66,
                long_max_failed_parts=1,
            )
            first = run_baseline_submission(
                records=[record],
                client=client,
                config=config,
            )

            self.assertEqual(first.api_calls, 3)
            self.assertFalse(any(cache_dir.rglob("sample.md")))
            self.assertEqual(len(list(cache_dir.rglob("sample_part*.md"))), 2)

            second = run_baseline_submission(
                records=[record],
                client=client,
                config=replace(config, output_csv=root / "second.csv"),
            )

            self.assertEqual(second.api_calls, 1)
            self.assertTrue(any(cache_dir.rglob("sample.md")))
            self.assertEqual(len(list(cache_dir.rglob("sample_part*.md"))), 3)

    def test_sparse_long_output_uses_smaller_fallback_geometry(self) -> None:
        record = _inked_record(
            "sample.jpg",
            source="data/raw/finix_huge_long_rest_B/images/sample.jpg",
            size=(200, 1000),
            ink_box=(20, 20, 180, 980),
        )
        config = BaselineConfig(
            output_csv=Path("unused.csv"),
            cache_dir=Path("unused-cache"),
            long_slice_height=1200,
            long_fallback_slice_height=600,
            long_fallback_overlap=30,
            long_low_confidence_char_density=0.20,
        )

        with patch(
            "afac_pipeline.baseline._call_long_record",
            return_value=("fallback text", 4),
        ) as fallback:
            result = _try_long_slice_fallback(
                client=_FakeClient(),
                record=record,
                config=config,
                calls=2,
                markdown="x",
            )

        self.assertEqual(result, ("fallback text", 6))
        fallback.assert_called_once()
        fallback_config = fallback.call_args.args[2]
        self.assertEqual(fallback_config.long_slice_height, 600)
        self.assertEqual(fallback_config.long_slice_overlap, 30)
        self.assertEqual(fallback_config.long_low_confidence_char_density, 0.0)

    def test_short_table_output_triggers_content_grid_repair(self) -> None:
        record = _record(
            "sample.jpg",
            source="data/raw/AFAC A榜评测数据集(2)/finix_huge_table_rest_A/images/sample.jpg",
        )
        client = _FakeClient(short_first=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_csv = root / "submission.csv"
            stats = run_baseline_submission(
                records=[record],
                client=client,
                config=BaselineConfig(
                    output_csv=output_csv,
                    cache_dir=root / "cache",
                    crop_sizes=(80,),
                    anchors=("top_left",),
                    table_mode="anchor",
                    min_chars=1,
                    table_repair_min_chars=20,
                    table_repair_min_gain=0,
                    table_repair_rows=1,
                    table_repair_cols=1,
                    table_repair_min_content_pixels=0,
                    table_repair_min_content_ratio=0,
                    table_repair_min_success_parts=1,
                ),
            )

            self.assertEqual(stats.api_calls, 2)
            self.assertEqual(client.calls, [
                "sample_crop_top_left_80.jpg",
                "sample_content_r001_c001.jpg",
            ])
            with output_csv.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertIn("| A | B |", rows[0]["ground_truth"])

    def test_plain_text_tabular_output_triggers_content_grid_repair(self) -> None:
        record = _record(
            "sample.jpg",
            source="data/raw/AFAC A榜评测数据集(2)/finix_huge_table_rest_A/images/sample.jpg",
        )
        client = _FakeClient(tabular_text_first=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_csv = root / "submission.csv"
            stats = run_baseline_submission(
                records=[record],
                client=client,
                config=BaselineConfig(
                    output_csv=output_csv,
                    cache_dir=root / "cache",
                    crop_sizes=(80,),
                    anchors=("top_left",),
                    table_mode="anchor",
                    min_chars=1,
                    table_repair_min_chars=20,
                    table_repair_min_gain=999,
                    table_repair_rows=1,
                    table_repair_cols=1,
                    table_repair_min_content_pixels=0,
                    table_repair_min_content_ratio=0,
                    table_repair_min_success_parts=1,
                ),
            )

            self.assertEqual(stats.api_calls, 2)
            self.assertEqual(client.calls, [
                "sample_crop_top_left_80.jpg",
                "sample_content_r001_c001.jpg",
            ])
            with output_csv.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertIn("| A | B |", rows[0]["ground_truth"])

    def test_invalid_short_crop_prefers_grid_repair_before_next_crop(self) -> None:
        record = _record(
            "sample.jpg",
            source="data/raw/AFAC A榜评测数据集(2)/finix_huge_table_rest_A/images/sample.jpg",
        )
        client = _FakeClient(short_first=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stats = run_baseline_submission(
                records=[record],
                client=client,
                config=BaselineConfig(
                    output_csv=root / "submission.csv",
                    cache_dir=root / "cache",
                    crop_sizes=(80, 60),
                    anchors=("top_left",),
                    table_mode="anchor",
                    min_chars=20,
                    table_repair_min_chars=100,
                    table_repair_min_gain=0,
                    table_repair_rows=1,
                    table_repair_cols=1,
                    table_repair_min_content_pixels=0,
                    table_repair_min_content_ratio=0,
                    table_repair_min_success_parts=1,
                ),
            )

            self.assertEqual(stats.api_calls, 2)
            self.assertEqual(
                client.calls,
                [
                    "sample_crop_top_left_80.jpg",
                    "sample_content_r001_c001.jpg",
                ],
            )

    def test_table_cache_with_plain_text_tabular_output_is_invalidated(self) -> None:
        record = _record(
            "sample.jpg",
            source="data/raw/AFAC A榜评测数据集(2)/finix_huge_table_rest_A/images/sample.jpg",
        )
        client = _FakeClient()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = BaselineConfig(
                output_csv=root / "submission.csv",
                cache_dir=root / "cache",
                crop_sizes=(80,),
                anchors=("top_left",),
                min_chars=1,
                table_repair_min_chars=20,
            )
            cache_dir = config.cache_dir / _baseline_cache_namespace(config, "table")
            cache_path = _cache_path(cache_dir, record.file_name)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                "年度 年龄 现金价值\n"
                "1 59 880.50\n"
                "2 60 900.94\n"
                "3 61 921.48\n"
                "4 62 942.13\n",
                encoding="utf-8",
            )

            stats = run_baseline_submission(
                records=[record],
                client=client,
                config=config,
            )

            self.assertEqual(stats.cache_hits, 0)
            self.assertEqual(stats.api_calls, 1)
            self.assertEqual(client.calls, ["sample_crop_top_left_80.jpg"])
            self.assertIn("<table>", cache_path.read_text(encoding="utf-8"))

    def test_multiple_html_tables_are_not_rejected_as_unstructured_text(self) -> None:
        record = _record(
            "sample.jpg",
            source="data/raw/AFAC A榜评测数据集(2)/finix_huge_table_rest_A/images/sample.jpg",
        )

        class _MultiTableClient(_FakeClient):
            def call_with_file(
                self,
                file_name: str,
                file_bytes: bytes,
                **kwargs: object,
            ) -> str:
                self.calls.append(file_name)
                return (
                    "<table><tr><th>年度</th><th>现金价值</th></tr>"
                    "<tr><td>1</td><td>880.50</td></tr>"
                    "<tr><td>2</td><td>900.94</td></tr></table>\n"
                    "<table><tr><th>年度</th><th>退保价值</th></tr>"
                    "<tr><td>1</td><td>700.20</td></tr>"
                    "<tr><td>2</td><td>720.30</td></tr></table>\n"
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stats = run_baseline_submission(
                records=[record],
                client=_MultiTableClient(),
                config=BaselineConfig(
                    output_csv=root / "submission.csv",
                    cache_dir=root / "cache",
                    crop_sizes=(80,),
                    anchors=("top_left",),
                    min_chars=1,
                ),
            )

            self.assertEqual(stats.api_calls, 1)
            with (root / "submission.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["ground_truth"].count("<table>"), 2)

    def test_content_grid_repair_skips_low_content_slices(self) -> None:
        record = _inked_record(
            "sample.jpg",
            source="data/raw/AFAC A榜评测数据集(2)/finix_huge_table_rest_A/images/sample.jpg",
            size=(100, 40),
            ink_box=(5, 5, 24, 24),
        )
        client = _FakeClient(short_first=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_csv = root / "submission.csv"
            stats = run_baseline_submission(
                records=[record],
                client=client,
                config=BaselineConfig(
                    output_csv=output_csv,
                    cache_dir=root / "cache",
                    crop_sizes=(80,),
                    anchors=("top_left",),
                    table_mode="anchor",
                    min_chars=1,
                    table_repair_min_chars=20,
                    table_repair_min_gain=0,
                    table_repair_rows=1,
                    table_repair_cols=2,
                    table_repair_overlap=0,
                    table_repair_content_threshold=250,
                    table_repair_content_scale=1.0,
                    table_repair_content_padding=100,
                    table_repair_min_content_pixels=10,
                    table_repair_min_content_ratio=0.001,
                    table_repair_min_success_parts=1,
                ),
            )

            self.assertEqual(stats.api_calls, 2)
            self.assertEqual(client.calls, [
                "sample_crop_top_left_80.jpg",
                "sample_content_r001_c001.jpg",
            ])
            with output_csv.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertIn("| A | B |", rows[0]["ground_truth"])


def _record(
    file_name: str,
    *,
    source: str = "memory",
    size: tuple[int, int] = (120, 120),
) -> ImageRecord:
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="JPEG", quality=95)
    image_bytes = buffer.getvalue()
    return ImageRecord(
        file_name=file_name,
        source=source,
        read_bytes=lambda: image_bytes,
    )


def _inked_record(
    file_name: str,
    *,
    source: str,
    size: tuple[int, int],
    ink_box: tuple[int, int, int, int],
) -> ImageRecord:
    buffer = io.BytesIO()
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle(ink_box, fill="black")
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    return ImageRecord(
        file_name=file_name,
        source=source,
        read_bytes=lambda: image_bytes,
    )


def _numeric_pipe_table(keys: tuple[int, ...]) -> str:
    lines = ["| Year | Value |", "| --- | --- |"]
    lines.extend(f"| {key} | {key * 10} |" for key in keys)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    unittest.main()
