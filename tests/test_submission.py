from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from afac_pipeline.submission import (
    build_conservative_submission_ensemble,
    build_submission_overlay,
    compact_submission_for_platform,
    remerge_cached_grid_submission,
    validate_submission_csv,
)


class ValidateSubmissionCsvTest(unittest.TestCase):
    def test_compaction_rewrites_oversized_html_table_without_losing_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base.csv"
            output = root / "output.csv"
            source = (
                "title\n<table><tr><th>A</th><th>B</th></tr>"
                "<tr><td>1</td><td>2</td></tr>"
                "<tr><td>3</td><td>4</td></tr></table>\n"
            )
            _write_rows(base, [{"file_name": "sample.jpg", "ground_truth": source}])

            result = compact_submission_for_platform(
                base_csv=base,
                output_csv=output,
                max_field_bytes=80,
            )

            self.assertEqual(result.compacted_file_names, ("sample.jpg",))
            with output.open(newline="", encoding="utf-8") as file:
                compacted = next(csv.DictReader(file))["ground_truth"]
            self.assertNotIn("<table", compacted)
            self.assertIn("| A | B |", compacted)
            self.assertIn("| 3 | 4 |", compacted)
            self.assertLessEqual(len(compacted.encode("utf-8")), 80)

    def test_compaction_rejects_oversized_non_table_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base.csv"
            _write_rows(
                base,
                [{"file_name": "sample.jpg", "ground_truth": "x" * 100}],
            )

            with self.assertRaisesRegex(ValueError, "no complete HTML table"):
                compact_submission_for_platform(
                    base_csv=base,
                    output_csv=root / "output.csv",
                    max_field_bytes=80,
                )

    def test_compaction_can_rewrite_all_html_tables_below_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base.csv"
            output = root / "output.csv"
            _write_rows(
                base,
                [
                    {
                        "file_name": "table.jpg",
                        "ground_truth": "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>\n",
                    },
                    {"file_name": "text.jpg", "ground_truth": "# text\n"},
                ],
            )

            result = compact_submission_for_platform(
                base_csv=base,
                output_csv=output,
                max_field_bytes=1_000,
                compact_all_html_tables=True,
            )
            with output.open(newline="", encoding="utf-8") as file:
                compacted = {row["file_name"]: row["ground_truth"] for row in csv.DictReader(file)}

        self.assertEqual(result.compacted_count, 1)
        self.assertTrue(result.compact_all_html_tables)
        self.assertNotIn("<table", compacted["table.jpg"].lower())
        self.assertEqual(compacted["text.jpg"], "# text\n")

    def test_compaction_can_leave_oversized_non_table_markdown_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base.csv"
            output = root / "output.csv"
            source = "# long document\n" + "text " * 100
            _write_rows(base, [{"file_name": "plain.jpg", "ground_truth": source}])

            result = compact_submission_for_platform(
                base_csv=base,
                output_csv=output,
                max_field_bytes=50,
                allow_non_table_oversize=True,
            )
            with output.open(newline="", encoding="utf-8") as file:
                rewritten = next(csv.DictReader(file))["ground_truth"]

        self.assertEqual(result.compacted_count, 0)
        self.assertEqual(rewritten, source)

    def test_compaction_can_treat_budget_as_html_complexity_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base.csv"
            output = root / "output.csv"
            source = "<table><tr><th>A</th><th>B</th></tr>" + "<tr><td>one</td><td>two</td></tr>" * 30 + "</table>"
            _write_rows(base, [{"file_name": "table.jpg", "ground_truth": source}])

            result = compact_submission_for_platform(
                base_csv=base,
                output_csv=output,
                max_field_bytes=20,
                allow_compacted_oversize=True,
            )
            with output.open(newline="", encoding="utf-8") as file:
                rewritten = next(csv.DictReader(file))["ground_truth"]

        self.assertEqual(result.compacted_count, 1)
        self.assertGreater(len(rewritten.encode("utf-8")), 20)
        self.assertNotIn("<table", rewritten)

    def test_remerge_cached_grid_preserves_rows_and_cells_while_joining_bands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base.csv"
            output = root / "output.csv"
            cache = root / "cache" / "sample"
            cache.mkdir(parents=True)
            base_markdown = (
                "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>\n"
                "<table><tr><th>B</th></tr><tr><td>2</td></tr></table>\n"
            )
            _write_rows(base, [{"file_name": "sample.jpg", "ground_truth": base_markdown}])
            (cache / "sample_content_r001_c001.md").write_text(
                "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>",
                encoding="utf-8",
            )
            (cache / "sample_content_r002_c001.md").write_text(
                "<table><tr><th>B</th></tr><tr><td>2</td></tr></table>",
                encoding="utf-8",
            )

            result = remerge_cached_grid_submission(
                base_csv=base,
                cache_roots=[root / "cache"],
                output_csv=output,
                rows=2,
                cols=1,
                min_success_parts=2,
                min_success_ratio=1.0,
            )

            self.assertEqual(result.remerged_file_names, ("sample.jpg",))
            with output.open(newline="", encoding="utf-8") as file:
                markdown = next(csv.DictReader(file))["ground_truth"]
            self.assertEqual(markdown.count("<table"), 1)
            self.assertEqual(markdown.count("<tr"), 4)
            self.assertEqual(markdown.count("<td") + markdown.count("<th"), 4)

    def test_remerge_cached_grid_allows_only_blank_rectangular_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base.csv"
            output = root / "output.csv"
            cache = root / "cache" / "sample"
            cache.mkdir(parents=True)
            base_markdown = (
                "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>\n"
                "<table><tr><th>A</th><th>B</th></tr>"
                "<tr><td>2</td><td>x</td></tr></table>\n"
            )
            _write_rows(base, [{"file_name": "sample.jpg", "ground_truth": base_markdown}])
            (cache / "sample_content_r001_c001.md").write_text(
                "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>",
                encoding="utf-8",
            )
            (cache / "sample_content_r002_c001.md").write_text(
                "<table><tr><th>A</th><th>B</th></tr>"
                "<tr><td>2</td><td>x</td></tr></table>",
                encoding="utf-8",
            )

            result = remerge_cached_grid_submission(
                base_csv=base,
                cache_roots=[root / "cache"],
                output_csv=output,
                rows=2,
                cols=1,
                min_success_parts=2,
                min_success_ratio=1.0,
            )

            self.assertEqual(result.remerged_file_names, ("sample.jpg",))
            with output.open(newline="", encoding="utf-8") as file:
                markdown = next(csv.DictReader(file))["ground_truth"]
            self.assertEqual(markdown.count("<table"), 1)
            self.assertIn("<tr><td>1</td><td></td></tr>", markdown)

    def test_remerge_cached_grid_can_recover_td_only_header_convention(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base.csv"
            output = root / "output.csv"
            cache = root / "cache" / "sample"
            cache.mkdir(parents=True)
            _write_rows(
                base,
                [
                    {
                        "file_name": "sample.jpg",
                        "ground_truth": (
                            "<table><tr><th>A</th></tr>"
                            "<tr><td>1</td></tr></table>"
                        ),
                    }
                ],
            )
            (cache / "sample_content_r001_c001.md").write_text(
                "<table><tr><td>A</td></tr><tr><td>1</td></tr></table>",
                encoding="utf-8",
            )

            result = remerge_cached_grid_submission(
                base_csv=base,
                cache_roots=[root / "cache"],
                output_csv=output,
                rows=1,
                cols=1,
                min_success_parts=1,
                min_success_ratio=1.0,
            )

            self.assertEqual(result.remerged_file_names, ("sample.jpg",))
            with output.open(newline="", encoding="utf-8") as file:
                markdown = next(csv.DictReader(file))["ground_truth"]
            self.assertNotIn("<th>", markdown)
            self.assertIn("<tr><td>A</td></tr>", markdown)

    def test_accepts_complete_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.csv"
            _write_rows(
                path,
                [
                    {"file_name": "a.jpg", "ground_truth": "<table></table>\n"},
                    {"file_name": "b.jpg", "ground_truth": "# ok\n"},
                ],
            )

            result = validate_submission_csv(
                submission_csv=path,
                expected_file_names=["a.jpg", "b.jpg"],
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.row_count, 2)

    def test_rejects_missing_extra_empty_and_malformed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.csv"
            _write_rows(
                path,
                [
                    {"file_name": "a.jpg", "ground_truth": ""},
                    {"file_name": "extra.jpg", "ground_truth": "<table>"},
                    {"file_name": "a.jpg", "ground_truth": "ERROR: failed"},
                    {"file_name": "dry.jpg", "ground_truth": "MVP dry-run placeholder."},
                ],
            )

            result = validate_submission_csv(
                submission_csv=path,
                expected_file_names=["a.jpg", "b.jpg", "dry.jpg"],
            )

            self.assertFalse(result.ok)
            joined = "\n".join(result.errors)
            self.assertIn("duplicate file_name", joined)
            self.assertIn("missing expected A-list files", joined)
            self.assertIn("unknown file_name", joined)
            self.assertIn("empty ground_truth", joined)
            self.assertIn("ERROR markers", joined)
            self.assertIn("HTML table structure has unclosed <table>", joined)
            self.assertIn("dry-run placeholder", joined)

    def test_b_label_is_used_for_dataset_specific_validation_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.csv"
            _write_rows(path, [{"file_name": "b-only.jpg", "ground_truth": "# ok\n"}])

            result = validate_submission_csv(
                submission_csv=path,
                expected_file_names=["b-only.jpg", "b-other.jpg"],
                expected_label="B-list",
            )

        joined = "\n".join(result.errors)
        self.assertIn("missing expected B-list files", joined)
        self.assertIn("current B-list data", joined)
        self.assertNotIn("A-list", joined)

    def test_allow_empty_downgrades_empty_outputs_to_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.csv"
            _write_rows(path, [{"file_name": "a.jpg", "ground_truth": ""}])

            result = validate_submission_csv(
                submission_csv=path,
                expected_file_names=["a.jpg"],
                allow_empty=True,
            )

            self.assertTrue(result.ok)
            self.assertIn("empty ground_truth", "\n".join(result.warnings))

    def test_rejects_wrong_header_and_oversized_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.csv"
            path.write_text("file_name,prediction\na.jpg,ok\n", encoding="utf-8")

            result = validate_submission_csv(
                submission_csv=path,
                expected_file_names=["a.jpg"],
                max_size_bytes=1,
            )

            self.assertFalse(result.ok)
            joined = "\n".join(result.errors)
            self.assertIn("exceeding limit", joined)
            self.assertIn("CSV header must be exactly", joined)

    def test_accepts_large_ground_truth_cells(self) -> None:
        large_markdown = "<table>" + ("x" * 200_000) + "</table>\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.csv"
            _write_rows(
                path,
                [{"file_name": "a.jpg", "ground_truth": large_markdown}],
            )

            result = validate_submission_csv(
                submission_csv=path,
                expected_file_names=["a.jpg"],
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.row_count, 1)

    def test_optional_per_field_byte_budget_rejects_oversized_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.csv"
            _write_rows(
                path,
                [{"file_name": "a.jpg", "ground_truth": "汉" * 100}],
            )

            result = validate_submission_csv(
                submission_csv=path,
                expected_file_names=["a.jpg"],
                max_field_bytes=200,
            )

        self.assertFalse(result.ok)
        self.assertIn("per-field UTF-8 byte budget", "\n".join(result.errors))

    def test_rejects_unclosed_html_table_rows_and_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.csv"
            _write_rows(
                path,
                [
                    {
                        "file_name": "a.jpg",
                        "ground_truth": "<table><tr><td>broken</td></table>",
                    }
                ],
            )

            result = validate_submission_csv(
                submission_csv=path,
                expected_file_names=["a.jpg"],
            )

        self.assertFalse(result.ok)
        self.assertIn("HTML table structure", "\n".join(result.errors))

    def test_conservative_ensemble_replaces_only_collapsed_table_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            primary = root / "primary.csv"
            fallback = root / "fallback.csv"
            output = root / "ensemble.csv"
            _write_rows(
                primary,
                [
                    {"file_name": "collapsed.jpg", "ground_truth": "bad"},
                    {"file_name": "short-but-not-collapsed.jpg", "ground_truth": "x" * 30},
                    {"file_name": "normal.jpg", "ground_truth": "# primary output that remains primary\n"},
                ],
            )
            _write_rows(
                fallback,
                [
                    {"file_name": "collapsed.jpg", "ground_truth": "<table><tr><td>fallback</td></tr></table>" * 20},
                    {"file_name": "short-but-not-collapsed.jpg", "ground_truth": "<table></table>" * 20},
                    {"file_name": "normal.jpg", "ground_truth": "<table></table>" * 20},
                ],
            )

            result = build_conservative_submission_ensemble(
                primary_csv=primary,
                fallback_csv=fallback,
                output_csv=output,
                max_primary_chars=10,
                max_primary_to_fallback_ratio=0.10,
            )

            self.assertEqual(result.fallback_file_names, ("collapsed.jpg",))
            with output.open(encoding="utf-8", newline="") as file:
                rows = {row["file_name"]: row["ground_truth"] for row in csv.DictReader(file)}
            self.assertIn("fallback", rows["collapsed.jpg"])
            self.assertEqual(rows["short-but-not-collapsed.jpg"], "x" * 30)
            self.assertEqual(rows["normal.jpg"], "# primary output that remains primary\n")

    def test_overlay_replaces_only_partial_rows_in_base_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base.csv"
            override = root / "override.csv"
            output = root / "overlay.csv"
            _write_rows(
                base,
                [
                    {"file_name": "a.jpg", "ground_truth": "base a"},
                    {"file_name": "b.jpg", "ground_truth": "base b"},
                ],
            )
            _write_rows(override, [{"file_name": "b.jpg", "ground_truth": "repaired b"}])

            result = build_submission_overlay(
                base_csv=base,
                override_csvs=[override],
                output_csv=output,
            )

            self.assertEqual(result.override_file_names, ("b.jpg",))
            with output.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["file_name"] for row in rows], ["a.jpg", "b.jpg"])
            self.assertEqual(rows[0]["ground_truth"], "base a")
            self.assertEqual(rows[1]["ground_truth"], "repaired b")

    def test_overlay_can_reject_shorter_nondeterministic_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base.csv"
            override = root / "override.csv"
            output = root / "overlay.csv"
            _write_rows(base, [{"file_name": "a.jpg", "ground_truth": "x" * 100}])
            _write_rows(override, [{"file_name": "a.jpg", "ground_truth": "x" * 99}])

            result = build_submission_overlay(
                base_csv=base,
                override_csvs=[override],
                output_csv=output,
                min_override_to_base_ratio=1.0,
            )

            self.assertEqual(result.override_count, 0)
            self.assertEqual(result.skipped_file_names, ("a.jpg",))
            with output.open(encoding="utf-8", newline="") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["ground_truth"], "x" * 100)

    def test_overlay_can_reject_trivial_character_gain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base.csv"
            override = root / "override.csv"
            output = root / "overlay.csv"
            _write_rows(base, [{"file_name": "a.jpg", "ground_truth": "x" * 100}])
            _write_rows(override, [{"file_name": "a.jpg", "ground_truth": "x" * 101}])

            result = build_submission_overlay(
                base_csv=base,
                override_csvs=[override],
                output_csv=output,
                min_override_to_base_ratio=1.0,
                min_override_char_gain=10,
            )

            self.assertEqual(result.override_count, 0)
            self.assertEqual(result.skipped_file_names, ("a.jpg",))

    def test_overlay_rejects_malformed_table_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base.csv"
            override = root / "override.csv"
            output = root / "overlay.csv"
            _write_rows(
                base,
                [{"file_name": "a.jpg", "ground_truth": "<table><tr><td>safe</td></tr></table>"}],
            )
            _write_rows(
                override,
                [{"file_name": "a.jpg", "ground_truth": "<table><tr><td>broken</td></table>"}],
            )

            result = build_submission_overlay(
                base_csv=base,
                override_csvs=[override],
                output_csv=output,
            )

            self.assertEqual(result.override_count, 0)
            self.assertEqual(result.skipped_file_names, ("a.jpg",))
            with output.open(encoding="utf-8", newline="") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["ground_truth"], "<table><tr><td>safe</td></tr></table>")

    def test_overlay_rejects_runaway_duplicate_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base.csv"
            override = root / "override.csv"
            output = root / "overlay.csv"
            _write_rows(base, [{"file_name": "a.jpg", "ground_truth": "base"}])
            _write_rows(
                override,
                [{"file_name": "a.jpg", "ground_truth": "<table></table>\n" + ("repeat\n" * 100)}],
            )

            result = build_submission_overlay(
                base_csv=base,
                override_csvs=[override],
                output_csv=output,
                max_override_duplicate_line_ratio=0.90,
            )

            self.assertEqual(result.override_count, 0)
            self.assertEqual(result.skipped_file_names, ("a.jpg",))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["file_name", "ground_truth"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
