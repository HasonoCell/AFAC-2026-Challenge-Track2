from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from afac_pipeline.evaluation import (
    evaluate_pair,
    evaluate_prediction_csv,
    format_evaluation_summary,
    read_prediction_csv,
    write_evaluation_rows,
)


class EvaluationTest(unittest.TestCase):
    def test_identical_markdown_scores_perfectly(self) -> None:
        markdown = (
            "# Title\n\n"
            "| A | B |\n"
            "| --- | --- |\n"
            "| 1 | 2 |\n"
        )

        row = evaluate_pair(
            file_name="sample.jpg",
            prediction=markdown,
            ground_truth=markdown,
        )

        self.assertEqual(row.text_similarity, 100)
        self.assertEqual(row.markdown_similarity, 100)
        self.assertEqual(row.markdown_similarity, 100)
        self.assertEqual(row.read_order_similarity, 100)
        self.assertEqual(row.table_structure_score, 100)
        self.assertEqual(row.overall_proxy, 100)

    def test_evaluate_prediction_csv_reports_missing_and_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prediction_csv = Path(tmpdir) / "pred.csv"
            _write_prediction_rows(
                prediction_csv,
                [
                    {"file_name": "a.jpg", "ground_truth": "# A\n"},
                    {"file_name": "unknown.jpg", "ground_truth": "# extra\n"},
                ],
            )

            summary = evaluate_prediction_csv(
                prediction_csv=prediction_csv,
                ground_truths={
                    "a.jpg": "# A\n",
                    "b.jpg": "# B\n",
                },
            )

            self.assertEqual(summary.evaluated, 1)
            self.assertEqual(summary.missing_predictions, ("b.jpg",))
            self.assertEqual(summary.unknown_predictions, ("unknown.jpg",))
            self.assertIn("overall_proxy_mean: 100.0000", format_evaluation_summary(summary))

    def test_write_evaluation_rows_outputs_metrics_csv(self) -> None:
        row = evaluate_pair(
            file_name="sample.jpg",
            prediction="<table><tr><td>1</td></tr></table>\n",
            ground_truth="<table><tr><td>1</td></tr></table>\n",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = Path(tmpdir) / "eval.csv"
            write_evaluation_rows(output_csv, [row])

            with output_csv.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(rows[0]["file_name"], "sample.jpg")
        self.assertEqual(rows[0]["overall_proxy"], "100.000000")

    def test_read_prediction_csv_handles_large_markdown_cells(self) -> None:
        large_markdown = "<table>" + ("x" * 200_000) + "</table>\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            prediction_csv = Path(tmpdir) / "pred.csv"
            _write_prediction_rows(
                prediction_csv,
                [{"file_name": "a.jpg", "ground_truth": large_markdown}],
            )

            predictions = read_prediction_csv(prediction_csv)

        self.assertEqual(predictions["a.jpg"], large_markdown)

    def test_large_markdown_similarity_uses_bounded_comparison(self) -> None:
        large_markdown = "<table>" + ("x" * 50_000) + "</table>\n"

        row = evaluate_pair(
            file_name="sample.jpg",
            prediction=large_markdown,
            ground_truth=large_markdown,
        )

        self.assertEqual(row.text_similarity, 100)

    def test_large_token_similarity_uses_bounded_fingerprints(self) -> None:
        # A page full of short numeric cells used to stay below the 20k-token
        # gate and feed a quadratic SequenceMatcher.  The evaluation path
        # must cap the compared sequence even when the text has whitespace.
        token_count = 2_100
        prediction = " ".join(str(index) for index in range(token_count))
        ground_truth = prediction.replace("2099", "different")

        with patch("afac_pipeline.evaluation._sequence_matcher_ratio") as matcher:
            matcher.return_value = 88.0
            row = evaluate_pair(
                file_name="sample.jpg",
                prediction=prediction,
                ground_truth=ground_truth,
            )

        self.assertGreater(row.text_similarity, 0)
        compared_left, compared_right = matcher.call_args.args
        self.assertLessEqual(len(compared_left), 4_096)
        self.assertLessEqual(len(compared_right), 4_096)

    def test_large_table_similarity_tolerates_one_shifted_cell(self) -> None:
        cells = [f"<td>{index:05d}</td>" for index in range(5000)]
        ground_truth = "<table><tr>" + "".join(cells) + "</tr></table>\n"
        prediction = (
            "<table><tr><td>extra</td>" + "".join(cells) + "</tr></table>\n"
        )

        row = evaluate_pair(
            file_name="sample.jpg",
            prediction=prediction,
            ground_truth=ground_truth,
        )

        self.assertGreater(row.text_similarity, 99.0)

    def test_very_large_shape_aligned_table_uses_cell_text_similarity(self) -> None:
        ground_cells = [f"<td>{index:05d}</td>" for index in range(21_000)]
        predicted_cells = [
            f"<td>{'wrong' if index % 20 == 0 else f'{index:05d}'}</td>"
            for index in range(21_000)
        ]
        ground_truth = "<table><tr>" + "".join(ground_cells) + "</tr></table>"
        prediction = "<table><tr>" + "".join(predicted_cells) + "</tr></table>"

        row = evaluate_pair(
            file_name="sample.jpg",
            prediction=prediction,
            ground_truth=ground_truth,
        )

        self.assertGreater(row.text_similarity, 94.0)
        self.assertLess(row.text_similarity, 100.0)

    def test_table_structure_is_independent_of_html_or_pipe_representation(self) -> None:
        ground_truth = (
            "<table><tr><td>A</td><td>B</td></tr>"
            "<tr><td>1</td><td>2</td></tr></table>"
        )
        pipe_equivalent = (
            "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
        )

        row = evaluate_pair(
            file_name="sample.jpg",
            prediction=pipe_equivalent,
            ground_truth=ground_truth,
        )

        self.assertEqual(row.table_structure_score, 100)

    def test_table_structure_distinguishes_span_extent(self) -> None:
        ground_truth = (
            '<table><tr><td colspan="3">A</td></tr></table>'
        )
        wrong_span = (
            '<table><tr><td colspan="2">A</td></tr></table>'
        )

        row = evaluate_pair(
            file_name="sample.jpg",
            prediction=wrong_span,
            ground_truth=ground_truth,
        )

        self.assertLess(row.table_structure_score, 100)

    def test_read_order_ignores_html_table_physical_line_wrapping(self) -> None:
        ground_truth = (
            "<table><tr><td>A</td></tr><tr><td>B</td></tr></table>"
        )
        wrapped = (
            "<table>\n<tr><td>A</td></tr>\n<tr><td>B</td></tr>\n</table>"
        )

        row = evaluate_pair(
            file_name="sample.jpg",
            prediction=wrapped,
            ground_truth=ground_truth,
        )

        self.assertEqual(row.read_order_similarity, 100)


def _write_prediction_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["file_name", "ground_truth"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
