from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from afac_pipeline.cli import build_parser, cmd_baseline_submit, cmd_evaluate_train


class CliTest(unittest.TestCase):
    def test_evaluate_train_subset_reads_only_matching_labels(self) -> None:
        args = SimpleNamespace(
            prediction_csv=Path("/tmp/predictions.csv"),
            raw_dir=Path("/tmp/raw"),
            output_csv=None,
            allow_subset=True,
            worst_k=3,
        )
        included = SimpleNamespace(
            file_name="included.jpg",
            read_text=lambda: "# included\n",
        )

        def unexpected_read() -> str:
            raise AssertionError("unmatched training label should not be read")

        skipped = SimpleNamespace(file_name="skipped.jpg", read_text=unexpected_read)
        expected_summary = SimpleNamespace()
        with (
            patch(
                "afac_pipeline.cli.read_prediction_csv",
                return_value={"included.jpg": "# prediction\n"},
            ),
            patch(
                "afac_pipeline.cli.iter_train_markdowns",
                return_value=[included, skipped],
            ),
            patch(
                "afac_pipeline.cli.evaluate_prediction_csv",
                return_value=expected_summary,
            ) as evaluate_mock,
            patch(
                "afac_pipeline.cli.format_evaluation_summary",
                return_value="summary",
            ),
            redirect_stdout(io.StringIO()),
        ):
            cmd_evaluate_train(args)

        self.assertEqual(
            evaluate_mock.call_args.kwargs["ground_truths"],
            {"included.jpg": "# included\n"},
        )

    def test_submission_commands_accept_b_dataset(self) -> None:
        parser = build_parser()

        self.assertEqual(
            parser.parse_args(["predict", "--dataset", "b", "--dry-run"]).dataset,
            "b",
        )
        self.assertEqual(
            parser.parse_args(["baseline-submit", "--dataset", "b"]).dataset,
            "b",
        )
        preset_args = parser.parse_args(
            ["baseline-submit", "--dataset", "b", "--preset", "b-generalization-v1"]
        )
        self.assertEqual(preset_args.preset, "b-generalization-v1")
        v2_args = parser.parse_args(
            ["baseline-submit", "--dataset", "b", "--preset", "b-generalization-v2"]
        )
        self.assertEqual(v2_args.preset, "b-generalization-v2")
        v3_args = parser.parse_args(
            ["baseline-submit", "--dataset", "b", "--preset", "b-generalization-v3"]
        )
        self.assertEqual(v3_args.preset, "b-generalization-v3")
        experiment_preset_args = parser.parse_args(
            [
                "experiment-train",
                "--preset",
                "b-generalization-v1",
                "--file-name",
                "sample.jpg",
            ]
        )
        self.assertEqual(experiment_preset_args.preset, "b-generalization-v1")
        self.assertEqual(experiment_preset_args.file_name, ["sample.jpg"])
        self.assertIsNone(experiment_preset_args.table_refine_max_depth)
        self.assertIsNone(experiment_preset_args.table_repair_workers)
        experiment_refine_args = parser.parse_args(
            [
                "experiment-train",
                "--preset",
                "b-generalization-v1",
                "--table-refine-max-depth",
                "1",
                "--table-repair-workers",
                "2",
            ]
        )
        self.assertEqual(experiment_refine_args.table_refine_max_depth, 1)
        self.assertEqual(experiment_refine_args.table_repair_workers, 2)
        self.assertEqual(
            parser.parse_args(
                [
                    "validate-submission",
                    "--dataset",
                    "b",
                    "--submission-csv",
                    "/tmp/b.csv",
                ]
            ).dataset,
            "b",
        )
        compact_args = parser.parse_args(
            [
                "compact-submission",
                "--base-csv",
                "/tmp/base.csv",
                "--output-csv",
                "/tmp/out.csv",
                "--max-field-bytes",
                "200000",
                "--all-html-tables",
            ]
        )
        self.assertTrue(compact_args.all_html_tables)

    def test_baseline_submit_accepts_train_dataset(self) -> None:
        args = SimpleNamespace(
            dataset="train",
            raw_dir=Path("/tmp/raw"),
            output_csv=Path("/tmp/output.csv"),
            cache_dir=Path("/tmp/cache"),
            user_id="finixB2002",
            api_key="test-key",
            timeout=12.0,
            offset=3,
            limit=5,
            crop_sizes="800,600",
            anchors="top_left,center",
            jpeg_quality=95,
            sleep=0.0,
            no_resume=False,
            retries=1,
            retry_sleep=60.0,
            min_chars=20,
            table_repair_min_chars=600,
            table_repair_min_gain=300,
            table_repair_grid="4x4",
            table_repair_overlap=120,
            table_repair_content_threshold=245,
            table_repair_content_scale=0.04,
            table_repair_content_padding=200,
            table_repair_min_content_pixels=1000,
            table_repair_min_content_ratio=0.001,
            table_repair_header_context_height=0,
            table_repair_left_context_width=0,
            table_repair_min_success_parts=4,
            long_slice_height=12000,
            long_slice_overlap=400,
            long_min_chars=20,
            on_error="raise",
            errors_csv=Path("/tmp/errors.csv"),
        )

        fake_record = SimpleNamespace(file_name="train_sample.jpg")
        fake_stats = SimpleNamespace(
            total_discovered=1,
            processed=1,
            cache_hits=0,
            api_calls=0,
            fallbacks=0,
            template_missing=0,
            output_csv=args.output_csv,
        )

        with (
            patch("afac_pipeline.cli.iter_dataset_images", return_value=[fake_record]) as iter_mock,
            patch("afac_pipeline.cli.FinixDocClient", return_value=object()) as client_mock,
            patch("afac_pipeline.cli.run_baseline_submission", return_value=fake_stats) as run_mock,
            redirect_stdout(io.StringIO()),
        ):
            cmd_baseline_submit(args)

        iter_mock.assert_called_once_with(args.raw_dir, "train")
        client_mock.assert_called_once_with(
            user_id=args.user_id,
            api_key=args.api_key,
            timeout=args.timeout,
        )
        self.assertEqual(run_mock.call_args.kwargs["records"], [fake_record])
        self.assertEqual(
            run_mock.call_args.kwargs["config"].cache_dir,
            args.cache_dir / "train.baseline",
        )


if __name__ == "__main__":
    unittest.main()
