from __future__ import annotations

import csv
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from afac_pipeline.baseline import BaselineConfig, run_baseline_submission
from afac_pipeline.datasets import ImageRecord


class _FakeClient:
    def __init__(
        self,
        fail_first: bool = False,
        malformed_first: bool = False,
        short_first: bool = False,
    ) -> None:
        self.calls: list[str] = []
        self.fail_first = fail_first
        self.malformed_first = malformed_first
        self.short_first = short_first

    def call_with_file(self, file_name: str, file_bytes: bytes, **kwargs: object) -> str:
        self.calls.append(file_name)
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("first crop failed")
        if self.malformed_first and len(self.calls) == 1:
            return "<table>"
        if self.short_first and len(self.calls) == 1:
            return "too short"
        if "_content_" in file_name:
            return "<table><tr><td>repair result with enough content</td></tr></table>\n"
        if "_part" in file_name:
            return f"# {file_name}\n\nslice text\n"
        return "<table><tr><td>ok</td></tr></table>\n"


class BaselineSubmissionTest(unittest.TestCase):
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
            size=(40, 160),
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
                    min_chars=1,
                    table_repair_min_chars=20,
                    table_repair_min_gain=0,
                    table_repair_rows=1,
                    table_repair_cols=1,
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
            self.assertIn("repair result with enough content", rows[0]["ground_truth"])


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


if __name__ == "__main__":
    unittest.main()
