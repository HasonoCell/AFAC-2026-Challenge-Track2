from __future__ import annotations

import csv
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from afac_pipeline.datasets import ImageRecord
from afac_pipeline.pipeline import PredictionConfig, run_prediction


class _FakeClient:
    def call_with_file(self, file_name: str, file_bytes: bytes) -> str:
        if "_col001" in file_name:
            return "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
        if "_col002" in file_name:
            return "| C | D |\n| --- | --- |\n| 3 | 4 |\n"
        return "| X | Y |\n| --- | --- |\n| 9 | 9 |\n"


class _CountingClient:
    def __init__(self) -> None:
        self.calls = 0

    def call_with_file(self, file_name: str, file_bytes: bytes) -> str:
        self.calls += 1
        return f"# {file_name}\n\ncall {self.calls}\n"


class PredictionPipelineTest(unittest.TestCase):
    def test_run_prediction_reconstructs_horizontal_tables(self) -> None:
        original = io.BytesIO()
        Image.new("RGB", (200, 80), "white").save(original, format="JPEG", quality=95)
        record = ImageRecord(
            file_name="sample.jpg",
            source="memory",
            read_bytes=lambda: original.getvalue(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_csv = root / "submission.csv"
            cache_dir = root / "cache"

            stats = run_prediction(
                records=[record],
                client=_FakeClient(),
                config=PredictionConfig(
                    output_csv=output_csv,
                    cache_dir=cache_dir,
                    slice_width=100,
                    slice_x_overlap=0,
                    on_error="raise",
                ),
            )

            self.assertEqual(stats.processed, 1)
            self.assertEqual(stats.api_calls, 2)

            with output_csv.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["ground_truth"],
                "| A | B | C | D |\n"
                "| --- | --- | --- | --- |\n"
                "| 1 | 2 | 3 | 4 |\n",
            )

    def test_cache_is_separated_by_image_processing_config(self) -> None:
        original = io.BytesIO()
        Image.new("RGB", (200, 80), "white").save(original, format="JPEG", quality=95)
        record = ImageRecord(
            file_name="sample.jpg",
            source="memory",
            read_bytes=lambda: original.getvalue(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            client = _CountingClient()

            first_stats = run_prediction(
                records=[record],
                client=client,
                config=PredictionConfig(
                    output_csv=root / "first.csv",
                    cache_dir=cache_dir,
                    on_error="raise",
                ),
            )
            second_stats = run_prediction(
                records=[record],
                client=client,
                config=PredictionConfig(
                    output_csv=root / "second.csv",
                    cache_dir=cache_dir,
                    on_error="raise",
                ),
            )
            resized_stats = run_prediction(
                records=[record],
                client=client,
                config=PredictionConfig(
                    output_csv=root / "resized.csv",
                    cache_dir=cache_dir,
                    max_width=100,
                    on_error="raise",
                ),
            )

            self.assertEqual(first_stats.api_calls, 1)
            self.assertEqual(second_stats.api_calls, 0)
            self.assertEqual(second_stats.cache_hits, 1)
            self.assertEqual(resized_stats.api_calls, 1)
            self.assertEqual(resized_stats.cache_hits, 0)
            self.assertEqual(client.calls, 2)
            self.assertEqual(len(list(cache_dir.rglob("sample.md"))), 2)


if __name__ == "__main__":
    unittest.main()
