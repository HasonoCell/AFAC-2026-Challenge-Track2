from __future__ import annotations

import csv
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from afac_pipeline.datasets import ImageRecord
from afac_pipeline.merge import (
    coalesce_adjacent_html_tables,
    merge_sliced_markdown,
    merge_markdown_parts_legacy,
)
from afac_pipeline.images import ImageSlice
from afac_pipeline.pipeline import (
    PredictionConfig,
    run_prediction,
)


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
    def test_legacy_vertical_merge_removes_line_overlap(self) -> None:
        self.assertEqual(
            merge_markdown_parts_legacy(["title\nbody\n", "body\nnext\n"]),
            "title\nbody\nnext\n",
        )

    def test_coalesces_only_adjacent_same_width_html_tables(self) -> None:
        first = (
            '<table><tr><td rowspan="2">第1组：A</td><td>B</td></tr>'
            '<tr><td>C</td></tr></table>'
        )
        second = '<table><tr><td>第2组：D</td><td>E</td></tr></table>'

        merged = coalesce_adjacent_html_tables(first + "\n\n" + second)

        self.assertEqual(merged.lower().count('<table'), 1)
        self.assertIn('rowspan="2"', merged)
        self.assertLess(merged.index('>C<'), merged.index('第2组：D'))

    def test_html_table_merge_preserves_plain_edge_slices(self) -> None:
        slices = [
            ImageSlice(
                file_name=f"part{index}.jpg",
                image_bytes=b"",
                x0=0,
                x1=1,
                y0=index,
                y1=index + 1,
                width=1,
                height=1,
                row=index + 1,
                col=1,
                rows=3,
                cols=1,
            )
            for index in range(3)
        ]
        parts = [
            "# Leading title\n\nprose before the table",
            "<table><tr><td>A</td></tr></table>",
            "prose after the table",
        ]

        merged = merge_sliced_markdown(slices, parts)

        self.assertIn("# Leading title", merged)
        self.assertIn("prose before the table", merged)
        self.assertIn("<td>A</td>", merged)
        self.assertIn("prose after the table", merged)
        self.assertLess(merged.index("prose before"), merged.index("<td>A"))
        self.assertLess(merged.index("<td>A"), merged.index("prose after"))

    def test_keeps_adjacent_tables_separate_when_prose_or_width_differs(self) -> None:
        two_columns = '<table><tr><td>第1组：A</td><td>B</td></tr></table>'
        next_two_columns = '<table><tr><td>第2组：C</td><td>D</td></tr></table>'
        three_columns = (
            '<table><tr><td>第2组：C</td><td>D</td><td>E</td></tr></table>'
        )

        with_heading = coalesce_adjacent_html_tables(
            two_columns + '\n## Next table\n' + next_two_columns
        )
        different_width = coalesce_adjacent_html_tables(
            two_columns + '\n' + three_columns
        )

        self.assertEqual(with_heading.lower().count('<table'), 2)
        self.assertEqual(different_width.lower().count('<table'), 2)

    def test_keeps_unnumbered_adjacent_same_width_tables_separate(self) -> None:
        basic = '<table><tr><td>基本部分</td><td>A</td></tr></table>'
        optional = '<table><tr><td>可选部分</td><td>B</td></tr></table>'

        merged = coalesce_adjacent_html_tables(basic + '\n' + optional)

        self.assertEqual(merged.lower().count('<table'), 2)

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
