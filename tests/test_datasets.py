from __future__ import annotations

import csv
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from afac_pipeline.datasets import inspect_raw_data, iter_dataset_images


class DatasetDiscoveryTest(unittest.TestCase):
    def test_iter_dataset_images_reads_extracted_directory_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            _write_image(
                raw_dir / "AFAC A榜评测数据集(2)" / "finix_huge_long_rest_A" / "images" / "long.jpg",
            )
            _write_image(
                raw_dir / "AFAC A榜评测数据集(2)" / "finix_huge_table_rest_A" / "images" / "table.jpg",
            )

            records = list(iter_dataset_images(raw_dir, "a"))

            self.assertCountEqual([record.file_name for record in records], ["long.jpg", "table.jpg"])
            self.assertTrue(all(record.source.startswith(str(raw_dir)) for record in records))

    def test_iter_dataset_images_falls_back_to_zip_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            zip_path = raw_dir / "AFAC A榜评测数据集(2).zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("finix_huge_long_rest_A/images/long.jpg", _image_bytes())
                zf.writestr("finix_huge_table_rest_A/images/table.jpg", _image_bytes())

            records = list(iter_dataset_images(raw_dir, "a"))

            self.assertCountEqual([record.file_name for record in records], ["long.jpg", "table.jpg"])
            self.assertTrue(all(record.source.startswith(zip_path.name) for record in records))

    def test_inspect_raw_data_counts_extracted_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            train_root = raw_dir / "AFAC 训练数据集"
            a_root = raw_dir / "AFAC A榜评测数据集(2)"

            _write_image(train_root / "finixdocbench_huge_long_100" / "images" / "long.jpg")
            _write_image(train_root / "finixdocbench_huge_table_100" / "images" / "table.jpg")
            _write_text(train_root / "finixdocbench_huge_long_100" / "mds" / "long.md", "# long")
            _write_text(train_root / "finixdocbench_huge_table_100" / "mds" / "table.md", "# table")
            _write_text(
                train_root / "finixdocbench_huge_long_100" / "id_mapping.csv",
                "file_name,ground_truth\nlong.jpg,gt\n",
            )
            _write_text(
                train_root / "finixdocbench_huge_table_100" / "id_mapping.csv",
                "file_name,ground_truth\ntable.jpg,gt\n",
            )
            _write_image(a_root / "finix_huge_long_rest_A" / "images" / "a-long.jpg")
            _write_image(a_root / "finix_huge_table_rest_A" / "images" / "a-table.jpg")
            _write_text(raw_dir / "finix_ab_A_submit_mock.csv", "file_name,ground_truth\na-long.jpg,gt\n")

            summary = inspect_raw_data(raw_dir)

            self.assertEqual(summary["train_images"], 2)
            self.assertEqual(summary["train_mds"], 2)
            self.assertEqual(summary["train_mapping_rows"], 2)
            self.assertEqual(summary["a_images"], 2)
            self.assertEqual(summary["mock_rows"], 1)


def _image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_image_bytes())


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
