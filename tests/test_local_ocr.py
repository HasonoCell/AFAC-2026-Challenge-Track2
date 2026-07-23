from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from PIL import Image

from afac_pipeline.images import ImageSlice
from afac_pipeline.local_ocr import (
    LocalOCRError,
    _ensure_root_tile_manifest,
    _parse_rapidocr_tsv,
    _serialize_rapidocr_lines,
    _split_image_slice,
    run_local_numeric_matrix_ocr,
    run_rapidocr_observations,
)


class LocalOCRTest(unittest.TestCase):
    def test_tile_manifest_rejects_same_name_different_geometry(self) -> None:
        with TemporaryDirectory() as directory:
            tile_dir = Path(directory) / "tiles"
            tile_dir.mkdir()
            first = ImageSlice("tile.jpg", b"first", 0, 10, 0, 10, 10, 10)
            changed = ImageSlice("tile.jpg", b"second", 0, 10, 0, 10, 10, 10)
            _ensure_root_tile_manifest(tile_dir, [first])
            _ensure_root_tile_manifest(tile_dir, [first])
            with self.assertRaisesRegex(LocalOCRError, "geometry"):
                _ensure_root_tile_manifest(tile_dir, [changed])

    def test_rapidocr_coordinates_are_mapped_to_the_source_page(self) -> None:
        image_slice = ImageSlice(
            file_name="tile.jpg",
            image_bytes=b"",
            x0=100,
            x1=300,
            y0=200,
            y1=400,
            width=100,
            height=100,
        )
        observations = _parse_rapidocr_tsv(
            "25\t50\t10\t20\t0.95\t123",
            image_slice,
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].x, 150)
        self.assertEqual(observations[0].y, 300)
        self.assertEqual(observations[0].width, 20)
        self.assertEqual(observations[0].height, 40)
        self.assertEqual(observations[0].text, "123")

    def test_rapidocr_serialization_uses_quad_bounds_and_sanitizes_text(self) -> None:
        text = _serialize_rapidocr_lines(
            [[[[0, 1], [4, 1], [4, 3], [0, 3]], "年\t度\n", 0.9]]
        )

        self.assertEqual(
            text,
            "2.000000\t2.000000\t4.000000\t2.000000\t0.90000000\t年 度",
        )

    def test_off_backend_returns_none_without_loading_an_engine(self) -> None:
        self.assertIsNone(
            run_local_numeric_matrix_ocr(
                backend="off",
                slices=[],
                cache_dir=None,  # type: ignore[arg-type]
            )
        )

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown local OCR backend"):
            run_local_numeric_matrix_ocr(
                backend="mystery",
                slices=[],
                cache_dir=None,  # type: ignore[arg-type]
            )

    def test_saturated_tile_children_keep_source_page_coordinates(self) -> None:
        buffer = BytesIO()
        Image.new("RGB", (100, 80), "white").save(buffer, format="JPEG")
        parent = ImageSlice(
            file_name="page_content_r001_c001.jpg",
            image_bytes=buffer.getvalue(),
            x0=1_000,
            x1=1_200,
            y0=2_000,
            y1=2_160,
            width=100,
            height=80,
        )

        children = _split_image_slice(parent)

        self.assertEqual(len(children), 4)
        self.assertEqual(
            [(item.x0, item.x1, item.y0, item.y1) for item in children],
            [
                (1_000, 1_100, 2_000, 2_080),
                (1_100, 1_200, 2_000, 2_080),
                (1_000, 1_100, 2_080, 2_160),
                (1_100, 1_200, 2_080, 2_160),
            ],
        )
        self.assertEqual(len({item.file_name for item in children}), 4)

    def test_saturated_parent_is_replaced_by_four_child_observations(self) -> None:
        buffer = BytesIO()
        Image.new("RGB", (100, 80), "white").save(buffer, format="JPEG")
        parent = ImageSlice(
            file_name="page.jpg",
            image_bytes=buffer.getvalue(),
            x0=0,
            x1=100,
            y0=0,
            y1=80,
            width=100,
            height=80,
        )
        line = [[[0, 0], [10, 0], [10, 10], [0, 10]], "1", 0.9]

        def infer(path: str):
            return ([line] if "_r" in Path(path).stem else [line] * 1000, None)

        engine = Mock(side_effect=infer)
        sentinel = Mock()
        with (
            TemporaryDirectory() as directory,
            patch("afac_pipeline.local_ocr._rapidocr_engine", return_value=engine),
            patch(
                "afac_pipeline.local_ocr.reconstruct_numeric_matrix_from_observations",
                return_value=sentinel,
            ) as reconstruct,
        ):
            result = run_local_numeric_matrix_ocr(
                backend="rapidocr",
                slices=[parent],
                cache_dir=Path(directory),
                workers=1,
                refine_saturated=True,
                max_refine_depth=1,
            )

        self.assertIs(result, sentinel)
        self.assertEqual(engine.call_count, 5)
        observations = reconstruct.call_args.args[0]
        self.assertEqual(len(observations), 4)

    def test_observation_route_reuses_ocr_without_numeric_reconstruction(self) -> None:
        buffer = BytesIO()
        Image.new("RGB", (40, 40), "white").save(buffer, format="JPEG")
        image_slice = ImageSlice("page.jpg", buffer.getvalue(), 0, 40, 0, 40, 40, 40)
        line = [[[0, 0], [20, 0], [20, 10], [0, 10]], "正文", 0.9]

        with (
            TemporaryDirectory() as directory,
            patch("afac_pipeline.local_ocr._rapidocr_engine", return_value=Mock(return_value=([line], None))),
            patch("afac_pipeline.local_ocr.reconstruct_numeric_matrix_from_observations") as reconstruct,
        ):
            observations = run_rapidocr_observations(
                slices=[image_slice],
                cache_dir=Path(directory),
                workers=1,
            )

        self.assertEqual([item.text for item in observations], ["正文"])
        reconstruct.assert_not_called()


if __name__ == "__main__":
    unittest.main()
