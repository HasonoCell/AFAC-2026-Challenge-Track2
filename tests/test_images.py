from __future__ import annotations

import io
import unittest

from PIL import Image

from afac_pipeline.images import make_grid_slices, make_vertical_slices


class MakeVerticalSlicesTest(unittest.TestCase):
    def test_max_width_does_not_require_slice_height(self) -> None:
        original = io.BytesIO()
        Image.new("RGB", (400, 200), "white").save(original, format="JPEG", quality=95)

        slices = make_vertical_slices(
            file_name="sample.jpg",
            image_bytes=original.getvalue(),
            max_width=100,
            slice_height=None,
        )

        self.assertEqual(len(slices), 1)
        self.assertEqual((slices[0].width, slices[0].height), (100, 50))

    def test_resized_single_slice_uses_resized_bytes(self) -> None:
        original = io.BytesIO()
        Image.new("RGB", (400, 200), "white").save(original, format="JPEG", quality=95)
        original_bytes = original.getvalue()

        slices = make_vertical_slices(
            file_name="sample.jpg",
            image_bytes=original_bytes,
            max_width=100,
            slice_height=200,
            overlap=10,
        )

        self.assertEqual(len(slices), 1)
        self.assertEqual((slices[0].width, slices[0].height), (100, 50))
        self.assertNotEqual(slices[0].image_bytes, original_bytes)

        with Image.open(io.BytesIO(slices[0].image_bytes)) as resized:
            self.assertEqual(resized.size, (100, 50))

    def test_grid_slices_record_coordinates_in_row_major_order(self) -> None:
        original = io.BytesIO()
        Image.new("RGB", (300, 220), "white").save(original, format="JPEG", quality=95)

        slices = make_grid_slices(
            file_name="sample.jpg",
            image_bytes=original.getvalue(),
            slice_width=120,
            slice_height=100,
            x_overlap=20,
            y_overlap=10,
        )

        self.assertEqual(len(slices), 9)
        self.assertEqual(slices[0].file_name, "sample_r001_c001.jpg")
        self.assertEqual((slices[0].row, slices[0].col), (1, 1))
        self.assertEqual((slices[0].rows, slices[0].cols), (3, 3))
        self.assertEqual((slices[0].x0, slices[0].x1, slices[0].y0, slices[0].y1), (0, 120, 0, 100))

        self.assertEqual((slices[1].row, slices[1].col), (1, 2))
        self.assertEqual((slices[1].x0, slices[1].x1, slices[1].y0, slices[1].y1), (100, 220, 0, 100))

        self.assertEqual(slices[-1].file_name, "sample_r003_c003.jpg")
        self.assertEqual((slices[-1].row, slices[-1].col), (3, 3))
        self.assertEqual((slices[-1].x0, slices[-1].x1, slices[-1].y0, slices[-1].y1), (200, 300, 180, 220))
        self.assertEqual((slices[-1].width, slices[-1].height), (100, 40))

        with Image.open(io.BytesIO(slices[-1].image_bytes)) as last:
            self.assertEqual(last.size, (100, 40))

    def test_horizontal_only_slices_use_column_names(self) -> None:
        original = io.BytesIO()
        Image.new("RGB", (250, 80), "white").save(original, format="JPEG", quality=95)

        slices = make_grid_slices(
            file_name="sample.jpg",
            image_bytes=original.getvalue(),
            slice_width=100,
            x_overlap=20,
        )

        self.assertEqual([image_slice.file_name for image_slice in slices], [
            "sample_col001.jpg",
            "sample_col002.jpg",
            "sample_col003.jpg",
        ])
        self.assertEqual([(image_slice.x0, image_slice.x1) for image_slice in slices], [
            (0, 100),
            (80, 180),
            (160, 250),
        ])


if __name__ == "__main__":
    unittest.main()
