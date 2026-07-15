from __future__ import annotations

import io
import unittest

from PIL import Image

from afac_pipeline.images import make_vertical_slices


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


if __name__ == "__main__":
    unittest.main()
