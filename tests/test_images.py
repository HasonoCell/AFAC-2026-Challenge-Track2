from __future__ import annotations

import io
import unittest

from PIL import Image, ImageDraw

from afac_pipeline.images import (
    _decode_sampled_grayscale,
    detect_content_box,
    image_dimensions,
    make_content_grid_slices,
    make_grid_slices,
    make_vertical_slices,
    measure_text_density,
    measure_content_density,
)


class MakeVerticalSlicesTest(unittest.TestCase):
    def test_image_dimensions_do_not_require_pixel_decode(self) -> None:
        encoded = io.BytesIO()
        Image.new("RGB", (321, 123), "white").save(encoded, format="PNG")

        self.assertEqual(
            image_dimensions(image_bytes=encoded.getvalue()),
            (321, 123),
        )

    def test_jpeg_profile_sample_uses_decoder_level_downsampling(self) -> None:
        encoded = io.BytesIO()
        Image.new("RGB", (2048, 1536), "white").save(
            encoded,
            format="JPEG",
            quality=95,
        )

        with Image.open(io.BytesIO(encoded.getvalue())) as image:
            original_pixels = image.width * image.height
            sample = _decode_sampled_grayscale(image, width=128, height=96)
            decoder_pixels = image.width * image.height

        self.assertEqual(sample.mode, "L")
        self.assertEqual(sample.size, (128, 96))
        self.assertLess(decoder_pixels, original_pixels)

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

    def test_detect_content_box_ignores_white_margins(self) -> None:
        original = io.BytesIO()
        image = Image.new("RGB", (100, 80), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 30, 79, 69), fill="black")
        image.save(original, format="PNG")

        box = detect_content_box(
            image_bytes=original.getvalue(),
            threshold=250,
            sample_scale=1.0,
            padding=0,
        )

        self.assertEqual((box.x0, box.y0, box.x1, box.y1), (20, 30, 80, 70))

    def test_content_grid_slices_cover_detected_content_box(self) -> None:
        original = io.BytesIO()
        image = Image.new("RGB", (100, 80), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 30, 79, 69), fill="black")
        image.save(original, format="PNG")

        slices = make_content_grid_slices(
            file_name="sample.jpg",
            image_bytes=original.getvalue(),
            rows=2,
            cols=2,
            threshold=250,
            sample_scale=1.0,
            padding=0,
            x_overlap=5,
            y_overlap=5,
        )

        self.assertEqual([image_slice.file_name for image_slice in slices], [
            "sample_content_r001_c001.jpg",
            "sample_content_r001_c002.jpg",
            "sample_content_r002_c001.jpg",
            "sample_content_r002_c002.jpg",
        ])
        self.assertEqual((slices[0].x0, slices[0].x1, slices[0].y0, slices[0].y1), (20, 55, 30, 55))
        self.assertEqual((slices[-1].x0, slices[-1].x1, slices[-1].y0, slices[-1].y1), (45, 80, 45, 70))
        self.assertIsNotNone(slices[0].content_pixels)
        self.assertGreater(slices[0].content_pixels, 0)
        self.assertIsNotNone(slices[0].content_ratio)
        self.assertGreater(slices[0].content_ratio, 0)

    def test_content_grid_can_snap_edges_to_nearby_ruling_lines(self) -> None:
        original = io.BytesIO()
        image = Image.new("RGB", (400, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 399, 199), outline="black", width=1)
        draw.line((210, 0, 210, 199), fill="black", width=2)
        draw.line((0, 110, 399, 110), fill="black", width=2)
        image.save(original, format="PNG")

        slices = make_content_grid_slices(
            file_name="sample.jpg",
            image_bytes=original.getvalue(),
            rows=2,
            cols=2,
            threshold=250,
            sample_scale=1.0,
            padding=0,
            snap_boundaries=True,
        )

        self.assertEqual(slices[0].x1, 210)
        self.assertEqual(slices[1].x0, 210)
        self.assertEqual(slices[0].y1, 110)
        self.assertEqual(slices[2].y0, 110)

    def test_content_grid_can_snap_only_vertical_edges(self) -> None:
        original = io.BytesIO()
        image = Image.new("RGB", (400, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 399, 199), outline="black", width=1)
        draw.line((210, 0, 210, 199), fill="black", width=2)
        draw.line((0, 110, 399, 110), fill="black", width=2)
        image.save(original, format="PNG")

        slices = make_content_grid_slices(
            file_name="sample.jpg",
            image_bytes=original.getvalue(),
            rows=2,
            cols=2,
            threshold=250,
            sample_scale=1.0,
            padding=0,
            snap_x_boundaries=True,
        )

        self.assertEqual(slices[0].x1, 210)
        self.assertEqual(slices[1].x0, 210)
        self.assertEqual(slices[0].y1, 100)
        self.assertEqual(slices[2].y0, 100)

    def test_measure_content_density_distinguishes_blank_and_ink(self) -> None:
        blank = io.BytesIO()
        Image.new("RGB", (20, 20), "white").save(blank, format="PNG")

        inked = io.BytesIO()
        image = Image.new("RGB", (20, 20), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 9, 9), fill="black")
        image.save(inked, format="PNG")

        blank_pixels, blank_ratio = measure_content_density(
            image_bytes=blank.getvalue(),
            threshold=250,
        )
        ink_pixels, ink_ratio = measure_content_density(
            image_bytes=inked.getvalue(),
            threshold=250,
        )

        self.assertEqual(blank_pixels, 0)
        self.assertEqual(blank_ratio, 0)
        self.assertGreater(ink_pixels, blank_pixels)
        self.assertGreater(ink_ratio, blank_ratio)

    def test_measure_text_density_ignores_ruling_lines(self) -> None:
        ruled = io.BytesIO()
        image = Image.new("RGB", (100, 100), "white")
        draw = ImageDraw.Draw(image)
        for offset in range(0, 101, 10):
            draw.line((0, offset, 99, offset), fill="black")
            draw.line((offset, 0, offset, 99), fill="black")
        image.save(ruled, format="PNG")

        text = io.BytesIO()
        image = Image.open(io.BytesIO(ruled.getvalue())).copy()
        ImageDraw.Draw(image).rectangle((42, 42, 57, 57), fill="black")
        image.save(text, format="PNG")

        ruled_pixels, ruled_ratio = measure_text_density(
            image_bytes=ruled.getvalue(),
            threshold=220,
            sample_scale=1.0,
        )
        text_pixels, text_ratio = measure_text_density(
            image_bytes=text.getvalue(),
            threshold=220,
            sample_scale=1.0,
        )

        self.assertEqual(ruled_pixels, 0)
        self.assertEqual(ruled_ratio, 0)
        self.assertGreater(text_pixels, ruled_pixels)
        self.assertGreater(text_ratio, ruled_ratio)

    def test_content_grid_slices_can_repeat_header_and_left_context(self) -> None:
        original = io.BytesIO()
        image = Image.new("RGB", (100, 80), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 89, 69), outline="black", width=2)
        image.save(original, format="PNG")

        slices = make_content_grid_slices(
            file_name="sample.jpg",
            image_bytes=original.getvalue(),
            rows=2,
            cols=2,
            threshold=250,
            sample_scale=1.0,
            padding=0,
            header_context_height=10,
            left_context_width=12,
        )

        bottom_right = slices[-1]
        self.assertEqual((bottom_right.x0, bottom_right.x1, bottom_right.y0, bottom_right.y1), (50, 90, 40, 70))
        self.assertEqual((bottom_right.width, bottom_right.height), (52, 40))

        with Image.open(io.BytesIO(bottom_right.image_bytes)) as uploaded:
            self.assertEqual(uploaded.size, (52, 40))
            grayscale = uploaded.convert("L")
            top_context = grayscale.crop((12, 0, 52, 10))
            left_context = grayscale.crop((0, 10, 12, 40))

        self.assertLess(_min_gray(top_context), 100)
        self.assertLess(_min_gray(left_context), 100)

def _min_gray(image: Image.Image) -> int:
    histogram = image.histogram()
    for value, count in enumerate(histogram):
        if count:
            return value
    return 255


if __name__ == "__main__":
    unittest.main()
