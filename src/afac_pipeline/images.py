"""Image resizing and slicing utilities."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class ImageSlice:
    file_name: str
    image_bytes: bytes
    y0: int
    y1: int
    width: int
    height: int


def make_vertical_slices(
    *,
    file_name: str,
    image_bytes: bytes,
    slice_height: int | None,
    overlap: int = 0,
    max_width: int | None = None,
    jpeg_quality: int = 95,
) -> list[ImageSlice]:
    if (not slice_height or slice_height <= 0) and not max_width:
        return [
            ImageSlice(
                file_name=file_name,
                image_bytes=image_bytes,
                y0=0,
                y1=0,
                width=0,
                height=0,
            )
        ]
    if overlap < 0:
        raise ValueError("--slice-overlap must be >= 0")
    if slice_height and slice_height > 0 and overlap >= slice_height:
        raise ValueError("--slice-overlap must be smaller than --slice-height")
    if max_width is not None and max_width <= 0:
        raise ValueError("--max-width must be positive when provided")

    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        width, height = image.size
        resized = False
        if max_width and width > max_width:
            scale = max_width / width
            new_size = (max_width, max(1, round(height * scale)))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            width, height = image.size
            resized = True
        if not slice_height or slice_height <= 0 or height <= slice_height:
            output_bytes = _encode_jpeg(image, jpeg_quality) if resized else image_bytes
            return [
                ImageSlice(
                    file_name=file_name,
                    image_bytes=output_bytes,
                    y0=0,
                    y1=height,
                    width=width,
                    height=height,
                )
            ]

        step = slice_height - overlap
        slices: list[ImageSlice] = []
        stem = Path(file_name).stem
        for index, y0 in enumerate(range(0, height, step), start=1):
            y1 = min(y0 + slice_height, height)
            crop = image.crop((0, y0, width, y1))
            buffer = io.BytesIO()
            if crop.mode not in ("RGB", "L"):
                crop = crop.convert("RGB")
            crop.save(buffer, format="JPEG", quality=jpeg_quality)
            slices.append(
                ImageSlice(
                    file_name=f"{stem}_part{index:03d}.jpg",
                    image_bytes=buffer.getvalue(),
                    y0=y0,
                    y1=y1,
                    width=width,
                    height=y1 - y0,
                )
            )
            if y1 >= height:
                break
    return slices


def _encode_jpeg(image: Image.Image, jpeg_quality: int) -> bytes:
    buffer = io.BytesIO()
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue()
