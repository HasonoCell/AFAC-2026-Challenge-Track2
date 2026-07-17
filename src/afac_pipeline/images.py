"""Image resizing and slicing utilities."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


Image.MAX_IMAGE_PIXELS = None

DEFAULT_CROP_ANCHORS = (
    "top_left",
    "top_center",
    "center",
    "bottom_left",
    "top_right",
)


@dataclass(frozen=True)
class ImageSlice:
    file_name: str
    image_bytes: bytes
    x0: int
    x1: int
    y0: int
    y1: int
    width: int
    height: int
    row: int = 1
    col: int = 1
    rows: int = 1
    cols: int = 1


def make_vertical_slices(
    *,
    file_name: str,
    image_bytes: bytes,
    slice_height: int | None,
    overlap: int = 0,
    max_width: int | None = None,
    jpeg_quality: int = 95,
) -> list[ImageSlice]:
    return make_grid_slices(
        file_name=file_name,
        image_bytes=image_bytes,
        slice_width=None,
        slice_height=slice_height,
        x_overlap=0,
        y_overlap=overlap,
        max_width=max_width,
        jpeg_quality=jpeg_quality,
    )


def make_grid_slices(
    *,
    file_name: str,
    image_bytes: bytes,
    slice_width: int | None = None,
    slice_height: int | None = None,
    x_overlap: int = 0,
    y_overlap: int = 0,
    max_width: int | None = None,
    jpeg_quality: int = 95,
) -> list[ImageSlice]:
    """Resize an image and split it into deterministic 2D tiles.

    Coordinates are expressed in the resized image coordinate system.  Passing
    only ``slice_height`` reproduces the old vertical-slicing behaviour, while
    ``slice_width`` enables horizontal splits for very wide tables.
    """

    if x_overlap < 0:
        raise ValueError("--slice-x-overlap must be >= 0")
    if y_overlap < 0:
        raise ValueError("--slice-overlap must be >= 0")
    if slice_width and slice_width > 0 and x_overlap >= slice_width:
        raise ValueError("--slice-x-overlap must be smaller than --slice-width")
    if slice_height and slice_height > 0 and y_overlap >= slice_height:
        raise ValueError("--slice-overlap must be smaller than --slice-height")
    if max_width is not None and max_width <= 0:
        raise ValueError("--max-width must be positive when provided")

    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size

        needs_width_slicing = bool(slice_width and slice_width > 0 and width > slice_width)
        needs_height_slicing = bool(slice_height and slice_height > 0 and height > slice_height)
        needs_resize = bool(max_width and width > max_width)

        if not needs_resize and not needs_width_slicing and not needs_height_slicing:
            return [
                ImageSlice(
                    file_name=file_name,
                    image_bytes=image_bytes,
                    x0=0,
                    x1=width,
                    y0=0,
                    y1=height,
                    width=width,
                    height=height,
                )
            ]

        image.load()

        if max_width and width > max_width:
            scale = max_width / width
            new_size = (max_width, max(1, round(height * scale)))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            width, height = image.size

        x_ranges = _axis_ranges(width, slice_width, x_overlap)
        y_ranges = _axis_ranges(height, slice_height, y_overlap)
        rows = len(y_ranges)
        cols = len(x_ranges)

        if rows == 1 and cols == 1:
            return [
                ImageSlice(
                    file_name=file_name,
                    image_bytes=_encode_jpeg(image, jpeg_quality),
                    x0=0,
                    x1=width,
                    y0=0,
                    y1=height,
                    width=width,
                    height=height,
                    rows=1,
                    cols=1,
                )
            ]

        slices: list[ImageSlice] = []
        stem = Path(file_name).stem
        for row, (y0, y1) in enumerate(y_ranges, start=1):
            for col, (x0, x1) in enumerate(x_ranges, start=1):
                crop = image.crop((x0, y0, x1, y1))
                slices.append(
                    ImageSlice(
                        file_name=_slice_file_name(stem, row, col, rows, cols),
                        image_bytes=_encode_jpeg(crop, jpeg_quality),
                        x0=x0,
                        x1=x1,
                        y0=y0,
                        y1=y1,
                        width=x1 - x0,
                        height=y1 - y0,
                        row=row,
                        col=col,
                        rows=rows,
                        cols=cols,
                    )
                )
    return slices


def make_anchor_crops(
    *,
    file_name: str,
    image_bytes: bytes,
    crop_sizes: Iterable[int],
    anchors: Iterable[str] = DEFAULT_CROP_ANCHORS,
    jpeg_quality: int = 95,
) -> list[ImageSlice]:
    """Create deterministic original-resolution crops for cheap API baselines."""

    unique_sizes = _unique_positive_ints(crop_sizes)
    unique_anchors = tuple(dict.fromkeys(anchors))
    if not unique_sizes:
        raise ValueError("crop_sizes must contain at least one positive integer")
    if not unique_anchors:
        raise ValueError("anchors must contain at least one value")

    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        width, height = image.size
        stem = Path(file_name).stem
        crops: list[ImageSlice] = []
        seen_boxes: set[tuple[int, int, int, int]] = set()

        for size in unique_sizes:
            crop_width = min(size, width)
            crop_height = min(size, height)
            for anchor in unique_anchors:
                x0, y0 = _anchor_origin(
                    anchor=anchor,
                    image_width=width,
                    image_height=height,
                    crop_width=crop_width,
                    crop_height=crop_height,
                )
                x1 = x0 + crop_width
                y1 = y0 + crop_height
                box = (x0, x1, y0, y1)
                if box in seen_boxes:
                    continue
                seen_boxes.add(box)
                crop = image.crop((x0, y0, x1, y1))
                crops.append(
                    ImageSlice(
                        file_name=f"{stem}_crop_{anchor}_{size}.jpg",
                        image_bytes=_encode_jpeg(crop, jpeg_quality),
                        x0=x0,
                        x1=x1,
                        y0=y0,
                        y1=y1,
                        width=crop_width,
                        height=crop_height,
                    )
                )
    return crops


def _axis_ranges(
    length: int,
    slice_length: int | None,
    overlap: int,
) -> list[tuple[int, int]]:
    if not slice_length or slice_length <= 0 or length <= slice_length:
        return [
            (0, length),
        ]

    step = slice_length - overlap
    ranges: list[tuple[int, int]] = []
    for start in range(0, length, step):
        end = min(start + slice_length, length)
        ranges.append((start, end))
        if end >= length:
            break
    return ranges


def _unique_positive_ints(values: Iterable[int]) -> tuple[int, ...]:
    unique: list[int] = []
    for value in values:
        if value <= 0:
            raise ValueError("crop sizes must be positive integers")
        if value not in unique:
            unique.append(value)
    return tuple(unique)


def _anchor_origin(
    *,
    anchor: str,
    image_width: int,
    image_height: int,
    crop_width: int,
    crop_height: int,
) -> tuple[int, int]:
    max_x = image_width - crop_width
    max_y = image_height - crop_height
    if anchor == "top_left":
        return 0, 0
    if anchor == "top_center":
        return max_x // 2, 0
    if anchor == "top_right":
        return max_x, 0
    if anchor == "center":
        return max_x // 2, max_y // 2
    if anchor == "middle_left":
        return 0, max_y // 2
    if anchor == "middle_right":
        return max_x, max_y // 2
    if anchor == "bottom_left":
        return 0, max_y
    if anchor == "bottom_center":
        return max_x // 2, max_y
    if anchor == "bottom_right":
        return max_x, max_y
    raise ValueError(f"unsupported crop anchor: {anchor}")


def _slice_file_name(stem: str, row: int, col: int, rows: int, cols: int) -> str:
    if cols == 1:
        return f"{stem}_part{row:03d}.jpg"
    if rows == 1:
        return f"{stem}_col{col:03d}.jpg"
    return f"{stem}_r{row:03d}_c{col:03d}.jpg"


def _encode_jpeg(image: Image.Image, jpeg_quality: int) -> bytes:
    buffer = io.BytesIO()
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue()
