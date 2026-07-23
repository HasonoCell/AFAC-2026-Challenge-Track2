"""Image resizing and slicing utilities."""

from __future__ import annotations

import io
import math
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
    content_pixels: int | None = None
    content_ratio: float | None = None
    text_pixels: int | None = None
    text_ratio: float | None = None


@dataclass(frozen=True)
class ContentBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

@dataclass(frozen=True)
class ProjectionProfile:
    dense_bands: int
    largest_gap: int


@dataclass(frozen=True)
class DocumentProfile:
    width: int
    height: int
    pixels: int
    aspect_ratio: float
    content_box: ContentBox
    content_pixels: int
    content_ratio: float
    horizontal_projection: ProjectionProfile
    vertical_projection: ProjectionProfile

    @property
    def content_width(self) -> int:
        return self.content_box.width

    @property
    def content_height(self) -> int:
        return self.content_box.height


def image_dimensions(*, image_bytes: bytes) -> tuple[int, int]:
    """Read encoded image dimensions without decoding its pixel buffer."""

    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.size


def profile_image(
    *,
    image_bytes: bytes,
    threshold: int = 245,
    sample_scale: float = 0.04,
    padding: int = 0,
    dense_axis_ratio: float = 0.002,
) -> DocumentProfile:
    """Build a low-cost profile for routing and adaptive slicing."""

    if not 0 <= threshold <= 255:
        raise ValueError("content threshold must be between 0 and 255")
    if sample_scale <= 0:
        raise ValueError("sample scale must be positive")
    if padding < 0:
        raise ValueError("content padding must be >= 0")

    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
        sample_width = max(1, round(width * sample_scale))
        sample_height = max(1, round(height * sample_scale))
        grayscale = _decode_sampled_grayscale(
            image,
            width=sample_width,
            height=sample_height,
        )
        mask = grayscale.point(lambda pixel: 255 if pixel < threshold else 0)
        bbox = mask.getbbox()
        if bbox is None:
            content_box = ContentBox(0, 0, width, height)
        else:
            x_scale = width / sample_width
            y_scale = height / sample_height
            content_box = ContentBox(
                max(0, math.floor(bbox[0] * x_scale) - padding),
                max(0, math.floor(bbox[1] * y_scale) - padding),
                min(width, math.ceil(bbox[2] * x_scale) + padding),
                min(height, math.ceil(bbox[3] * y_scale) + padding),
            )
        content_sample_pixels = _count_mask_pixels(mask)
        sample_pixels = max(1, sample_width * sample_height)
        content_ratio = content_sample_pixels / sample_pixels
        pixels = width * height
        return DocumentProfile(
            width=width,
            height=height,
            pixels=pixels,
            aspect_ratio=width / max(1, height),
            content_box=content_box,
            content_pixels=round(content_ratio * pixels),
            content_ratio=content_ratio,
            horizontal_projection=_projection_profile(
                mask,
                axis="horizontal",
                dense_axis_ratio=dense_axis_ratio,
            ),
            vertical_projection=_projection_profile(
                mask,
                axis="vertical",
                dense_axis_ratio=dense_axis_ratio,
            ),
        )


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


def detect_content_box(
    *,
    image_bytes: bytes,
    threshold: int = 245,
    sample_scale: float = 0.05,
    padding: int = 150,
) -> ContentBox:
    """Find the bounding box of non-white-ish pixels in original coordinates."""

    if not 0 <= threshold <= 255:
        raise ValueError("content threshold must be between 0 and 255")
    if sample_scale <= 0:
        raise ValueError("sample scale must be positive")
    if padding < 0:
        raise ValueError("content padding must be >= 0")

    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
        sample_width = max(1, round(width * sample_scale))
        sample_height = max(1, round(height * sample_scale))
        grayscale = _decode_sampled_grayscale(
            image,
            width=sample_width,
            height=sample_height,
        )
        mask = grayscale.point(lambda pixel: 255 if pixel < threshold else 0)
        bbox = mask.getbbox()
        if bbox is None:
            return ContentBox(0, 0, width, height)

        x_scale = width / sample_width
        y_scale = height / sample_height
        x0 = max(0, math.floor(bbox[0] * x_scale) - padding)
        y0 = max(0, math.floor(bbox[1] * y_scale) - padding)
        x1 = min(width, math.ceil(bbox[2] * x_scale) + padding)
        y1 = min(height, math.ceil(bbox[3] * y_scale) + padding)
        return ContentBox(x0, y0, x1, y1)


def measure_content_density(
    *,
    image_bytes: bytes,
    threshold: int = 245,
) -> tuple[int, float]:
    """Count non-white-ish pixels and their ratio in an encoded image."""

    if not 0 <= threshold <= 255:
        raise ValueError("content threshold must be between 0 and 255")

    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        return _measure_content_density(image, threshold)


def measure_text_density(
    *,
    image_bytes: bytes,
    threshold: int = 220,
    sample_scale: float = 0.2,
    ruling_line_ratio: float = 0.8,
) -> tuple[int, float]:
    """Estimate text density while excluding long table ruling lines.

    This is intentionally conservative: it only removes pixels from rows or
    columns that are dark for most of their length.  It makes large triangular
    or otherwise empty regions inside ruled financial tables skippable without
    relying on filenames or document-specific geometry.
    """

    if not 0 <= threshold <= 255:
        raise ValueError("text threshold must be between 0 and 255")
    if sample_scale <= 0:
        raise ValueError("text sample scale must be positive")
    if not 0 < ruling_line_ratio <= 1:
        raise ValueError("ruling line ratio must be in (0, 1]")

    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        return _measure_text_density(
            image,
            threshold=threshold,
            sample_scale=sample_scale,
            ruling_line_ratio=ruling_line_ratio,
        )


def make_content_grid_slices(
    *,
    file_name: str,
    image_bytes: bytes,
    rows: int,
    cols: int,
    threshold: int = 245,
    sample_scale: float = 0.05,
    padding: int = 150,
    x_overlap: int = 0,
    y_overlap: int = 0,
    header_context_height: int = 0,
    left_context_width: int = 0,
    snap_boundaries: bool = False,
    snap_x_boundaries: bool = False,
    snap_y_boundaries: bool = False,
    snap_search_ratio: float = 0.12,
    snap_min_line_ratio: float = 0.12,
    jpeg_quality: int = 95,
) -> list[ImageSlice]:
    """Crop the detected content box into a deterministic row-major grid."""

    if rows <= 0:
        raise ValueError("content grid rows must be positive")
    if cols <= 0:
        raise ValueError("content grid cols must be positive")
    if x_overlap < 0:
        raise ValueError("content grid x overlap must be >= 0")
    if y_overlap < 0:
        raise ValueError("content grid y overlap must be >= 0")
    if header_context_height < 0:
        raise ValueError("content grid header context height must be >= 0")
    if left_context_width < 0:
        raise ValueError("content grid left context width must be >= 0")
    if not 0 <= snap_search_ratio <= 0.25:
        raise ValueError("content grid snap search ratio must be between 0 and 0.25")
    if not 0 < snap_min_line_ratio <= 1:
        raise ValueError("content grid snap line ratio must be in (0, 1]")

    content_box = detect_content_box(
        image_bytes=image_bytes,
        threshold=threshold,
        sample_scale=sample_scale,
        padding=padding,
    )
    x_edges = _even_edges(content_box.x0, content_box.x1, cols)
    y_edges = _even_edges(content_box.y0, content_box.y1, rows)

    slices: list[ImageSlice] = []
    stem = Path(file_name).stem
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        if snap_boundaries or snap_x_boundaries:
            x_edges = _snap_edges_to_ruling_lines(
                image,
                content_box,
                x_edges,
                axis="vertical",
                search_ratio=snap_search_ratio,
                min_line_ratio=snap_min_line_ratio,
            )
        if snap_boundaries or snap_y_boundaries:
            y_edges = _snap_edges_to_ruling_lines(
                image,
                content_box,
                y_edges,
                axis="horizontal",
                search_ratio=snap_search_ratio,
                min_line_ratio=snap_min_line_ratio,
            )
        for row in range(1, rows + 1):
            y0 = max(content_box.y0, y_edges[row - 1] - y_overlap)
            y1 = min(content_box.y1, y_edges[row] + y_overlap)
            for col in range(1, cols + 1):
                x0 = max(content_box.x0, x_edges[col - 1] - x_overlap)
                x1 = min(content_box.x1, x_edges[col] + x_overlap)
                crop = image.crop((x0, y0, x1, y1))
                content_pixels, content_ratio = _measure_content_density(crop, threshold)
                text_pixels, text_ratio = _measure_text_density(
                    crop,
                    threshold=220,
                    sample_scale=0.2,
                    ruling_line_ratio=0.8,
                )
                upload_crop = _compose_context_crop(
                    image=image,
                    crop=crop,
                    content_box=content_box,
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    row=row,
                    col=col,
                    header_context_height=header_context_height,
                    left_context_width=left_context_width,
                )
                slices.append(
                    ImageSlice(
                        file_name=f"{stem}_content_r{row:03d}_c{col:03d}.jpg",
                        image_bytes=_encode_jpeg(upload_crop, jpeg_quality),
                        x0=x0,
                        x1=x1,
                        y0=y0,
                        y1=y1,
                        width=upload_crop.width,
                        height=upload_crop.height,
                        row=row,
                        col=col,
                        rows=rows,
                        cols=cols,
                        content_pixels=content_pixels,
                        content_ratio=content_ratio,
                        text_pixels=text_pixels,
                        text_ratio=text_ratio,
                    )
                )
    return slices


def make_adaptive_content_grid_slices(
    *,
    file_name: str,
    image_bytes: bytes,
    target_tile_width: int = 2800,
    target_tile_height: int = 4200,
    max_rows: int = 8,
    max_cols: int = 10,
    min_rows: int = 1,
    min_cols: int = 1,
    threshold: int = 245,
    sample_scale: float = 0.05,
    padding: int = 150,
    overlap_ratio: float = 0.05,
    min_overlap: int = 80,
    max_overlap: int = 320,
    header_context_height: int = 0,
    left_context_width: int = 0,
    snap_boundaries: bool = False,
    snap_x_boundaries: bool = False,
    snap_y_boundaries: bool = False,
    snap_search_ratio: float = 0.12,
    snap_min_line_ratio: float = 0.12,
    jpeg_quality: int = 95,
) -> list[ImageSlice]:
    """Slice the detected content box with a size-driven adaptive grid."""

    if target_tile_width <= 0 or target_tile_height <= 0:
        raise ValueError("target tile dimensions must be positive")
    if max_rows <= 0 or max_cols <= 0:
        raise ValueError("max grid dimensions must be positive")
    if min_rows <= 0 or min_cols <= 0:
        raise ValueError("min grid dimensions must be positive")
    if min_rows > max_rows or min_cols > max_cols:
        raise ValueError("min grid dimensions must not exceed max grid dimensions")
    if overlap_ratio < 0:
        raise ValueError("overlap ratio must be >= 0")

    profile = profile_image(
        image_bytes=image_bytes,
        threshold=threshold,
        sample_scale=sample_scale,
        padding=padding,
    )
    rows = _clamp(
        math.ceil(profile.content_height / target_tile_height),
        min_rows,
        max_rows,
    )
    cols = _clamp(
        math.ceil(profile.content_width / target_tile_width),
        min_cols,
        max_cols,
    )
    tile_width = max(1, round(profile.content_width / cols))
    tile_height = max(1, round(profile.content_height / rows))
    x_overlap = _adaptive_overlap(
        tile_width,
        overlap_ratio,
        min_overlap if cols > 1 else 0,
        max_overlap,
    )
    y_overlap = _adaptive_overlap(
        tile_height,
        overlap_ratio,
        min_overlap if rows > 1 else 0,
        max_overlap,
    )
    return make_content_grid_slices(
        file_name=file_name,
        image_bytes=image_bytes,
        rows=rows,
        cols=cols,
        threshold=threshold,
        sample_scale=sample_scale,
        padding=padding,
        x_overlap=x_overlap,
        y_overlap=y_overlap,
        header_context_height=header_context_height,
        left_context_width=left_context_width,
        snap_boundaries=snap_boundaries,
        snap_x_boundaries=snap_x_boundaries,
        snap_y_boundaries=snap_y_boundaries,
        snap_search_ratio=snap_search_ratio,
        snap_min_line_ratio=snap_min_line_ratio,
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


def _even_edges(start: int, end: int, parts: int) -> list[int]:
    return [start + round((end - start) * index / parts) for index in range(parts + 1)]


def _snap_edges_to_ruling_lines(
    image: Image.Image,
    content_box: ContentBox,
    edges: list[int],
    *,
    axis: str,
    search_ratio: float,
    min_line_ratio: float,
) -> list[int]:
    """Move interior grid edges to nearby long dark ruling lines.

    Equal-width cuts often bisect a financial-table column or row. A ruling
    line is useful only when it remains dark across a substantial fraction of
    the perpendicular content axis; ordinary glyph strokes therefore do not
    qualify. Edges without a nearby qualifying local peak remain unchanged.
    """

    if len(edges) <= 2 or search_ratio <= 0:
        return edges
    if axis not in {"vertical", "horizontal"}:
        raise ValueError("snap axis must be vertical or horizontal")

    crop = image.crop(
        (content_box.x0, content_box.y0, content_box.x1, content_box.y1)
    ).convert("L")
    max_width = 1600
    max_height = 1200
    scale = min(
        1.0,
        max_width / max(1, crop.width),
        max_height / max(1, crop.height),
    )
    if scale < 1:
        crop = crop.resize(
            (
                max(1, round(crop.width * scale)),
                max(1, round(crop.height * scale)),
            ),
            Image.Resampling.BILINEAR,
        )

    pixels = crop.load()
    if axis == "vertical":
        length = crop.width
        perpendicular = crop.height
        ratios = [
            sum(pixels[position, offset] < 180 for offset in range(perpendicular))
            / max(1, perpendicular)
            for position in range(length)
        ]
        box_start = content_box.x0
    else:
        length = crop.height
        perpendicular = crop.width
        ratios = [
            sum(pixels[offset, position] < 180 for offset in range(perpendicular))
            / max(1, perpendicular)
            for position in range(length)
        ]
        box_start = content_box.y0

    nominal_step = (edges[-1] - edges[0]) / max(1, len(edges) - 1)
    search_radius = max(1, round(nominal_step * search_ratio * scale))
    snapped = [edges[0]]
    for edge in edges[1:-1]:
        expected = round((edge - box_start) * scale)
        low = max(1, expected - search_radius)
        high = min(length - 2, expected + search_radius)
        peaks = [
            position
            for position in range(low, high + 1)
            if ratios[position] >= min_line_ratio
            and ratios[position] >= ratios[position - 1]
            and ratios[position] >= ratios[position + 1]
        ]
        if peaks:
            best = min(
                peaks,
                key=lambda position: (
                    abs(position - expected),
                    -ratios[position],
                ),
            )
            candidate = box_start + round(best / scale)
            minimum_gap = max(1, round(nominal_step * 0.50))
            if candidate - snapped[-1] >= minimum_gap:
                snapped.append(candidate)
                continue
        snapped.append(edge)
    snapped.append(edges[-1])
    return snapped


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


def _compose_context_crop(
    *,
    image: Image.Image,
    crop: Image.Image,
    content_box: ContentBox,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    row: int,
    col: int,
    header_context_height: int,
    left_context_width: int,
) -> Image.Image:
    include_header = row > 1 and header_context_height > 0
    include_left = col > 1 and left_context_width > 0
    if not include_header and not include_left:
        return crop

    header_y0 = content_box.y0
    header_y1 = min(content_box.y1, header_y0 + header_context_height) if include_header else header_y0
    header_height = max(0, header_y1 - header_y0)

    left_x0 = content_box.x0
    left_x1 = min(content_box.x1, left_x0 + left_context_width) if include_left else left_x0
    left_width = max(0, left_x1 - left_x0)

    if header_height == 0 and left_width == 0:
        return crop

    canvas = Image.new(
        "RGB",
        (left_width + crop.width, header_height + crop.height),
        "white",
    )
    if include_header and header_height > 0:
        if left_width > 0:
            _paste_region(
                canvas,
                image,
                (left_x0, header_y0, left_x1, header_y1),
                (0, 0),
            )
        _paste_region(
            canvas,
            image,
            (x0, header_y0, x1, header_y1),
            (left_width, 0),
        )
    if include_left and left_width > 0:
        _paste_region(
            canvas,
            image,
            (left_x0, y0, left_x1, y1),
            (0, header_height),
        )
    if crop.mode != canvas.mode:
        crop = crop.convert(canvas.mode)
    canvas.paste(crop, (left_width, header_height))
    return canvas


def _paste_region(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
    xy: tuple[int, int],
) -> None:
    region = image.crop(box)
    if region.mode != canvas.mode:
        region = region.convert(canvas.mode)
    canvas.paste(region, xy)


def _encode_jpeg(image: Image.Image, jpeg_quality: int) -> bytes:
    buffer = io.BytesIO()
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue()


def _measure_content_density(image: Image.Image, threshold: int) -> tuple[int, float]:
    grayscale = image.convert("L")
    histogram = grayscale.histogram()
    content_pixels = sum(histogram[:threshold])
    total_pixels = max(1, grayscale.width * grayscale.height)
    return content_pixels, content_pixels / total_pixels


def _measure_text_density(
    image: Image.Image,
    *,
    threshold: int,
    sample_scale: float,
    ruling_line_ratio: float,
) -> tuple[int, float]:
    grayscale = image.convert("L")
    width = max(1, round(grayscale.width * sample_scale))
    height = max(1, round(grayscale.height * sample_scale))
    sample = grayscale.resize((width, height), Image.Resampling.NEAREST)
    values = sample.tobytes()
    dark = [pixel < threshold for pixel in values]
    row_counts = [sum(dark[row * width : (row + 1) * width]) for row in range(height)]
    column_counts = [0] * width
    for index, is_dark in enumerate(dark):
        if is_dark:
            column_counts[index % width] += 1

    ruled_rows = {
        row for row, count in enumerate(row_counts) if count / width >= ruling_line_ratio
    }
    ruled_columns = {
        column
        for column, count in enumerate(column_counts)
        if count / height >= ruling_line_ratio
    }
    text_pixels = sum(
        is_dark
        and index // width not in ruled_rows
        and index % width not in ruled_columns
        for index, is_dark in enumerate(dark)
    )
    total_pixels = max(1, width * height)
    return text_pixels, text_pixels / total_pixels


def _count_mask_pixels(mask: Image.Image) -> int:
    histogram = mask.histogram()
    return histogram[255] if len(histogram) > 255 else 0


def _decode_sampled_grayscale(
    image: Image.Image,
    *,
    width: int,
    height: int,
) -> Image.Image:
    """Decode a small grayscale routing sample without expanding a huge JPEG.

    JPEG ``draft`` asks Pillow's decoder for a native downsampled image. Other
    formats ignore it and retain the previous convert-and-resize behavior.
    """

    if width <= 0 or height <= 0:
        raise ValueError("sample dimensions must be positive")

    image.draft("L", (width, height))
    grayscale = image if image.mode == "L" else image.convert("L")
    if grayscale.size != (width, height):
        return grayscale.resize((width, height), Image.Resampling.BILINEAR)
    return grayscale.copy()


def _projection_profile(
    mask: Image.Image,
    *,
    axis: str,
    dense_axis_ratio: float,
) -> ProjectionProfile:
    width, height = mask.size
    pixels = mask.load()
    if axis == "horizontal":
        line_count = height
        line_length = width
        values = [
            sum(1 for x in range(width) if pixels[x, y])
            for y in range(height)
        ]
    elif axis == "vertical":
        line_count = width
        line_length = height
        values = [
            sum(1 for y in range(height) if pixels[x, y])
            for x in range(width)
        ]
    else:
        raise ValueError(f"unsupported projection axis: {axis}")

    dense_threshold = max(1, round(line_length * dense_axis_ratio))
    dense_lines = [value >= dense_threshold for value in values]
    dense_bands = 0
    in_band = False
    largest_gap = 0
    current_gap = 0
    for dense in dense_lines:
        if dense:
            if not in_band:
                dense_bands += 1
            in_band = True
            largest_gap = max(largest_gap, current_gap)
            current_gap = 0
        else:
            in_band = False
            current_gap += 1
    largest_gap = max(largest_gap, current_gap)
    if line_count == 0:
        largest_gap = 0
    return ProjectionProfile(dense_bands=dense_bands, largest_gap=largest_gap)


def _adaptive_overlap(
    tile_length: int,
    overlap_ratio: float,
    min_overlap: int,
    max_overlap: int,
) -> int:
    if min_overlap <= 0 or overlap_ratio == 0:
        return 0
    return min(max_overlap, max(min_overlap, round(tile_length * overlap_ratio)))


def _clamp(value: int, lower: int, upper: int) -> int:
    return min(upper, max(lower, value))
