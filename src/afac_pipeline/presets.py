"""Named, versioned Pipeline configurations for reproducible submissions."""

from __future__ import annotations

from dataclasses import replace

from .baseline import BaselineConfig


B_GENERALIZATION_V1 = "b-generalization-v1"
B_GENERALIZATION_V2 = "b-generalization-v2"
B_GENERALIZATION_V3 = "b-generalization-v3"
B_GENERALIZATION_V4 = "b-generalization-v4"
B_GENERALIZATION_V5 = "b-generalization-v5"
BASELINE_PRESETS = (
    B_GENERALIZATION_V1,
    B_GENERALIZATION_V2,
    B_GENERALIZATION_V3,
    B_GENERALIZATION_V4,
    B_GENERALIZATION_V5,
)


def apply_baseline_preset(config: BaselineConfig, preset: str) -> BaselineConfig:
    """Apply a frozen routing preset after ordinary CLI defaults are parsed.

    A named preset deliberately owns every quality- and safety-sensitive knob
    below. Runtime paths and dataset selection remain outside the preset.
    """

    if preset not in BASELINE_PRESETS:
        allowed = ", ".join(BASELINE_PRESETS)
        raise ValueError(f"unknown baseline preset {preset!r}; expected {allowed}")
    return replace(
        config,
        sleep_seconds=3.0,
        retries=1,
        retry_sleep_seconds=10.0,
        table_mode="anchor",
        table_anchor_max_candidates=1,
        table_anchor_max_attempts=1,
        table_repair_min_chars=50_000,
        table_repair_target_tile_width=(
            1_500 if preset == B_GENERALIZATION_V5 else 800
        ),
        # The sparse mid-resolution route uses 6x4 content tiles on a
        # 15-MP portrait page; 650px would create an unnecessary eighth row.
        table_repair_target_tile_height=(
            800 if preset in {B_GENERALIZATION_V4, B_GENERALIZATION_V5} else 650
        ),
        table_repair_vertical_aspect_threshold=1.9,
        # v5's validated local-OCR cache was generated with this tighter
        # content box.  Keep cache coordinates and replay coordinates exact.
        table_repair_content_scale=(
            0.05 if preset == B_GENERALIZATION_V5 else config.table_repair_content_scale
        ),
        table_repair_content_padding=(
            150 if preset == B_GENERALIZATION_V5 else config.table_repair_content_padding
        ),
        table_repair_overlap=0,
        table_repair_snap_boundaries=False,
        table_repair_snap_x_boundaries=False,
        table_repair_snap_y_boundaries=False,
        table_repair_min_text_pixels=10,
        table_repair_header_context_height=0,
        table_repair_left_context_width=0,
        table_repair_min_success_parts=4,
        table_repair_min_success_ratio=0.60,
        table_repair_max_calls=48,
        table_repair_max_failed_parts=8,
        table_repair_max_failed_ratio=0.35,
        table_repair_max_identical_parts=3,
        table_repair_identical_min_chars=1000,
        table_repair_workers=5,
        table_local_ocr_backend=(
            "rapidocr"
            if preset in {
                B_GENERALIZATION_V2,
                B_GENERALIZATION_V3,
                B_GENERALIZATION_V4,
                B_GENERALIZATION_V5,
            }
            else "off"
        ),
        table_local_ocr_min_pixels=(
            10_000_000
            if preset == B_GENERALIZATION_V4
            else (20_000_000 if preset == B_GENERALIZATION_V5 else 100_000_000)
        ),
        table_local_ocr_max_pixels=(
            20_000_000
            if preset == B_GENERALIZATION_V4
            else (40_000_000 if preset == B_GENERALIZATION_V5 else 0)
        ),
        table_local_ocr_trigger_max_chars=(
            1_000 if preset in {B_GENERALIZATION_V4, B_GENERALIZATION_V5} else 0
        ),
        table_local_ocr_workers=4,
        table_local_ocr_refine_saturated=(preset == B_GENERALIZATION_V3),
        table_local_ocr_max_refine_depth=1,
        table_refine_max_depth=0,
        long_slice_height=12_000,
        long_slice_overlap=400,
        long_low_confidence_char_density=0.14,
        long_fallback_slice_height=8_000,
        long_fallback_overlap=300,
        long_min_success_ratio=0.80,
        long_max_failed_parts=1,
        on_error="placeholder",
        # The B evaluator accepted all-pipe v9/v17 and rejected submissions
        # retaining HTML tables.  This is a platform-compatibility constraint,
        # not an A-list-specific score tweak.
        table_output_format="markdown",
    )
