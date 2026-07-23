from __future__ import annotations

import unittest
from pathlib import Path

from afac_pipeline.baseline import BaselineConfig
from afac_pipeline.presets import (
    B_GENERALIZATION_V1,
    B_GENERALIZATION_V2,
    B_GENERALIZATION_V3,
    B_GENERALIZATION_V4,
    B_GENERALIZATION_V5,
    B_GENERALIZATION_V6,
    apply_baseline_preset,
)


class BaselinePresetTest(unittest.TestCase):
    def test_b_generalization_v1_freezes_validated_quality_and_safety_knobs(self) -> None:
        config = apply_baseline_preset(
            BaselineConfig(
                output_csv=Path("submission.csv"),
                cache_dir=Path("cache"),
            ),
            B_GENERALIZATION_V1,
        )

        self.assertEqual(config.table_mode, "anchor")
        self.assertEqual(config.table_repair_min_chars, 50_000)
        self.assertEqual(
            (
                config.table_repair_target_tile_width,
                config.table_repair_target_tile_height,
            ),
            (800, 650),
        )
        self.assertFalse(config.table_repair_snap_boundaries)
        self.assertFalse(config.table_repair_snap_x_boundaries)
        self.assertFalse(config.table_repair_snap_y_boundaries)
        self.assertEqual(config.table_repair_min_success_ratio, 0.60)
        self.assertEqual(config.table_repair_max_calls, 48)
        self.assertEqual(config.table_repair_workers, 5)
        self.assertEqual(config.table_refine_max_depth, 0)
        self.assertEqual(config.table_local_ocr_backend, "off")
        self.assertEqual(config.long_min_success_ratio, 0.80)
        self.assertEqual(config.long_max_failed_parts, 1)
        self.assertEqual(config.on_error, "placeholder")
        self.assertEqual(config.table_output_format, "markdown")

    def test_b_generalization_v2_adds_guarded_cross_platform_ocr(self) -> None:
        config = apply_baseline_preset(
            BaselineConfig(
                output_csv=Path("submission.csv"),
                cache_dir=Path("cache"),
            ),
            B_GENERALIZATION_V2,
        )

        self.assertEqual(config.table_local_ocr_backend, "rapidocr")
        self.assertEqual(config.table_local_ocr_min_pixels, 100_000_000)
        self.assertEqual(config.table_local_ocr_workers, 4)
        self.assertFalse(config.table_local_ocr_refine_saturated)

    def test_b_generalization_v3_refines_only_saturated_local_tiles(self) -> None:
        config = apply_baseline_preset(
            BaselineConfig(
                output_csv=Path("submission.csv"),
                cache_dir=Path("cache"),
            ),
            B_GENERALIZATION_V3,
        )

        self.assertEqual(config.table_local_ocr_backend, "rapidocr")
        self.assertTrue(config.table_local_ocr_refine_saturated)
        self.assertEqual(config.table_local_ocr_max_refine_depth, 1)

    def test_b_generalization_v4_targets_only_sparse_mid_resolution_outputs(self) -> None:
        config = apply_baseline_preset(
            BaselineConfig(output_csv=Path("submission.csv"), cache_dir=Path("cache")),
            B_GENERALIZATION_V4,
        )

        self.assertEqual(config.table_local_ocr_backend, "rapidocr")
        self.assertEqual(config.table_local_ocr_min_pixels, 10_000_000)
        self.assertEqual(config.table_local_ocr_max_pixels, 20_000_000)
        self.assertEqual(config.table_local_ocr_trigger_max_chars, 1_000)
        self.assertEqual(config.table_repair_target_tile_height, 800)

    def test_b_generalization_v5_uses_wider_tiles_for_sparse_30mp_tables(self) -> None:
        config = apply_baseline_preset(
            BaselineConfig(output_csv=Path("submission.csv"), cache_dir=Path("cache")),
            B_GENERALIZATION_V5,
        )

        self.assertEqual(config.table_local_ocr_backend, "rapidocr")
        self.assertEqual(config.table_local_ocr_min_pixels, 20_000_000)
        self.assertEqual(config.table_local_ocr_max_pixels, 40_000_000)
        self.assertEqual(config.table_local_ocr_trigger_max_chars, 1_000)
        self.assertEqual(
            (
                config.table_repair_target_tile_width,
                config.table_repair_target_tile_height,
            ),
            (1_500, 800),
        )
        self.assertEqual(config.table_repair_content_scale, 0.05)
        self.assertEqual(config.table_repair_content_padding, 150)

    def test_b_generalization_v6_adds_only_sparse_narrow_long_text_fallback(self) -> None:
        config = apply_baseline_preset(
            BaselineConfig(output_csv=Path("submission.csv"), cache_dir=Path("cache")),
            B_GENERALIZATION_V6,
        )

        self.assertEqual(config.table_local_ocr_min_pixels, 20_000_000)
        self.assertEqual(config.long_local_ocr_backend, "rapidocr")
        self.assertEqual(config.long_local_ocr_min_pixels, 100_000_000)
        self.assertEqual(config.long_local_ocr_max_width, 2_000)
        self.assertEqual(config.long_local_ocr_trigger_char_density, 0.06)
        self.assertEqual(config.long_local_ocr_min_char_density, 0.08)

    def test_unknown_preset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown baseline preset"):
            apply_baseline_preset(
                BaselineConfig(
                    output_csv=Path("submission.csv"),
                    cache_dir=Path("cache"),
                ),
                "future-unvalidated-preset",
            )


if __name__ == "__main__":
    unittest.main()
