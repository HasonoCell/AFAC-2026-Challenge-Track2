from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from .api import ALLOWED_USER_IDS, DEFAULT_API_KEY, FinixDocClient, RotatingFinixDocClient
from .baseline import (
    BaselineConfig,
    rebuild_cached_local_matrix_repairs,
    run_baseline_submission,
)
from .datasets import (
    extract_dataset,
    inspect_raw_data,
    iter_dataset_images,
    iter_dir_images,
    iter_train_markdowns,
    read_mock_submission,
    submission_template_source,
)
from .evaluation import (
    evaluate_prediction_csv,
    format_evaluation_summary,
    read_prediction_csv,
    write_evaluation_rows,
)
from .experiment import run_train_experiment
from .pipeline import PredictionConfig, run_prediction
from .presets import BASELINE_PRESETS, apply_baseline_preset
from .submission import (
    DEFAULT_MAX_SIZE_BYTES,
    build_conservative_submission_ensemble,
    build_submission_overlay,
    compact_submission_for_platform,
    format_validation_result,
    remerge_cached_grid_submission,
    validate_submission_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="afac",
        description="AFAC 2026 challenge track 2 document parsing pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-data", help="Inspect downloaded raw data.")
    inspect_parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    inspect_parser.set_defaults(func=cmd_inspect_data)

    extract_parser = subparsers.add_parser("extract", help="Extract downloaded zip files.")
    extract_parser.add_argument(
        "--dataset",
        choices=["train", "a", "b", "mock", "all"],
        default="all",
        help="Which dataset to extract.",
    )
    extract_parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    extract_parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTED_DIR)
    extract_parser.set_defaults(func=cmd_extract)

    predict_parser = subparsers.add_parser("predict", help="Generate a submission CSV.")
    source = predict_parser.add_mutually_exclusive_group()
    source.add_argument(
        "--dataset",
        choices=["train", "a", "b"],
        default="a",
        help="Read images directly from raw dataset zip files.",
    )
    source.add_argument(
        "--input-dir",
        type=Path,
        help="Read images from an extracted directory instead of raw zip files.",
    )
    predict_parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    predict_parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "submission.csv",
    )
    predict_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "cache",
    )
    predict_parser.add_argument(
        "--user-id",
        default=os.environ.get("FINIXDOC_USER_ID", "finixB2002"),
        help="FinixDoc-VL whitelisted userId.",
    )
    predict_parser.add_argument(
        "--user-ids",
        help="Optional comma-separated official userIds to rotate per API call; overrides --user-id.",
    )
    predict_parser.add_argument(
        "--api-key",
        default=os.environ.get("FINIXDOC_API_KEY", DEFAULT_API_KEY),
        help="FinixDoc-VL apiKey. Defaults to the official competition key.",
    )
    predict_parser.add_argument("--timeout", type=float, default=180.0)
    predict_parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep after API calls.")
    predict_parser.add_argument("--retries", type=int, default=1, help="Retries per image/slice after API errors.")
    predict_parser.add_argument(
        "--retry-sleep",
        type=float,
        default=60.0,
        help="Seconds to wait before retrying an API error.",
    )
    predict_parser.add_argument("--offset", type=int, default=0, help="Skip the first N discovered images.")
    predict_parser.add_argument("--limit", type=int, help="Process only the first N images.")
    predict_parser.add_argument(
        "--slice-height",
        type=int,
        help="Split tall images into vertical slices of this height before API calls.",
    )
    predict_parser.add_argument(
        "--slice-width",
        type=int,
        help="Split wide images into horizontal slices of this width before API calls.",
    )
    predict_parser.add_argument(
        "--max-width",
        type=int,
        help="Downscale images wider than this width before slicing and upload.",
    )
    predict_parser.add_argument(
        "--slice-overlap",
        type=int,
        default=120,
        help="Pixel overlap between adjacent vertical slices.",
    )
    predict_parser.add_argument(
        "--slice-x-overlap",
        type=int,
        default=120,
        help="Pixel overlap between adjacent horizontal slices.",
    )
    predict_parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for generated slices.",
    )
    predict_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore cached markdown and call/generate again.",
    )
    predict_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the API; write placeholder Markdown for smoke tests.",
    )
    predict_parser.add_argument(
        "--on-error",
        choices=["raise", "empty", "message"],
        default="raise",
        help="How to handle an image/slice that fails after retries.",
    )
    predict_parser.add_argument(
        "--errors-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "errors.csv",
        help="Where to write failed image details.",
    )
    predict_parser.set_defaults(func=cmd_predict)

    baseline_parser = subparsers.add_parser(
        "baseline-submit",
        help="Generate a low-call-count real-API baseline or train-set experiment.",
    )
    baseline_parser.add_argument(
        "--dataset",
        choices=["train", "a", "b"],
        default="a",
        help="Dataset to process: a/b for submission, train for experiments.",
    )
    baseline_parser.add_argument(
        '--preset',
        choices=BASELINE_PRESETS,
        help=(
            'Apply a frozen versioned Pipeline preset. Preset-owned routing '
            'and safety values override the individual knob defaults.'
        ),
    )
    baseline_parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    baseline_parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "submission.csv",
    )
    baseline_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "cache",
    )
    baseline_parser.add_argument(
        "--user-id",
        default=os.environ.get("FINIXDOC_USER_ID", "finixB2002"),
        help="FinixDoc-VL whitelisted userId.",
    )
    baseline_parser.add_argument(
        "--user-ids",
        help="Optional comma-separated official userIds to rotate per API call; overrides --user-id.",
    )
    baseline_parser.add_argument(
        "--api-key",
        default=os.environ.get("FINIXDOC_API_KEY", DEFAULT_API_KEY),
        help="FinixDoc-VL apiKey. Defaults to the official competition key.",
    )
    baseline_parser.add_argument("--timeout", type=float, default=300.0)
    baseline_parser.add_argument(
        "--sleep",
        type=float,
        default=12.0,
        help="Seconds to sleep after API calls; coverage tables default to 12s to respect RPM limits.",
    )
    baseline_parser.add_argument("--retries", type=int, default=0, help="Retries per crop after API errors.")
    baseline_parser.add_argument(
        "--retry-sleep",
        type=float,
        default=70.0,
        help="Seconds to wait before retrying an API error.",
    )
    baseline_parser.add_argument("--offset", type=int, default=0, help="Skip the first N discovered images.")
    baseline_parser.add_argument("--limit", type=int, help="Process only the first N images.")
    baseline_parser.add_argument(
        "--file-names",
        help=(
            "Optional comma-separated image names to process. This is mutually "
            "exclusive with --offset/--limit and supports partial recomputation."
        ),
    )
    baseline_parser.add_argument(
        "--crop-sizes",
        default="800,600,500,400",
        help="Comma-separated original-resolution square crop sizes to try.",
    )
    baseline_parser.add_argument(
        "--anchors",
        default="top_left,top_center,center,bottom_left,top_right",
        help="Comma-separated crop anchors to try for each crop size.",
    )
    baseline_parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for generated crops.",
    )
    baseline_parser.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Reject candidate crop outputs shorter than this many characters.",
    )
    baseline_parser.add_argument(
        "--table-repair-min-chars",
        type=int,
        default=600,
        help=(
            "If a table crop output is shorter than this many characters, retry "
            "with content-box grid slices. Use 0 to disable."
        ),
    )
    baseline_parser.add_argument(
        "--table-repair-min-chars-per-content-pixel",
        type=float,
        default=0.0,
        help=(
            "Raise the repair trigger in proportion to detected content pixels; "
            "use 0 to keep the fixed threshold only."
        ),
    )
    baseline_parser.add_argument(
        "--table-repair-min-gain",
        type=int,
        default=300,
        help="Only keep table repair output if it improves length by at least this many characters.",
    )
    baseline_parser.add_argument(
        "--table-repair-grid",
        default="4x4",
        help="Content-box repair grid as ROWSxCOLS, for example 4x4 or 2x2.",
    )
    baseline_parser.add_argument(
        "--table-repair-target-tile-width",
        type=int,
        default=0,
        help="If positive, derive repair columns from content width and this target pixel width.",
    )
    baseline_parser.add_argument(
        "--table-repair-target-tile-height",
        type=int,
        default=0,
        help="If positive, derive repair rows from content height and this target pixel height.",
    )
    baseline_parser.add_argument(
        "--table-repair-overlap",
        type=int,
        default=120,
        help="Pixel overlap in both directions for content-box table repair slices.",
    )
    baseline_parser.add_argument(
        "--table-repair-snap-boundaries",
        action="store_true",
        help="Snap repair grid edges to nearby long table ruling lines.",
    )
    baseline_parser.add_argument(
        "--table-repair-snap-x-boundaries",
        action="store_true",
        help="Snap only vertical grid edges to nearby table ruling lines.",
    )
    baseline_parser.add_argument(
        "--table-repair-snap-y-boundaries",
        action="store_true",
        help="Snap only horizontal grid edges to nearby table ruling lines.",
    )
    baseline_parser.add_argument(
        "--table-repair-content-threshold",
        type=int,
        default=245,
        help="Grayscale threshold for detecting non-white table content.",
    )
    baseline_parser.add_argument(
        "--table-repair-content-scale",
        type=float,
        default=0.04,
        help="Downsample scale used while detecting table content bounds.",
    )
    baseline_parser.add_argument(
        "--table-repair-content-padding",
        type=int,
        default=200,
        help="Padding around detected table content bounds before grid slicing.",
    )
    baseline_parser.add_argument(
        "--table-repair-min-content-pixels",
        type=int,
        default=1000,
        help=(
            "Skip a repair grid slice when its non-white-ish pixel count is below "
            "this value and its content ratio is also below the ratio threshold. "
            "Use 0 with --table-repair-min-content-ratio 0 to disable."
        ),
    )
    baseline_parser.add_argument(
        "--table-repair-min-content-ratio",
        type=float,
        default=0.001,
        help=(
            "Skip a repair grid slice when its non-white-ish pixel ratio is below "
            "this value and its content pixel count is also below the pixel threshold."
        ),
    )
    baseline_parser.add_argument(
        "--table-repair-min-text-pixels",
        type=int,
        default=10,
        help=(
            "Skip a ruled-table tile when it contains fewer than this many "
            "line-filtered text pixels. Use 0 to disable."
        ),
    )
    baseline_parser.add_argument(
        "--table-repair-header-context-height",
        type=int,
        default=0,
        help=(
            "Pixels of the detected table top band to prepend to non-first-row "
            "repair grid slices. Use 0 to disable repeated header context."
        ),
    )
    baseline_parser.add_argument(
        "--table-repair-left-context-width",
        type=int,
        default=0,
        help=(
            "Pixels of the detected table left band to prepend to non-first-column "
            "repair grid slices. Use 0 to disable repeated key-column context."
        ),
    )
    baseline_parser.add_argument(
        "--table-repair-min-success-parts",
        type=int,
        default=4,
        help="Minimum successful grid slices required before accepting table repair.",
    )
    baseline_parser.add_argument('--table-repair-min-success-ratio', type=float, default=0.0)
    baseline_parser.add_argument(
        "--table-repair-max-calls",
        type=int,
        default=24,
        help=(
            "Maximum top-level tiles used by one content-grid table repair; "
            "use 0 for no limit. Recursive child calls have a separate bounded "
            "tree derived from --table-refine-max-depth."
        ),
    )
    baseline_parser.add_argument(
        "--table-repair-max-failed-parts",
        type=int,
        default=0,
        help="Abort a table repair after this many failed tiles; 0 disables the budget.",
    )
    baseline_parser.add_argument(
        "--table-repair-max-failed-ratio",
        type=float,
        default=0.0,
        help="Allow this failed-tile fraction on larger repair grids; 0 keeps the absolute budget only.",
    )
    baseline_parser.add_argument(
        "--table-repair-max-identical-parts",
        type=int,
        default=3,
        help="Abort after this many identical large repair-tile outputs; 0 disables.",
    )
    baseline_parser.add_argument(
        "--table-repair-identical-min-chars",
        type=int,
        default=1000,
        help="Minimum tile-output length counted by the identical-output circuit breaker.",
    )
    baseline_parser.add_argument(
        '--table-repair-workers',
        type=int,
        default=1,
        help=(
            'Concurrent content-grid repair workers. Use multiple rotating '
            'official userIds to avoid per-user rate limits.'
        ),
    )
    baseline_parser.add_argument(
        '--table-local-ocr-backend',
        choices=['off', 'rapidocr', 'vision'],
        default='off',
        help='Guarded local OCR backend for extremely large regular numeric matrices.',
    )
    baseline_parser.add_argument('--table-local-ocr-min-pixels', type=int, default=100_000_000)
    baseline_parser.add_argument('--table-local-ocr-workers', type=int, default=4)
    baseline_parser.add_argument(
        '--table-local-ocr-refine-saturated',
        action='store_true',
        help='Split only RapidOCR tiles that hit the detector candidate cap.',
    )
    baseline_parser.add_argument('--table-local-ocr-max-refine-depth', type=int, default=1)
    baseline_parser.add_argument('--table-anchor-max-candidates', type=int, default=1)
    baseline_parser.add_argument('--table-anchor-max-attempts', type=int, default=0)
    baseline_parser.add_argument(
        '--table-mode',
        choices=['coverage', 'anchor', 'hybrid'],
        default='coverage',
        help='Table route: coverage uses adaptive full-content tiling before anchor fallback.',
    )
    baseline_parser.add_argument('--table-target-tile-width', type=int, default=2800)
    baseline_parser.add_argument('--table-target-tile-height', type=int, default=4200)
    baseline_parser.add_argument('--table-max-rows', type=int, default=8)
    baseline_parser.add_argument('--table-max-cols', type=int, default=10)
    baseline_parser.add_argument('--table-overlap-ratio', type=float, default=0.05)
    baseline_parser.add_argument('--table-min-overlap', type=int, default=80)
    baseline_parser.add_argument('--table-max-blocks', type=int, default=32)
    baseline_parser.add_argument('--table-min-score', type=float, default=45.0)
    baseline_parser.add_argument(
        '--table-max-duplicate-line-ratio',
        type=float,
        default=0.30,
        help='Reject a coverage result above this duplicate-line ratio and fall back to anchor; use a negative value to disable.',
    )
    baseline_parser.add_argument(
        '--table-hybrid-min-content-ratio',
        type=float,
        default=0.50,
        help='In hybrid mode, use coverage only when the image content-pixel ratio reaches this value; use a negative value to always use coverage.',
    )
    baseline_parser.add_argument('--table-workers', type=int, default=1, help='Concurrent coverage tile workers; keep 1 unless rotating multiple official userIds.')
    baseline_parser.add_argument('--table-refine-max-depth', type=int, default=1)
    baseline_parser.add_argument('--table-refine-rows', type=int, default=2)
    baseline_parser.add_argument('--table-refine-cols', type=int, default=2)
    baseline_parser.add_argument('--table-fragment-max-blocks', type=int, default=1)
    baseline_parser.add_argument('--table-fragment-refine-cols', type=int, default=2)
    baseline_parser.add_argument('--long-aspect-threshold', type=float, default=0.12)
    baseline_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore cached baseline markdown and call again.",
    )
    baseline_parser.add_argument(
        "--on-error",
        choices=["raise", "placeholder"],
        default="raise",
        help="How to handle an image whose crop candidates all fail.",
    )
    baseline_parser.add_argument(
        "--long-slice-height",
        type=int,
        default=12000,
        help="Height for vertical slices on very tall pages.",
    )
    baseline_parser.add_argument(
        "--long-slice-overlap",
        type=int,
        default=400,
        help="Overlap between adjacent vertical slices on very tall pages.",
    )
    baseline_parser.add_argument(
        "--long-min-chars",
        type=int,
        default=20,
        help="Minimum accepted length for long-page slice outputs.",
    )
    baseline_parser.add_argument(
        '--long-min-success-ratio',
        type=float,
        default=1.0,
        help='Minimum usable fraction of long-page slices before accepting a partial result.',
    )
    baseline_parser.add_argument(
        '--long-max-failed-parts',
        type=int,
        default=0,
        help='Maximum failed long-page slices; 0 preserves strict all-slices-required behavior.',
    )
    baseline_parser.add_argument(
        "--no-mock-template",
        action="store_true",
        help="Do not expand full A-list output to the mock submission template.",
    )
    baseline_parser.add_argument(
        "--errors-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "baseline_errors.csv",
        help="Where to write failed image details.",
    )
    baseline_parser.set_defaults(func=cmd_baseline_submit)

    experiment_parser = subparsers.add_parser(
        'experiment-train',
        help='Run a scoreable train split experiment and write predictions, metrics, manifest, and errors.',
    )
    experiment_parser.add_argument('--raw-dir', type=Path, default=DEFAULT_RAW_DIR)
    experiment_parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR / 'experiments' / 'v034-dev')
    experiment_parser.add_argument('--run-id', default='v034-dev')
    experiment_parser.add_argument(
        '--split',
        choices=['dev', 'validation', 'rest', 'all'],
        default='dev',
        help='Deterministic stratified train split to run.',
    )
    experiment_parser.add_argument(
        '--kind',
        choices=['all', 'long', 'table'],
        default='all',
        help='Run only one routed document kind within the selected split.',
    )
    experiment_parser.add_argument('--offset', type=int, default=0)
    experiment_parser.add_argument('--limit', type=int)
    experiment_parser.add_argument(
        '--file-name',
        action='append',
        default=[],
        help='Select an exact member of the chosen split; repeat for multiple files.',
    )
    experiment_parser.add_argument('--cache-dir', type=Path, default=DEFAULT_OUTPUT_DIR / 'cache' / 'experiments')
    experiment_parser.add_argument(
        '--preset',
        choices=BASELINE_PRESETS,
        help='Apply the same frozen Pipeline preset used by baseline-submit.',
    )
    experiment_parser.add_argument('--user-id', default=os.environ.get('FINIXDOC_USER_ID', 'finixB2002'))
    experiment_parser.add_argument('--user-ids', help='Optional comma-separated official userIds to rotate per API call; overrides --user-id.')
    experiment_parser.add_argument('--api-key', default=os.environ.get('FINIXDOC_API_KEY', DEFAULT_API_KEY))
    experiment_parser.add_argument('--timeout', type=float, default=300.0)
    experiment_parser.add_argument(
        '--sleep',
        type=float,
        default=12.0,
        help='Seconds to sleep after API calls; defaults to 12s for multi-tile coverage stability.',
    )
    experiment_parser.add_argument('--retries', type=int, default=0)
    experiment_parser.add_argument('--retry-sleep', type=float, default=70.0)
    experiment_parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Ignore completed record caches while retaining completed repair-tile caches.',
    )
    experiment_parser.add_argument('--jpeg-quality', type=int, default=95)
    experiment_parser.add_argument(
        '--table-repair-min-chars',
        type=int,
        default=600,
        help='Trigger anchor content-grid repair when an accepted table crop is shorter than this many characters.',
    )
    experiment_parser.add_argument('--table-repair-min-chars-per-content-pixel', type=float, default=0.0)
    experiment_parser.add_argument(
        '--table-repair-grid',
        default='4x4',
        help='Content-box repair grid as ROWSxCOLS, for example 4x4 or 6x6.',
    )
    experiment_parser.add_argument('--table-repair-target-tile-width', type=int, default=0)
    experiment_parser.add_argument('--table-repair-target-tile-height', type=int, default=0)
    experiment_parser.add_argument(
        '--table-repair-overlap',
        type=int,
        default=120,
        help='Pixel overlap in both directions for table repair tiles.',
    )
    experiment_parser.add_argument(
        '--table-repair-snap-boundaries',
        action='store_true',
        help='Snap repair grid edges to nearby long table ruling lines.',
    )
    experiment_parser.add_argument(
        '--table-repair-snap-x-boundaries',
        action='store_true',
        help='Snap only vertical grid edges to nearby table ruling lines.',
    )
    experiment_parser.add_argument(
        '--table-repair-snap-y-boundaries',
        action='store_true',
        help='Snap only horizontal grid edges to nearby table ruling lines.',
    )
    experiment_parser.add_argument(
        '--table-repair-min-gain',
        type=int,
        default=300,
        help='Minimum character gain before accepting a repaired table.',
    )
    experiment_parser.add_argument('--table-mode', choices=['coverage', 'anchor', 'hybrid'], default='coverage')
    experiment_parser.add_argument('--table-target-tile-width', type=int, default=2800)
    experiment_parser.add_argument('--table-target-tile-height', type=int, default=4200)
    experiment_parser.add_argument('--table-max-rows', type=int, default=8)
    experiment_parser.add_argument('--table-max-cols', type=int, default=10)
    experiment_parser.add_argument('--table-overlap-ratio', type=float, default=0.05)
    experiment_parser.add_argument('--table-min-overlap', type=int, default=80)
    experiment_parser.add_argument('--table-max-blocks', type=int, default=32)
    experiment_parser.add_argument('--table-min-score', type=float, default=45.0)
    experiment_parser.add_argument('--table-max-duplicate-line-ratio', type=float, default=0.30)
    experiment_parser.add_argument('--table-hybrid-min-content-ratio', type=float, default=0.50)
    experiment_parser.add_argument('--table-workers', type=int, default=1)
    experiment_parser.add_argument(
        '--table-refine-max-depth',
        type=int,
        default=None,
        help=(
            'Maximum recursive repair depth. Defaults to 1 without a preset; '
            'when supplied explicitly, it overrides the experiment preset only.'
        ),
    )
    experiment_parser.add_argument('--table-refine-rows', type=int, default=2)
    experiment_parser.add_argument('--table-refine-cols', type=int, default=2)
    experiment_parser.add_argument('--table-fragment-max-blocks', type=int, default=1)
    experiment_parser.add_argument('--table-fragment-refine-cols', type=int, default=2)
    experiment_parser.add_argument('--table-repair-content-threshold', type=int, default=245)
    experiment_parser.add_argument('--table-repair-content-scale', type=float, default=0.04)
    experiment_parser.add_argument('--table-repair-content-padding', type=int, default=200)
    experiment_parser.add_argument('--table-repair-min-content-pixels', type=int, default=1000)
    experiment_parser.add_argument('--table-repair-min-content-ratio', type=float, default=0.001)
    experiment_parser.add_argument('--table-repair-min-text-pixels', type=int, default=10)
    experiment_parser.add_argument('--table-repair-header-context-height', type=int, default=240)
    experiment_parser.add_argument('--table-repair-left-context-width', type=int, default=240)
    experiment_parser.add_argument('--table-repair-min-success-parts', type=int, default=4)
    experiment_parser.add_argument('--table-repair-min-success-ratio', type=float, default=0.0)
    experiment_parser.add_argument('--table-repair-max-calls', type=int, default=24)
    experiment_parser.add_argument('--table-repair-max-failed-parts', type=int, default=0)
    experiment_parser.add_argument('--table-repair-max-failed-ratio', type=float, default=0.0)
    experiment_parser.add_argument('--table-repair-max-identical-parts', type=int, default=3)
    experiment_parser.add_argument('--table-repair-identical-min-chars', type=int, default=1000)
    experiment_parser.add_argument(
        '--table-repair-workers',
        type=int,
        default=None,
        help=(
            'Concurrent repair workers. Defaults to 1 without a preset; when '
            'supplied explicitly, it overrides the experiment preset only.'
        ),
    )
    experiment_parser.add_argument(
        '--table-local-ocr-backend',
        choices=['off', 'rapidocr', 'vision'],
        default=None,
        help='Guarded local OCR backend for extremely large regular numeric matrices.',
    )
    experiment_parser.add_argument('--table-local-ocr-min-pixels', type=int, default=100_000_000)
    experiment_parser.add_argument('--table-local-ocr-workers', type=int, default=4)
    experiment_parser.add_argument(
        '--table-local-ocr-refine-saturated',
        action='store_true',
        help='Split only RapidOCR tiles that hit the detector candidate cap.',
    )
    experiment_parser.add_argument('--table-local-ocr-max-refine-depth', type=int, default=1)
    experiment_parser.add_argument('--table-anchor-max-candidates', type=int, default=1)
    experiment_parser.add_argument('--table-anchor-max-attempts', type=int, default=0)
    experiment_parser.add_argument('--long-slice-height', type=int, default=12000)
    experiment_parser.add_argument('--long-slice-overlap', type=int, default=400)
    experiment_parser.add_argument('--long-min-chars', type=int, default=20)
    experiment_parser.add_argument('--long-min-success-ratio', type=float, default=1.0)
    experiment_parser.add_argument('--long-max-failed-parts', type=int, default=0)
    experiment_parser.add_argument('--long-aspect-threshold', type=float, default=0.12)
    experiment_parser.set_defaults(func=cmd_experiment_train)

    validate_parser = subparsers.add_parser(
        "validate-submission",
        help="Validate a submission CSV before uploading it to Tianchi.",
    )
    validate_parser.add_argument(
        "--submission-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "submission.csv",
        help="Submission CSV to validate.",
    )
    validate_parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    validate_parser.add_argument(
        "--dataset",
        choices=["a", "b"],
        default="a",
        help="Dataset whose image names should be covered by the submission.",
    )
    validate_parser.add_argument(
        "--max-size-mb",
        type=float,
        default=DEFAULT_MAX_SIZE_BYTES / 1_000_000,
        help="Maximum allowed CSV size in decimal MB.",
    )
    validate_parser.add_argument(
        "--max-field-bytes",
        type=int,
        help="Optional per-ground_truth UTF-8 byte budget for platform compatibility.",
    )
    validate_parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Treat empty ground_truth cells as warnings instead of errors.",
    )
    validate_parser.set_defaults(func=cmd_validate_submission)

    ensemble_parser = subparsers.add_parser(
        "ensemble-submissions",
        help="Build a conservative primary/fallback submission ensemble.",
    )
    ensemble_parser.add_argument("--primary-csv", type=Path, required=True)
    ensemble_parser.add_argument("--fallback-csv", type=Path, required=True)
    ensemble_parser.add_argument("--output-csv", type=Path, required=True)
    ensemble_parser.add_argument("--max-primary-chars", type=int, default=600)
    ensemble_parser.add_argument("--max-primary-to-fallback-ratio", type=float, default=0.10)
    ensemble_parser.add_argument(
        "--allow-non-table-fallback",
        action="store_true",
        help="Allow a fallback without an HTML table when the collapse rule matches.",
    )
    ensemble_parser.set_defaults(func=cmd_ensemble_submissions)

    overlay_parser = subparsers.add_parser(
        "overlay-submissions",
        help="Overlay checked partial recomputations onto a full submission.",
    )
    overlay_parser.add_argument("--base-csv", type=Path, required=True)
    overlay_parser.add_argument(
        "--override-csv",
        type=Path,
        required=True,
        action="append",
        help="Partial CSV to overlay; may be repeated.",
    )
    overlay_parser.add_argument("--output-csv", type=Path, required=True)
    overlay_parser.add_argument(
        "--min-override-to-base-ratio",
        type=float,
        default=0.0,
        help="Require each partial override to be at least this fraction of the base output length.",
    )
    overlay_parser.add_argument(
        "--min-override-char-gain",
        type=int,
        default=0,
        help="Require each partial override to add at least this many characters over the base output.",
    )
    overlay_parser.add_argument(
        "--max-override-duplicate-line-ratio",
        type=float,
        default=None,
        help="Reject overrides whose repeated nonblank-line ratio exceeds this optional limit.",
    )
    overlay_parser.set_defaults(func=cmd_overlay_submissions)

    compact_parser = subparsers.add_parser(
        "compact-submission",
        help="Losslessly compact oversized HTML table fields for platform compatibility.",
    )
    compact_parser.add_argument("--base-csv", type=Path, required=True)
    compact_parser.add_argument("--output-csv", type=Path, required=True)
    compact_parser.add_argument(
        "--max-field-bytes",
        type=int,
        required=True,
        help="Fail unless every CSV ground_truth field fits this UTF-8 byte budget.",
    )
    compact_parser.add_argument(
        "--all-html-tables",
        action="store_true",
        help="Rewrite every complete HTML table to equivalent pipe Markdown for platform processability.",
    )
    compact_parser.add_argument(
        "--allow-non-table-oversize",
        action="store_true",
        help="When using a table complexity budget, leave long fields without HTML tables untouched.",
    )
    compact_parser.add_argument(
        "--allow-compacted-oversize",
        action="store_true",
        help="Treat the byte budget as an HTML-table selection threshold, not a final field-size limit.",
    )
    compact_parser.set_defaults(func=cmd_compact_submission)

    local_rebuild_parser = subparsers.add_parser(
        "rebuild-cached-local-matrix",
        help="Rebuild guarded local OCR matrices from complete cached TSV tiles only.",
    )
    local_rebuild_parser.add_argument("--dataset", choices=["a", "b"], required=True)
    local_rebuild_parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    local_rebuild_parser.add_argument("--base-csv", type=Path, required=True)
    local_rebuild_parser.add_argument("--output-csv", type=Path, required=True)
    local_rebuild_parser.add_argument(
        "--local-cache-root",
        type=Path,
        required=True,
        help="Directory containing one cached local-OCR folder per image stem.",
    )
    local_rebuild_parser.add_argument(
        "--preset",
        choices=BASELINE_PRESETS,
        default="b-generalization-v3",
    )
    local_rebuild_parser.set_defaults(func=cmd_rebuild_cached_local_matrix)

    remerge_parser = subparsers.add_parser(
        "remerge-cached-tables",
        help="Rebuild multi-band table rows from cached OCR grid tiles without API calls.",
    )
    remerge_parser.add_argument("--base-csv", type=Path, required=True)
    remerge_parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        action="append",
        help="Cache root to scan; may be repeated.",
    )
    remerge_parser.add_argument("--output-csv", type=Path, required=True)
    remerge_parser.add_argument("--grid", default="5x5")
    remerge_parser.add_argument("--min-success-parts", type=int, default=4)
    remerge_parser.add_argument("--min-success-ratio", type=float, default=0.60)
    remerge_parser.add_argument("--max-duplicate-line-ratio", type=float, default=0.30)
    remerge_parser.set_defaults(func=cmd_remerge_cached_tables)

    evaluate_parser = subparsers.add_parser(
        "evaluate-train",
        help="Evaluate a training-set prediction CSV with local proxy metrics.",
    )
    evaluate_parser.add_argument(
        "--prediction-csv",
        type=Path,
        required=True,
        help="Prediction CSV with file_name,ground_truth columns for train images.",
    )
    evaluate_parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    evaluate_parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional per-sample metric CSV output.",
    )
    evaluate_parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Evaluate only prediction rows that match train labels instead of requiring all train files.",
    )
    evaluate_parser.add_argument(
        "--worst-k",
        type=int,
        default=10,
        help="Number of lowest-scoring samples to print.",
    )
    evaluate_parser.set_defaults(func=cmd_evaluate_train)

    return parser


def cmd_inspect_data(args: argparse.Namespace) -> None:
    summary = inspect_raw_data(args.raw_dir)
    for key, value in summary.items():
        print(f"{key}: {value}")


def cmd_extract(args: argparse.Namespace) -> None:
    datasets = ["train", "a", "b", "mock"] if args.dataset == "all" else [args.dataset]
    total = 0
    for dataset in datasets:
        extracted = extract_dataset(args.raw_dir, args.output_dir, dataset)
        total += len(extracted)
        print(f"{dataset}: extracted {len(extracted)} files")
    print(f"done: extracted {total} files into {args.output_dir}")


def cmd_predict(args: argparse.Namespace) -> None:
    if args.input_dir:
        records = list(iter_dir_images(args.input_dir))
        dataset_name = args.input_dir.name
    else:
        records = list(iter_dataset_images(args.raw_dir, args.dataset))
        dataset_name = args.dataset

    if not records:
        raise SystemExit("No images found. Run `inspect-data` and check data/raw first.")

    cache_suffix = f"{dataset_name}.dry-run" if args.dry_run else dataset_name
    cache_dir = args.cache_dir / cache_suffix
    output_csv = args.output_csv
    client = None
    if not args.dry_run:
        client = _build_finix_client(args)

    stats = run_prediction(
        records=records,
        client=client,
        config=PredictionConfig(
            output_csv=output_csv,
            cache_dir=cache_dir,
            dry_run=args.dry_run,
            offset=args.offset,
            limit=args.limit,
            sleep_seconds=args.sleep,
            resume=not args.no_resume,
            resume_repair_tiles=True,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            slice_overlap=args.slice_overlap,
            slice_x_overlap=args.slice_x_overlap,
            max_width=args.max_width,
            jpeg_quality=args.jpeg_quality,
            retries=args.retries,
            retry_sleep_seconds=args.retry_sleep,
            on_error=args.on_error,
            errors_csv=args.errors_csv,
        ),
    )
    print(
        "done: "
        f"discovered={stats.total_discovered}, "
        f"processed={stats.processed}, "
        f"cache_hits={stats.cache_hits}, "
        f"api_calls={stats.api_calls}, "
        f"output={stats.output_csv}"
    )


def cmd_baseline_submit(args: argparse.Namespace) -> None:
    records = list(iter_dataset_images(args.raw_dir, args.dataset))
    if not records:
        raise SystemExit("No images found. Run `inspect-data` and check data/raw first.")

    requested_file_names = getattr(args, "file_names", None)
    if requested_file_names:
        if args.offset != 0 or args.limit is not None:
            raise SystemExit("--file-names cannot be combined with --offset or --limit")
        names = _parse_str_list(requested_file_names, "--file-names")
        if len(set(names)) != len(names):
            raise SystemExit("--file-names must not contain duplicates")
        records_by_name = {record.file_name: record for record in records}
        unknown = [name for name in names if name not in records_by_name]
        if unknown:
            raise SystemExit(
                "--file-names contains unknown dataset images: " + ", ".join(unknown[:5])
            )
        records = [records_by_name[name] for name in names]

    client = _build_finix_client(args)
    baseline_config = BaselineConfig(
            output_csv=args.output_csv,
            cache_dir=args.cache_dir / f"{args.dataset}.baseline",
            offset=args.offset,
            limit=args.limit,
            crop_sizes=_parse_int_list(args.crop_sizes, "--crop-sizes"),
            anchors=_parse_str_list(args.anchors, "--anchors"),
            jpeg_quality=args.jpeg_quality,
            sleep_seconds=args.sleep,
            resume=not args.no_resume,
            retries=args.retries,
            retry_sleep_seconds=args.retry_sleep,
            min_chars=args.min_chars,
            table_repair_min_chars=args.table_repair_min_chars,
            table_repair_min_chars_per_content_pixel=getattr(
                args,
                'table_repair_min_chars_per_content_pixel',
                0.0,
            ),
            table_repair_min_gain=args.table_repair_min_gain,
            table_repair_rows=_parse_grid(args.table_repair_grid)[0],
            table_repair_cols=_parse_grid(args.table_repair_grid)[1],
            table_repair_target_tile_width=getattr(args, 'table_repair_target_tile_width', 0),
            table_repair_target_tile_height=getattr(args, 'table_repair_target_tile_height', 0),
            table_repair_overlap=args.table_repair_overlap,
            table_repair_snap_boundaries=getattr(
                args,
                'table_repair_snap_boundaries',
                False,
            ),
            table_repair_snap_x_boundaries=getattr(
                args,
                'table_repair_snap_x_boundaries',
                False,
            ),
            table_repair_snap_y_boundaries=getattr(
                args,
                'table_repair_snap_y_boundaries',
                False,
            ),
            table_repair_content_threshold=args.table_repair_content_threshold,
            table_repair_content_scale=args.table_repair_content_scale,
            table_repair_content_padding=args.table_repair_content_padding,
            table_repair_min_content_pixels=args.table_repair_min_content_pixels,
            table_repair_min_content_ratio=args.table_repair_min_content_ratio,
            table_repair_min_text_pixels=getattr(args, 'table_repair_min_text_pixels', 10),
            table_repair_header_context_height=args.table_repair_header_context_height,
            table_repair_left_context_width=args.table_repair_left_context_width,
            table_repair_min_success_parts=args.table_repair_min_success_parts,
            table_repair_min_success_ratio=getattr(args, 'table_repair_min_success_ratio', 0.0),
            table_repair_max_calls=getattr(args, 'table_repair_max_calls', 24),
            table_repair_max_failed_parts=getattr(args, 'table_repair_max_failed_parts', 0),
            table_repair_max_failed_ratio=getattr(args, 'table_repair_max_failed_ratio', 0.0),
            table_repair_max_identical_parts=getattr(args, 'table_repair_max_identical_parts', 3),
            table_repair_identical_min_chars=getattr(args, 'table_repair_identical_min_chars', 1000),
            table_repair_workers=getattr(args, 'table_repair_workers', 1),
            table_local_ocr_backend=(
                getattr(args, 'table_local_ocr_backend', None) or 'off'
            ),
            table_local_ocr_min_pixels=getattr(
                args,
                'table_local_ocr_min_pixels',
                100_000_000,
            ),
            table_local_ocr_workers=getattr(
                args,
                'table_local_ocr_workers',
                4,
            ),
            table_local_ocr_refine_saturated=getattr(
                args,
                'table_local_ocr_refine_saturated',
                False,
            ),
            table_local_ocr_max_refine_depth=getattr(
                args,
                'table_local_ocr_max_refine_depth',
                1,
            ),
            table_anchor_max_candidates=getattr(args, 'table_anchor_max_candidates', 1),
            table_anchor_max_attempts=getattr(args, 'table_anchor_max_attempts', 0),
            table_mode=getattr(args, 'table_mode', 'coverage'),
            table_target_tile_width=getattr(args, 'table_target_tile_width', 2800),
            table_target_tile_height=getattr(args, 'table_target_tile_height', 4200),
            table_max_rows=getattr(args, 'table_max_rows', 8),
            table_max_cols=getattr(args, 'table_max_cols', 10),
            table_overlap_ratio=getattr(args, 'table_overlap_ratio', 0.05),
            table_min_overlap=getattr(args, 'table_min_overlap', 80),
            table_max_blocks=getattr(args, 'table_max_blocks', 32),
            table_min_score=getattr(args, 'table_min_score', 45.0),
            table_max_duplicate_line_ratio=getattr(args, 'table_max_duplicate_line_ratio', 0.30),
            table_hybrid_min_content_ratio=getattr(args, 'table_hybrid_min_content_ratio', 0.50),
            table_refine_max_depth=getattr(args, 'table_refine_max_depth', 1),
            table_refine_rows=getattr(args, 'table_refine_rows', 2),
            table_refine_cols=getattr(args, 'table_refine_cols', 2),
            table_fragment_max_blocks=getattr(args, 'table_fragment_max_blocks', 1),
            table_fragment_refine_cols=getattr(args, 'table_fragment_refine_cols', 2),
            table_workers=getattr(args, 'table_workers', 1),
            long_aspect_threshold=getattr(args, 'long_aspect_threshold', 0.12),
            long_slice_height=args.long_slice_height,
            long_slice_overlap=args.long_slice_overlap,
            long_min_chars=args.long_min_chars,
            long_min_success_ratio=getattr(args, 'long_min_success_ratio', 1.0),
            long_max_failed_parts=getattr(args, 'long_max_failed_parts', 0),
            on_error=args.on_error,
            errors_csv=args.errors_csv,
        )
    preset = getattr(args, 'preset', None)
    if preset:
        baseline_config = apply_baseline_preset(baseline_config, preset)
    stats = run_baseline_submission(
        records=records,
        client=client,
        config=baseline_config,
    )
    print(
        "done: "
        f"discovered={stats.total_discovered}, "
        f"processed={stats.processed}, "
        f"cache_hits={stats.cache_hits}, "
        f"api_calls={stats.api_calls}, "
        f"fallbacks={stats.fallbacks}, "
        f"template_missing={stats.template_missing}, "
        f"output={stats.output_csv}"
    )


def cmd_experiment_train(args: argparse.Namespace) -> None:
    client = _build_finix_client(args)
    baseline_config = BaselineConfig(
            output_csv=args.output_dir / 'predictions.csv',
            cache_dir=args.cache_dir / args.run_id,
            jpeg_quality=args.jpeg_quality,
            sleep_seconds=args.sleep,
            resume=not args.no_resume,
            retries=args.retries,
            retry_sleep_seconds=args.retry_sleep,
            table_repair_min_chars=args.table_repair_min_chars,
            table_repair_min_chars_per_content_pixel=args.table_repair_min_chars_per_content_pixel,
            table_repair_min_gain=args.table_repair_min_gain,
            table_repair_rows=_parse_grid(args.table_repair_grid)[0],
            table_repair_cols=_parse_grid(args.table_repair_grid)[1],
            table_repair_target_tile_width=args.table_repair_target_tile_width,
            table_repair_target_tile_height=args.table_repair_target_tile_height,
            table_repair_overlap=args.table_repair_overlap,
            table_repair_snap_boundaries=getattr(
                args,
                'table_repair_snap_boundaries',
                False,
            ),
            table_repair_snap_x_boundaries=getattr(
                args,
                'table_repair_snap_x_boundaries',
                False,
            ),
            table_repair_snap_y_boundaries=getattr(
                args,
                'table_repair_snap_y_boundaries',
                False,
            ),
            table_repair_content_threshold=args.table_repair_content_threshold,
            table_repair_content_scale=args.table_repair_content_scale,
            table_repair_content_padding=args.table_repair_content_padding,
            table_repair_min_content_pixels=args.table_repair_min_content_pixels,
            table_repair_min_content_ratio=args.table_repair_min_content_ratio,
            table_repair_min_text_pixels=args.table_repair_min_text_pixels,
            table_repair_header_context_height=args.table_repair_header_context_height,
            table_repair_left_context_width=args.table_repair_left_context_width,
            table_repair_min_success_parts=args.table_repair_min_success_parts,
            table_repair_min_success_ratio=args.table_repair_min_success_ratio,
            table_repair_max_calls=args.table_repair_max_calls,
            table_repair_max_failed_parts=args.table_repair_max_failed_parts,
            table_repair_max_failed_ratio=args.table_repair_max_failed_ratio,
            table_repair_max_identical_parts=args.table_repair_max_identical_parts,
            table_repair_identical_min_chars=args.table_repair_identical_min_chars,
            table_repair_workers=(
                1
                if args.table_repair_workers is None
                else args.table_repair_workers
            ),
            table_local_ocr_backend=args.table_local_ocr_backend or "off",
            table_local_ocr_min_pixels=args.table_local_ocr_min_pixels,
            table_local_ocr_workers=args.table_local_ocr_workers,
            table_local_ocr_refine_saturated=args.table_local_ocr_refine_saturated,
            table_local_ocr_max_refine_depth=args.table_local_ocr_max_refine_depth,
            table_anchor_max_candidates=args.table_anchor_max_candidates,
            table_anchor_max_attempts=args.table_anchor_max_attempts,
            table_mode=args.table_mode,
            table_target_tile_width=args.table_target_tile_width,
            table_target_tile_height=args.table_target_tile_height,
            table_max_rows=args.table_max_rows,
            table_max_cols=args.table_max_cols,
            table_overlap_ratio=args.table_overlap_ratio,
            table_min_overlap=args.table_min_overlap,
            table_max_blocks=args.table_max_blocks,
            table_min_score=args.table_min_score,
            table_max_duplicate_line_ratio=args.table_max_duplicate_line_ratio,
            table_hybrid_min_content_ratio=args.table_hybrid_min_content_ratio,
            table_refine_max_depth=(
                1
                if args.table_refine_max_depth is None
                else args.table_refine_max_depth
            ),
            table_refine_rows=args.table_refine_rows,
            table_refine_cols=args.table_refine_cols,
            table_fragment_max_blocks=args.table_fragment_max_blocks,
            table_fragment_refine_cols=args.table_fragment_refine_cols,
            table_workers=args.table_workers,
            long_slice_height=args.long_slice_height,
            long_slice_overlap=args.long_slice_overlap,
            long_min_chars=args.long_min_chars,
            long_min_success_ratio=args.long_min_success_ratio,
            long_max_failed_parts=args.long_max_failed_parts,
            long_aspect_threshold=args.long_aspect_threshold,
            errors_csv=args.output_dir / 'errors.csv',
        )
    if args.preset:
        baseline_config = apply_baseline_preset(baseline_config, args.preset)
        # Train experiments may opt into one explicit boundary-axis variant
        # while inheriting every other frozen preset knob. Submission runs
        # retain the stricter preset-owns-all behavior in cmd_baseline_submit.
        baseline_config = replace(
            baseline_config,
            table_repair_snap_boundaries=(
                baseline_config.table_repair_snap_boundaries
                or args.table_repair_snap_boundaries
            ),
            table_repair_snap_x_boundaries=(
                baseline_config.table_repair_snap_x_boundaries
                or args.table_repair_snap_x_boundaries
            ),
            table_repair_snap_y_boundaries=(
                baseline_config.table_repair_snap_y_boundaries
                or args.table_repair_snap_y_boundaries
            ),
            table_refine_max_depth=(
                baseline_config.table_refine_max_depth
                if args.table_refine_max_depth is None
                else args.table_refine_max_depth
            ),
            table_repair_workers=(
                baseline_config.table_repair_workers
                if args.table_repair_workers is None
                else args.table_repair_workers
            ),
            table_local_ocr_backend=(
                baseline_config.table_local_ocr_backend
                if args.table_local_ocr_backend is None
                else args.table_local_ocr_backend
            ),
            table_local_ocr_min_pixels=args.table_local_ocr_min_pixels,
            table_local_ocr_workers=args.table_local_ocr_workers,
            table_local_ocr_refine_saturated=(
                baseline_config.table_local_ocr_refine_saturated
                or args.table_local_ocr_refine_saturated
            ),
            table_local_ocr_max_refine_depth=args.table_local_ocr_max_refine_depth,
        )
    result = run_train_experiment(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        split_name=args.split,
        run_id=args.run_id,
        client=client,
        baseline_config=baseline_config,
        kind=args.kind,
        offset=args.offset,
        limit=args.limit,
        file_names=tuple(args.file_name),
    )
    print(result.summary_text)
    print(
        'experiment_done: '
        f'predictions={result.predictions_csv}, '
        f'metrics={result.metrics_csv}, '
        f'manifest={result.manifest_json}, '
        f'api_calls={result.stats.api_calls}, '
        f'cache_hits={result.stats.cache_hits}'
    )


def cmd_validate_submission(args: argparse.Namespace) -> None:
    expected_file_names = _expected_submission_file_names(args.raw_dir, args.dataset)
    if not expected_file_names:
        raise SystemExit("No images found. Run `inspect-data` and check data/raw first.")

    max_size_bytes = int(args.max_size_mb * 1_000_000)
    result = validate_submission_csv(
        submission_csv=args.submission_csv,
        expected_file_names=expected_file_names,
        max_size_bytes=max_size_bytes,
        max_field_bytes=args.max_field_bytes,
        allow_empty=args.allow_empty,
        expected_label=f"{args.dataset.upper()}-list",
    )
    print(format_validation_result(result))
    if not result.ok:
        raise SystemExit(1)


def cmd_ensemble_submissions(args: argparse.Namespace) -> None:
    result = build_conservative_submission_ensemble(
        primary_csv=args.primary_csv,
        fallback_csv=args.fallback_csv,
        output_csv=args.output_csv,
        max_primary_chars=args.max_primary_chars,
        max_primary_to_fallback_ratio=args.max_primary_to_fallback_ratio,
        require_fallback_html_table=not args.allow_non_table_fallback,
    )
    print(
        "ensemble_done: "
        f"output={result.output_csv}, rows={result.row_count}, "
        f"fallbacks={result.fallback_count}"
    )


def cmd_overlay_submissions(args: argparse.Namespace) -> None:
    result = build_submission_overlay(
        base_csv=args.base_csv,
        override_csvs=args.override_csv,
        output_csv=args.output_csv,
        min_override_to_base_ratio=args.min_override_to_base_ratio,
        min_override_char_gain=args.min_override_char_gain,
        max_override_duplicate_line_ratio=args.max_override_duplicate_line_ratio,
    )
    print(
        "overlay_done: "
        f"output={result.output_csv}, rows={result.row_count}, "
        f"overrides={result.override_count}, skipped={result.skipped_count}"
    )


def cmd_compact_submission(args: argparse.Namespace) -> None:
    result = compact_submission_for_platform(
        base_csv=args.base_csv,
        output_csv=args.output_csv,
        max_field_bytes=args.max_field_bytes,
        compact_all_html_tables=args.all_html_tables,
        allow_non_table_oversize=args.allow_non_table_oversize,
        allow_compacted_oversize=args.allow_compacted_oversize,
    )
    print(
        "submission_compacted: "
        f"output={result.output_csv}, rows={result.row_count}, "
        f"compacted={result.compacted_count}, "
        f"max_field_bytes={result.max_field_bytes}, "
        f"all_html_tables={result.compact_all_html_tables}"
    )


def cmd_rebuild_cached_local_matrix(args: argparse.Namespace) -> None:
    records = list(iter_dataset_images(args.raw_dir, args.dataset))
    if not records:
        raise SystemExit("No images found. Run `inspect-data` and check data/raw first.")
    config = apply_baseline_preset(
        BaselineConfig(output_csv=args.output_csv, cache_dir=args.local_cache_root),
        args.preset,
    )
    result = rebuild_cached_local_matrix_repairs(
        records=records,
        base_csv=args.base_csv,
        output_csv=args.output_csv,
        local_cache_root=args.local_cache_root,
        config=config,
    )
    print(
        "cached_local_matrix_rebuild_done: "
        f"output={result.output_csv}, scanned={result.scanned}, "
        f"cached={result.cached_records}, selected={result.selected}"
    )


def cmd_remerge_cached_tables(args: argparse.Namespace) -> None:
    rows, cols = _parse_grid(args.grid)
    result = remerge_cached_grid_submission(
        base_csv=args.base_csv,
        cache_roots=args.cache_root,
        output_csv=args.output_csv,
        rows=rows,
        cols=cols,
        min_success_parts=args.min_success_parts,
        min_success_ratio=args.min_success_ratio,
        max_duplicate_line_ratio=args.max_duplicate_line_ratio,
    )
    print(
        "cached_table_remerge_done: "
        f"output={result.output_csv}, rows={result.row_count}, "
        f"remerged={result.remerged_count}, "
        f"skipped_cached={len(result.skipped_cached_file_names)}"
    )


def cmd_evaluate_train(args: argparse.Namespace) -> None:
    # Training labels can each be hundreds of KB.  A focused cached-OCR
    # counterfactual must not materialize every one of the 200 labels merely
    # to evaluate a handful of selected predictions.
    prediction_names = (
        set(read_prediction_csv(args.prediction_csv))
        if args.allow_subset
        else None
    )
    ground_truths = {
        record.file_name: record.read_text()
        for record in iter_train_markdowns(args.raw_dir)
        if prediction_names is None or record.file_name in prediction_names
    }
    if not ground_truths:
        raise SystemExit("No training Markdown labels found. Run `inspect-data` and check data/raw first.")

    summary = evaluate_prediction_csv(
        prediction_csv=args.prediction_csv,
        ground_truths=ground_truths,
    )
    if args.output_csv:
        write_evaluation_rows(args.output_csv, summary.rows)
    print(format_evaluation_summary(summary, worst_k=args.worst_k))
    if args.output_csv:
        print(f"detail_csv: {args.output_csv}")


def _parse_int_list(value: str, argument_name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise SystemExit(f"{argument_name} must be a comma-separated list of integers") from exc
    if not parsed:
        raise SystemExit(f"{argument_name} must contain at least one value")
    if any(item <= 0 for item in parsed):
        raise SystemExit(f"{argument_name} values must be positive")
    return parsed


def _parse_str_list(value: str, argument_name: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        raise SystemExit(f"{argument_name} must contain at least one value")
    return parsed


def _build_finix_client(args: argparse.Namespace) -> FinixDocClient | RotatingFinixDocClient:
    raw_user_ids = getattr(args, "user_ids", None)
    user_ids = (
        _parse_str_list(raw_user_ids, "--user-ids")
        if raw_user_ids
        else (
            tuple(sorted(ALLOWED_USER_IDS))
            if getattr(args, 'preset', None)
            else (args.user_id,)
        )
    )
    clients = tuple(
        FinixDocClient(
            user_id=user_id,
            api_key=args.api_key,
            timeout=args.timeout,
        )
        for user_id in user_ids
    )
    return clients[0] if len(clients) == 1 else RotatingFinixDocClient(clients)


def _parse_grid(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(" ", "")
    if "x" not in normalized:
        raise SystemExit("--table-repair-grid must be formatted as ROWSxCOLS")
    row_text, col_text = normalized.split("x", 1)
    try:
        rows = int(row_text)
        cols = int(col_text)
    except ValueError as exc:
        raise SystemExit("--table-repair-grid rows and cols must be integers") from exc
    if rows <= 0 or cols <= 0:
        raise SystemExit("--table-repair-grid rows and cols must be positive")
    return rows, cols


def _mock_template_names_for_full_run(args: argparse.Namespace) -> tuple[str, ...] | None:
    if args.no_mock_template or args.offset != 0 or args.limit is not None:
        return None
    mock_source = submission_template_source(args.raw_dir, args.dataset)
    if not mock_source:
        return None
    return tuple(row["file_name"] for row in read_mock_submission(mock_source))


def _expected_submission_file_names(raw_dir: Path, dataset: str) -> tuple[str, ...]:
    mock_source = submission_template_source(raw_dir, dataset)
    if mock_source:
        return tuple(row["file_name"] for row in read_mock_submission(mock_source))
    return tuple(record.file_name for record in iter_dataset_images(raw_dir, dataset))
