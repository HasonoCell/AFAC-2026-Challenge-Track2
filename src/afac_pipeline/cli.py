from __future__ import annotations

import argparse
import os
from pathlib import Path

from .api import DEFAULT_API_KEY, FinixDocClient
from .baseline import BaselineConfig, run_baseline_submission
from .datasets import (
    extract_dataset,
    inspect_raw_data,
    iter_dataset_images,
    iter_dir_images,
    mock_submission_source,
    read_mock_submission,
)
from .pipeline import PredictionConfig, run_prediction
from .submission import DEFAULT_MAX_SIZE_BYTES, format_validation_result, validate_submission_csv


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
        choices=["train", "a", "mock", "all"],
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
        choices=["train", "a"],
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
        help="Generate a low-call-count real-API A-list submission baseline.",
    )
    baseline_parser.add_argument(
        "--dataset",
        choices=["a"],
        default="a",
        help="Dataset to process.",
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
        "--api-key",
        default=os.environ.get("FINIXDOC_API_KEY", DEFAULT_API_KEY),
        help="FinixDoc-VL apiKey. Defaults to the official competition key.",
    )
    baseline_parser.add_argument("--timeout", type=float, default=300.0)
    baseline_parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep after API calls.")
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
        default=100,
        help=(
            "If a table crop output is shorter than this many characters, retry "
            "with content-box grid slices. Use 0 to disable."
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
        "--table-repair-overlap",
        type=int,
        default=120,
        help="Pixel overlap in both directions for content-box table repair slices.",
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
        "--table-repair-min-success-parts",
        type=int,
        default=4,
        help="Minimum successful grid slices required before accepting table repair.",
    )
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
        choices=["a"],
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
        "--allow-empty",
        action="store_true",
        help="Treat empty ground_truth cells as warnings instead of errors.",
    )
    validate_parser.set_defaults(func=cmd_validate_submission)

    return parser


def cmd_inspect_data(args: argparse.Namespace) -> None:
    summary = inspect_raw_data(args.raw_dir)
    for key, value in summary.items():
        print(f"{key}: {value}")


def cmd_extract(args: argparse.Namespace) -> None:
    datasets = ["train", "a", "mock"] if args.dataset == "all" else [args.dataset]
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
        client = FinixDocClient(
            user_id=args.user_id,
            api_key=args.api_key,
            timeout=args.timeout,
        )

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

    client = FinixDocClient(
        user_id=args.user_id,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    stats = run_baseline_submission(
        records=records,
        client=client,
        config=BaselineConfig(
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
            table_repair_min_gain=args.table_repair_min_gain,
            table_repair_rows=_parse_grid(args.table_repair_grid)[0],
            table_repair_cols=_parse_grid(args.table_repair_grid)[1],
            table_repair_overlap=args.table_repair_overlap,
            table_repair_content_threshold=args.table_repair_content_threshold,
            table_repair_content_scale=args.table_repair_content_scale,
            table_repair_content_padding=args.table_repair_content_padding,
            table_repair_min_success_parts=args.table_repair_min_success_parts,
            long_slice_height=args.long_slice_height,
            long_slice_overlap=args.long_slice_overlap,
            long_min_chars=args.long_min_chars,
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
        f"fallbacks={stats.fallbacks}, "
        f"template_missing={stats.template_missing}, "
        f"output={stats.output_csv}"
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
        allow_empty=args.allow_empty,
    )
    print(format_validation_result(result))
    if not result.ok:
        raise SystemExit(1)


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
    mock_source = mock_submission_source(args.raw_dir)
    if not mock_source:
        return None
    return tuple(row["file_name"] for row in read_mock_submission(mock_source))


def _expected_submission_file_names(raw_dir: Path, dataset: str) -> tuple[str, ...]:
    if dataset == "a":
        mock_source = mock_submission_source(raw_dir)
        if mock_source:
            return tuple(row["file_name"] for row in read_mock_submission(mock_source))
    return tuple(record.file_name for record in iter_dataset_images(raw_dir, dataset))
