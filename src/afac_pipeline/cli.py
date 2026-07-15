from __future__ import annotations

import argparse
import os
from pathlib import Path

from .api import DEFAULT_API_KEY, FinixDocClient
from .datasets import (
    extract_dataset,
    inspect_raw_data,
    iter_dataset_images,
    iter_dir_images,
)
from .pipeline import PredictionConfig, run_prediction


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
            slice_overlap=args.slice_overlap,
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
