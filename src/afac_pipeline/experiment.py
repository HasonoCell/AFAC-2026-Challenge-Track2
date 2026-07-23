"""Train-set experiment orchestration for score-driven OCR iteration."""

from __future__ import annotations

import json
from html import unescape
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import islice
from pathlib import Path
from typing import Any

from .baseline import BaselineConfig, BaselineStats, run_baseline_submission
from .datasets import iter_dataset_images, iter_train_markdowns
from .evaluation import evaluate_prediction_csv, format_evaluation_summary, write_evaluation_rows
from .images import image_dimensions


@dataclass(frozen=True)
class ExperimentSplit:
    name: str
    file_names: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentResult:
    output_dir: Path
    predictions_csv: Path
    metrics_csv: Path
    manifest_json: Path
    errors_csv: Path
    stats: BaselineStats
    summary_text: str


def build_train_splits(raw_dir: Path) -> dict[str, ExperimentSplit]:
    records = list(iter_dataset_images(raw_dir, 'train'))
    # Keep lightweight lazy records instead of materializing every huge table
    # label at once. The table half alone can contain tens of millions of HTML
    # characters, which made split construction compete with image profiling
    # for memory and occasionally get killed before an experiment started.
    ground_truths = {
        record.file_name: record
        for record in iter_train_markdowns(raw_dir)
    }
    rows: list[dict[str, object]] = []
    for record in records:
        markdown_record = ground_truths.get(record.file_name)
        if markdown_record is None:
            continue
        markdown = markdown_record.read_text()
        width, height = image_dimensions(image_bytes=record.read_bytes())
        rows.append(
            {
                'file_name': record.file_name,
                'kind': 'long' if width / max(1, height) <= 0.12 else 'table',
                'pixels': width * height,
                'gt_length': len(markdown),
                'gt_tables': markdown.lower().count('<table'),
                'family': _family_key(markdown),
            }
        )

    split_names: dict[str, list[str]] = {'dev': [], 'validation': [], 'rest': []}
    for kind in ('long', 'table'):
        group = [row for row in rows if row['kind'] == kind]
        assigned = _assign_family_grouped_splits(
            group,
            targets={'dev': 20, 'validation': 20, 'rest': 60},
        )
        for split_name, assigned_rows in assigned.items():
            split_names[split_name].extend(
                str(row['file_name']) for row in assigned_rows
            )

    all_names = sorted(str(row['file_name']) for row in rows)
    return {
        'dev': ExperimentSplit('dev', tuple(sorted(split_names['dev']))),
        'validation': ExperimentSplit('validation', tuple(sorted(split_names['validation']))),
        'rest': ExperimentSplit('rest', tuple(sorted(split_names['rest']))),
        'all': ExperimentSplit('all', tuple(all_names)),
    }


def run_train_experiment(
    *,
    raw_dir: Path,
    output_dir: Path,
    split_name: str,
    run_id: str,
    client: Any,
    baseline_config: BaselineConfig,
    kind: str = 'all',
    offset: int = 0,
    limit: int | None = None,
    file_names: tuple[str, ...] = (),
) -> ExperimentResult:
    splits = build_train_splits(raw_dir)
    if split_name not in splits:
        allowed = ', '.join(sorted(splits))
        raise ValueError(f'unknown split {split_name!r}; expected one of {allowed}')
    if kind not in {'all', 'long', 'table'}:
        raise ValueError("kind must be one of: all, long, table")
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if file_names and (offset != 0 or limit is not None):
        raise ValueError("file_names cannot be combined with offset or limit")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = output_dir / 'predictions.csv'
    metrics_csv = output_dir / 'metrics.csv'
    manifest_json = output_dir / 'manifest.json'
    errors_csv = output_dir / 'errors.csv'

    wanted = set(splits[split_name].file_names)
    requested_file_names = set(file_names)
    unknown_requested = requested_file_names - wanted
    if unknown_requested:
        raise ValueError(
            "requested files are not members of split "
            f"{split_name!r}: {', '.join(sorted(unknown_requested))}"
        )
    records = [record for record in iter_dataset_images(raw_dir, 'train') if record.file_name in wanted]
    if requested_file_names:
        records = [
            record for record in records
            if record.file_name in requested_file_names
        ]
    if kind != 'all':
        records = [
            record
            for record in records
            if _record_kind(record, baseline_config) == kind
        ]
    if requested_file_names and {record.file_name for record in records} != requested_file_names:
        raise ValueError(
            "one or more requested files do not match the selected document kind"
        )
    records_after_offset = records[offset:]
    records = records_after_offset[:limit] if limit is not None else records_after_offset
    wanted = {record.file_name for record in records}
    ground_truths = {
        record.file_name: record.read_text()
        for record in iter_train_markdowns(raw_dir)
        if record.file_name in wanted
    }

    started = time.time()
    stats = run_baseline_submission(
        records=records,
        client=client,
        config=replace(
            baseline_config,
            output_csv=predictions_csv,
            errors_csv=errors_csv,
        ),
    )
    elapsed = time.time() - started

    summary = evaluate_prediction_csv(prediction_csv=predictions_csv, ground_truths=ground_truths)
    write_evaluation_rows(metrics_csv, summary.rows)
    summary_text = format_evaluation_summary(summary)

    manifest = {
        'run_id': run_id,
        'route_version': 'family-isolated-adaptive-v3',
        'split_schema': 'family-grouped-v3',
        'split': split_name,
        'kind': kind,
        'offset': offset,
        'limit': limit,
        'requested_file_names': list(file_names),
        'split_counts': {name: len(split.file_names) for name, split in splits.items()},
        'selected_count': len(records),
        'elapsed_seconds': round(elapsed, 3),
        'api_calls': stats.api_calls,
        'cache_hits': stats.cache_hits,
        'output_files': {
            'predictions_csv': str(predictions_csv),
            'metrics_csv': str(metrics_csv),
            'errors_csv': str(errors_csv),
        },
        'baseline_config': _jsonable(asdict(baseline_config)),
        'summary': {
            'evaluated': summary.evaluated,
            'overall_mean': summary.overall_mean,
            'overall_median': summary.overall_median,
            'text_mean': summary.text_mean,
            'table_mean': summary.table_mean,
            'read_order_mean': summary.read_order_mean,
            'missing_predictions': len(summary.missing_predictions),
            'unknown_predictions': len(summary.unknown_predictions),
        },
    }
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return ExperimentResult(
        output_dir=output_dir,
        predictions_csv=predictions_csv,
        metrics_csv=metrics_csv,
        manifest_json=manifest_json,
        errors_csv=errors_csv,
        stats=stats,
        summary_text=summary_text,
    )


def _assign_family_grouped_splits(
    rows: list[dict[str, object]],
    *,
    targets: dict[str, int],
) -> dict[str, list[dict[str, object]]]:
    """Create deterministic stratified splits without document-family leakage."""

    if sum(targets.values()) != len(rows):
        raise ValueError("split targets must sum to the number of rows")
    lengths = sorted(int(row['gt_length']) for row in rows)
    pixels = sorted(int(row['pixels']) for row in rows)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row['family'])].append(row)

    def stratum(row: dict[str, object]) -> tuple[int, int, int]:
        table_count = int(row['gt_tables'])
        return (
            min(table_count, 3),
            _quantile_bucket(int(row['gt_length']), lengths, 8),
            _quantile_bucket(int(row['pixels']), pixels, 8),
        )

    total_strata = Counter(stratum(row) for row in rows)
    split_order = tuple(targets)
    selected: dict[str, list[dict[str, object]]] = {
        split_name: [] for split_name in split_order
    }
    selected_strata = {split_name: Counter() for split_name in split_order}
    total_rows = len(rows)

    family_groups = sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    for _, family_rows in family_groups:
        family_rows = sorted(family_rows, key=lambda row: str(row['file_name']))
        family_strata = Counter(stratum(row) for row in family_rows)
        group_size = len(family_rows)

        def assignment_score(split_name: str) -> tuple[float, float, int]:
            target = targets[split_name]
            new_size = len(selected[split_name]) + group_size
            overflow = max(0, new_size - target)
            utilization = new_size / max(1, target)
            fraction = target / total_rows
            stratum_pressure = 0.0
            for key, count in family_strata.items():
                desired = total_strata[key] * fraction
                stratum_pressure += (
                    selected_strata[split_name][key] + count
                ) / max(1.0, desired)
            stratum_pressure /= max(1, len(family_strata))
            return (overflow * 1000.0 + utilization + stratum_pressure, utilization, split_order.index(split_name))

        chosen = min(split_order, key=assignment_score)
        selected[chosen].extend(family_rows)
        selected_strata[chosen].update(family_strata)

    return selected


def _quantile_bucket(value: int, sorted_values: list[int], buckets: int) -> int:
    if not sorted_values or buckets <= 1:
        return 0
    lower_count = sum(candidate < value for candidate in sorted_values)
    return min(buckets - 1, lower_count * buckets // len(sorted_values))


def _family_key(markdown: str) -> str:
    table_start = markdown.lower().find('<table')
    preamble = markdown[:table_start] if table_start >= 0 else markdown
    for line in preamble.splitlines():
        normalized = _normalize_family_text(
            line,
            mask_numbers=True,
            compact=True,
        )
        if normalized:
            return normalized[:96]

    cells = (
        match.group(1)
        for match in islice(
            re.finditer(
                r'<t[dh]\b[^>]*>(.*?)</t[dh]>',
                markdown,
                flags=re.IGNORECASE | re.DOTALL,
            ),
            16,
        )
    )
    normalized_cells = [
        _normalize_family_text(re.sub(r'<[^>]+>', ' ', cell), mask_numbers=True)
        for cell in cells
    ]
    signature = ' | '.join(cell for cell in normalized_cells if cell)
    if signature:
        return signature[:256]

    for line in markdown.splitlines():
        normalized = _normalize_family_text(
            line,
            mask_numbers=True,
            compact=True,
        )
        if normalized:
            return normalized[:96]
    return ''


def _normalize_family_text(
    text: str,
    *,
    mask_numbers: bool = False,
    compact: bool = False,
) -> str:
    normalized = unicodedata.normalize('NFKC', unescape(text))
    normalized = ' '.join(normalized.strip('# ').split())
    if mask_numbers:
        normalized = re.sub(r'\d+(?:[.,]\d+)?', '#', normalized)
    if compact:
        # Product-title families should not split on OCR/layout variants such
        # as spaces around parentheses, fullwidth punctuation, or edition
        # years. Keep letters/CJK text while discarding formatting and the
        # masked numeric placeholders. Table schema cells use compact=False,
        # so their column boundaries remain explicit.
        normalized = re.sub(r'[\W_]+', '', normalized, flags=re.UNICODE)
    return normalized


def _record_kind(record: Any, config: BaselineConfig) -> str:
    width, height = image_dimensions(image_bytes=record.read_bytes())
    return 'long' if width / max(1, height) <= config.long_aspect_threshold else 'table'


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
