"""Submission CSV validation utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .api import FinixDocError, normalize_markdown_payload


SUBMISSION_COLUMNS = ["file_name", "ground_truth"]
DEFAULT_MAX_SIZE_BYTES = 100_000_000


@dataclass(frozen=True)
class SubmissionValidationResult:
    submission_csv: Path
    file_size_bytes: int
    row_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_submission_csv(
    *,
    submission_csv: Path,
    expected_file_names: Iterable[str],
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
    allow_empty: bool = False,
) -> SubmissionValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    expected_names = set(expected_file_names)

    if not submission_csv.exists():
        return SubmissionValidationResult(
            submission_csv=submission_csv,
            file_size_bytes=0,
            row_count=0,
            errors=(f"submission file does not exist: {submission_csv}",),
            warnings=(),
        )

    file_size = submission_csv.stat().st_size
    if file_size > max_size_bytes:
        errors.append(
            f"submission file is {file_size} bytes, exceeding limit {max_size_bytes} bytes"
        )

    try:
        with submission_csv.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except csv.Error as exc:
        return SubmissionValidationResult(
            submission_csv=submission_csv,
            file_size_bytes=file_size,
            row_count=0,
            errors=tuple(errors + [f"failed to parse CSV: {exc}"]),
            warnings=tuple(warnings),
        )
    except UnicodeDecodeError as exc:
        return SubmissionValidationResult(
            submission_csv=submission_csv,
            file_size_bytes=file_size,
            row_count=0,
            errors=tuple(errors + [f"CSV must be UTF-8 encoded: {exc}"]),
            warnings=tuple(warnings),
        )

    if fieldnames != SUBMISSION_COLUMNS:
        errors.append(
            "CSV header must be exactly "
            f"{','.join(SUBMISSION_COLUMNS)}; got {','.join(fieldnames) or '<empty>'}"
        )

    seen: set[str] = set()
    duplicate_names: set[str] = set()
    actual_names: set[str] = set()
    empty_names = 0
    empty_outputs: list[str] = []
    error_outputs: list[str] = []
    malformed_outputs: list[str] = []

    for row_index, row in enumerate(rows, start=2):
        file_name = (row.get("file_name") or "").strip()
        ground_truth = row.get("ground_truth") or ""
        if not file_name:
            empty_names += 1
            continue
        if file_name in seen:
            duplicate_names.add(file_name)
        seen.add(file_name)
        actual_names.add(file_name)

        if not ground_truth.strip():
            empty_outputs.append(file_name)
        if ground_truth.lstrip().startswith("ERROR:"):
            error_outputs.append(file_name)
        issue = _validate_ground_truth_text(ground_truth)
        if issue:
            malformed_outputs.append(f"{file_name} (row {row_index}: {issue})")

    missing_names = sorted(expected_names - actual_names)
    unknown_names = sorted(actual_names - expected_names)

    if empty_names:
        errors.append(f"{empty_names} row(s) have an empty file_name")
    if duplicate_names:
        errors.append(
            f"duplicate file_name values: {_format_sample(sorted(duplicate_names))}"
        )
    if missing_names:
        errors.append(f"missing expected A-list files: {_format_sample(missing_names)}")
    if unknown_names:
        errors.append(f"unknown file_name values: {_format_sample(unknown_names)}")
    if empty_outputs:
        message = f"empty ground_truth values: {_format_sample(empty_outputs)}"
        if allow_empty:
            warnings.append(message)
        else:
            errors.append(message)
    if error_outputs:
        errors.append(f"ground_truth contains ERROR markers: {_format_sample(error_outputs)}")
    if malformed_outputs:
        errors.append(
            "malformed ground_truth values: " + _format_sample(malformed_outputs, limit=5)
        )

    if len(rows) != len(expected_names):
        errors.append(
            f"row count is {len(rows)}, expected {len(expected_names)} for current A-list data"
        )

    if file_size > 90_000_000 and file_size <= max_size_bytes:
        warnings.append(
            f"submission file is {file_size} bytes; close to 100MB platform limit"
        )

    return SubmissionValidationResult(
        submission_csv=submission_csv,
        file_size_bytes=file_size,
        row_count=len(rows),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def format_validation_result(result: SubmissionValidationResult) -> str:
    status = "OK" if result.ok else "FAIL"
    lines = [
        f"status: {status}",
        f"submission_csv: {result.submission_csv}",
        f"file_size_bytes: {result.file_size_bytes}",
        f"row_count: {result.row_count}",
    ]
    if result.errors:
        lines.append("errors:")
        lines.extend(f"- {error}" for error in result.errors)
    if result.warnings:
        lines.append("warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)


def _validate_ground_truth_text(text: str) -> str | None:
    if "MVP dry-run placeholder" in text:
        return "dry-run placeholder output must not be submitted"

    try:
        normalize_markdown_payload(text)
    except FinixDocError as exc:
        return str(exc)

    lowered = text.lower()
    if lowered.count("<table") != lowered.count("</table>"):
        return "HTML table tag count is not balanced"
    return None


def _format_sample(values: list[str], limit: int = 10) -> str:
    sample = values[:limit]
    suffix = "" if len(values) <= limit else f" ... (+{len(values) - limit} more)"
    return ", ".join(sample) + suffix
