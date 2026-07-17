from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from afac_pipeline.submission import validate_submission_csv


class ValidateSubmissionCsvTest(unittest.TestCase):
    def test_accepts_complete_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.csv"
            _write_rows(
                path,
                [
                    {"file_name": "a.jpg", "ground_truth": "<table></table>\n"},
                    {"file_name": "b.jpg", "ground_truth": "# ok\n"},
                ],
            )

            result = validate_submission_csv(
                submission_csv=path,
                expected_file_names=["a.jpg", "b.jpg"],
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.row_count, 2)

    def test_rejects_missing_extra_empty_and_malformed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.csv"
            _write_rows(
                path,
                [
                    {"file_name": "a.jpg", "ground_truth": ""},
                    {"file_name": "extra.jpg", "ground_truth": "<table>"},
                    {"file_name": "a.jpg", "ground_truth": "ERROR: failed"},
                    {"file_name": "dry.jpg", "ground_truth": "MVP dry-run placeholder."},
                ],
            )

            result = validate_submission_csv(
                submission_csv=path,
                expected_file_names=["a.jpg", "b.jpg", "dry.jpg"],
            )

            self.assertFalse(result.ok)
            joined = "\n".join(result.errors)
            self.assertIn("duplicate file_name", joined)
            self.assertIn("missing expected A-list files", joined)
            self.assertIn("unknown file_name", joined)
            self.assertIn("empty ground_truth", joined)
            self.assertIn("ERROR markers", joined)
            self.assertIn("HTML table tag count is not balanced", joined)
            self.assertIn("dry-run placeholder", joined)

    def test_allow_empty_downgrades_empty_outputs_to_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.csv"
            _write_rows(path, [{"file_name": "a.jpg", "ground_truth": ""}])

            result = validate_submission_csv(
                submission_csv=path,
                expected_file_names=["a.jpg"],
                allow_empty=True,
            )

            self.assertTrue(result.ok)
            self.assertIn("empty ground_truth", "\n".join(result.warnings))

    def test_rejects_wrong_header_and_oversized_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.csv"
            path.write_text("file_name,prediction\na.jpg,ok\n", encoding="utf-8")

            result = validate_submission_csv(
                submission_csv=path,
                expected_file_names=["a.jpg"],
                max_size_bytes=1,
            )

            self.assertFalse(result.ok)
            joined = "\n".join(result.errors)
            self.assertIn("exceeding limit", joined)
            self.assertIn("CSV header must be exactly", joined)


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["file_name", "ground_truth"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
