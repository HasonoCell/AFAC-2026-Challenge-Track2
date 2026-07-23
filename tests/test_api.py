from __future__ import annotations

import json
import unittest

from afac_pipeline.api import (
    FinixDocClient,
    FinixDocError,
    RotatingFinixDocClient,
    html_table_structure_issue,
    normalize_markdown_payload,
    repair_implicit_html_table_closures,
)


class NormalizeMarkdownPayloadTest(unittest.TestCase):
    def test_extracts_nested_chat_completion_content(self) -> None:
        chat_completion = {
            "object": "chat.completion",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "```markdown\n# Title\n\nBody\n```",
                    }
                }
            ],
        }
        response = {
            "success": True,
            "result": json.dumps(chat_completion, ensure_ascii=False),
        }

        actual = normalize_markdown_payload(json.dumps(response, ensure_ascii=False))

        self.assertEqual(actual, "# Title\n\nBody\n")

    def test_plain_markdown_is_unchanged_except_trailing_newline(self) -> None:
        self.assertEqual(normalize_markdown_payload("# Title"), "# Title")

    def test_detects_unclosed_html_table_cell(self) -> None:
        issue = html_table_structure_issue("<table><tr><td>broken</td></table>")

        self.assertEqual(issue, "HTML table structure closes </table> while <tr> is open")

    def test_rejects_unclosed_markdown_fence(self) -> None:
        chat_completion = {
            "choices": [
                {
                    "message": {
                        "content": "```markdown\n<table><tr><th>",
                    }
                }
            ]
        }

        with self.assertRaisesRegex(FinixDocError, "appears truncated"):
            normalize_markdown_payload(json.dumps(chat_completion))

    def test_can_leniently_keep_unclosed_fence_and_balance_tables(self) -> None:
        chat_completion = {
            "choices": [
                {
                    "message": {
                        "content": "```markdown\n<table><tr><td>ok</td></tr>",
                    }
                }
            ]
        }

        actual = normalize_markdown_payload(
            json.dumps(chat_completion),
            allow_unclosed_fence=True,
            balance_html_tables=True,
        )

        self.assertEqual(actual, "<table><tr><td>ok</td></tr>\n</table>\n")

    def test_repairs_only_html_implied_cell_and_row_closures(self) -> None:
        malformed = (
            "<table><tr><td>A<td>B</td></tr>"
            "<tr><td>C</tr></table>"
        )

        repaired = repair_implicit_html_table_closures(malformed)

        self.assertEqual(
            repaired,
            "<table><tr><td>A</td><td>B</td></tr>"
            "<tr><td>C</td></tr></table>",
        )
        self.assertIsNone(html_table_structure_issue(repaired))

    def test_lenient_balance_does_not_repair_a_table_missing_its_close(self) -> None:
        truncated = "<table><tr><td>broken"

        balanced = normalize_markdown_payload(
            json.dumps({"result": truncated}),
            balance_html_tables=True,
        )

        self.assertIsNotNone(html_table_structure_issue(balanced))

    def test_rotating_client_cycles_through_clients(self) -> None:
        clients = (
            FinixDocClient(user_id="finixA1001"),
            FinixDocClient(user_id="finixB2002"),
        )
        rotating = RotatingFinixDocClient(clients)
        calls: list[str] = []

        def fake_call(self: FinixDocClient, *args: object, **kwargs: object) -> str:
            calls.append(self.user_id)
            return self.user_id

        from unittest.mock import patch

        with patch.object(FinixDocClient, "call_with_file", fake_call):
            results = [rotating.call_with_file("sample.jpg", b"x") for _ in range(3)]

        self.assertEqual(results, ["finixA1001", "finixB2002", "finixA1001"])
        self.assertEqual(calls, results)


if __name__ == "__main__":
    unittest.main()
