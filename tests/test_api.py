from __future__ import annotations

import json
import unittest

from afac_pipeline.api import FinixDocError, normalize_markdown_payload


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


if __name__ == "__main__":
    unittest.main()
