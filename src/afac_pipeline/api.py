"""FinixDoc-VL API client and response normalization."""

from __future__ import annotations

import json
import mimetypes
import re
import secrets
import ssl
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import certifi


FINIXDOC_API_URL = "https://finixdocapi.alipay.com/api/finix_doc/call_with_file"
DEFAULT_API_KEY = "F935A5503983FB19F26FA3F00A94EBF9"
ALLOWED_USER_IDS = {
    "finixA1001",
    "finixB2002",
    "finixC3003",
    "finixD4004",
    "finixE5005",
}


class FinixDocError(RuntimeError):
    """Raised when FinixDoc-VL returns an unusable response."""


_TABLE_STRUCTURE_TAG = re.compile(
    r"<\s*(/?)\s*(table|tr|td|th)(?:\s[^<>]*)?>",
    flags=re.IGNORECASE,
)


def html_table_structure_issue(text: str) -> str | None:
    """Return a concise error for explicitly malformed table markup.

    Table results written by this pipeline use explicit close tags. Counting
    only ``<table>`` tags lets an unclosed row or cell reach a downstream CSV
    consumer, where it may abort the whole evaluation. This narrow checker
    leaves unrelated Markdown and non-table HTML untouched.
    """

    stack: list[str] = []
    for match in _TABLE_STRUCTURE_TAG.finditer(text):
        closing, tag = match.group(1), match.group(2).lower()
        if not closing:
            stack.append(tag)
            continue
        if not stack:
            return f"HTML table structure closes </{tag}> without an opening tag"
        opening = stack.pop()
        if opening != tag:
            return f"HTML table structure closes </{tag}> while <{opening}> is open"
    if stack:
        return "HTML table structure has unclosed " + ", ".join(
            f"<{tag}>" for tag in stack[-3:]
        )
    return None


def repair_implicit_html_table_closures(text: str) -> str:
    """Make HTML's optional cell/row end tags explicit.

    The labelled corpus contains otherwise complete tables that use constructs
    such as ``<td><td>`` or close ``</tr>`` while the final cell is still open.
    Browsers accept those forms, but the pipeline deliberately emits explicit
    markup before CSV validation.  This helper only inserts closures implied by
    a following table tag; it never invents a missing ``</table>``.
    """

    output: list[str] = []
    stack: list[str] = []
    cursor = 0
    for match in _TABLE_STRUCTURE_TAG.finditer(text):
        output.append(text[cursor : match.start()])
        closing, tag = match.group(1), match.group(2).lower()
        if not closing:
            if tag == "tr":
                if stack and stack[-1] in {"td", "th"}:
                    output.append(f"</{stack.pop()}>")
                if stack and stack[-1] == "tr":
                    output.append("</tr>")
                    stack.pop()
            elif tag in {"td", "th"} and stack and stack[-1] in {"td", "th"}:
                output.append(f"</{stack.pop()}>")
            output.append(match.group(0))
            stack.append(tag)
        else:
            if tag == "tr" and stack and stack[-1] in {"td", "th"}:
                output.append(f"</{stack.pop()}>")
            elif tag == "table":
                if stack and stack[-1] in {"td", "th"}:
                    output.append(f"</{stack.pop()}>")
                if stack and stack[-1] == "tr":
                    output.append("</tr>")
                    stack.pop()
            output.append(match.group(0))
            if stack and stack[-1] == tag:
                stack.pop()
            else:
                # Preserve unfamiliar mismatches for the strict validator to
                # reject; repairing arbitrary nesting could hide truncation.
                stack.append(f"mismatch:{tag}")
        cursor = match.end()
    output.append(text[cursor:])
    return "".join(output)


@dataclass
class RotatingFinixDocClient:
    """Round-robin requests across official FinixDoc user IDs.

    Rotation is sequential and deterministic.  It improves throughput when the
    service enforces rate limits per whitelisted ``userId`` while preserving
    the single-client interface used by the prediction pipeline.
    """

    clients: tuple["FinixDocClient", ...]
    _next_index: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("RotatingFinixDocClient requires at least one client")

    def call_with_file(
        self,
        file_name: str,
        file_bytes: bytes,
        *,
        allow_unclosed_fence: bool = False,
        balance_html_tables: bool = False,
    ) -> str:
        with self._lock:
            client = self.clients[self._next_index]
            self._next_index = (self._next_index + 1) % len(self.clients)
        return client.call_with_file(
            file_name,
            file_bytes,
            allow_unclosed_fence=allow_unclosed_fence,
            balance_html_tables=balance_html_tables,
        )


@dataclass(frozen=True)
class FinixDocClient:
    user_id: str
    api_key: str = DEFAULT_API_KEY
    endpoint: str = FINIXDOC_API_URL
    timeout: float = 180.0

    def __post_init__(self) -> None:
        if self.user_id not in ALLOWED_USER_IDS:
            allowed = ", ".join(sorted(ALLOWED_USER_IDS))
            raise ValueError(f"Invalid userId {self.user_id!r}. Allowed values: {allowed}")

    def call_with_file(
        self,
        file_name: str,
        file_bytes: bytes,
        *,
        allow_unclosed_fence: bool = False,
        balance_html_tables: bool = False,
    ) -> str:
        body, content_type = _build_multipart_body(
            fields={
                "userId": self.user_id,
                "apiKey": self.api_key,
                "fileName": file_name,
            },
            files={
                "file": (file_name, file_bytes, _guess_content_type(file_name)),
            },
        )
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
                context=_ssl_context(),
            ) as response:
                response_text = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise FinixDocError(f"HTTP {exc.code} from FinixDoc-VL: {error_body}") from exc
        except TimeoutError as exc:
            raise FinixDocError(
                "FinixDoc-VL request timed out. Try a larger --timeout or use "
                "--slice-height to split large images."
            ) from exc
        except urllib.error.URLError as exc:
            raise FinixDocError(f"Failed to call FinixDoc-VL: {exc}") from exc

        markdown = _extract_markdown(
            response_text,
            allow_unclosed_fence=allow_unclosed_fence,
            balance_html_tables=balance_html_tables,
        )
        if not markdown.strip():
            raise FinixDocError(f"FinixDoc-VL returned an empty result for {file_name}")
        return markdown


def _guess_content_type(file_name: str) -> str:
    return mimetypes.guess_type(file_name)[0] or "application/octet-stream"


def _ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def _build_multipart_body(
    *,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = f"----afac2026-{secrets.token_hex(16)}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    for name, (file_name, file_bytes, content_type) in files.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{file_name}"\r\n'
                ).encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                file_bytes,
                b"\r\n",
            ]
        )

    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _extract_markdown(
    response_text: str,
    *,
    allow_unclosed_fence: bool = False,
    balance_html_tables: bool = False,
) -> str:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return response_text

    if isinstance(payload, dict):
        success = payload.get("success")
        if success is False:
            message = payload.get("message") or payload.get("error") or payload
            raise FinixDocError(f"FinixDoc-VL returned an error: {message}")
        if payload.get("code") not in (None, "SUCCESS", 0, "0"):
            message = payload.get("message") or payload
            raise FinixDocError(f"FinixDoc-VL returned an error: {message}")

    extracted = _find_markdown_value(payload)
    if extracted is None:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return _clean_markdown(
        extracted,
        allow_unclosed_fence=allow_unclosed_fence,
        balance_html_tables=balance_html_tables,
    )


def _find_markdown_value(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                nested = json.loads(stripped)
            except json.JSONDecodeError:
                pass
            else:
                return _find_markdown_value(nested)
        return value
    if isinstance(value, list):
        parts = [_find_markdown_value(item) for item in value]
        parts = [part for part in parts if part]
        return "\n\n".join(parts) if parts else None
    if not isinstance(value, dict):
        return None

    preferred_keys = (
        "markdown",
        "md",
        "ground_truth",
        "result",
        "content",
        "text",
        "data",
        "choices",
        "message",
    )
    for key in preferred_keys:
        if key in value:
            found = _find_markdown_value(value[key])
            if found:
                return found

    for nested in value.values():
        if isinstance(nested, (dict, list)):
            found = _find_markdown_value(nested)
            if found:
                return found
    return None


def normalize_markdown_payload(
    text: str,
    *,
    allow_unclosed_fence: bool = False,
    balance_html_tables: bool = False,
) -> str:
    """Normalize API responses accidentally persisted as cache content."""
    return _extract_markdown(
        text,
        allow_unclosed_fence=allow_unclosed_fence,
        balance_html_tables=balance_html_tables,
    )


def _clean_markdown(
    text: str,
    *,
    allow_unclosed_fence: bool = False,
    balance_html_tables: bool = False,
) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline == -1:
            raise FinixDocError("FinixDoc-VL returned an incomplete fenced response.")
        if not stripped.endswith("```"):
            if not allow_unclosed_fence:
                raise FinixDocError(
                    "FinixDoc-VL response appears truncated because its Markdown "
                    "code fence is not closed. Use a smaller --slice-height."
                )
            stripped = stripped[first_newline + 1 :].strip()
        else:
            stripped = stripped[first_newline + 1 : -3].strip()
    if balance_html_tables:
        lowered = stripped.lower()
        missing_closes = lowered.count("<table") - lowered.count("</table>")
        if missing_closes == 0:
            stripped = repair_implicit_html_table_closures(stripped)
        if missing_closes > 0:
            stripped = stripped.rstrip() + "\n" + "\n".join("</table>" for _ in range(missing_closes))
    return stripped + "\n"
