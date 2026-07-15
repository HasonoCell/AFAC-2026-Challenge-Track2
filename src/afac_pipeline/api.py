"""FinixDoc-VL API client and response normalization."""

from __future__ import annotations

import json
import mimetypes
import secrets
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
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

    def call_with_file(self, file_name: str, file_bytes: bytes) -> str:
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

        markdown = _extract_markdown(response_text)
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


def _extract_markdown(response_text: str) -> str:
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
    return _clean_markdown(extracted)


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


def normalize_markdown_payload(text: str) -> str:
    """Normalize API responses accidentally persisted as cache content."""
    return _extract_markdown(text)


def _clean_markdown(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline == -1:
            raise FinixDocError("FinixDoc-VL returned an incomplete fenced response.")
        if not stripped.endswith("```"):
            raise FinixDocError(
                "FinixDoc-VL response appears truncated because its Markdown "
                "code fence is not closed. Use a smaller --slice-height."
            )
        stripped = stripped[first_newline + 1 : -3].strip()
    return stripped + "\n"
