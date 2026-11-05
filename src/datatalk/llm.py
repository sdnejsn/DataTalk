"""OpenAI-compatible tool-calling adapter for supported API providers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .config import Settings
from .prompts import EXECUTE_TOOL


class ModelError(RuntimeError):
    """Provider or model response failure with a credential-free message."""


@dataclass
class Generation:
    code: str
    call_id: str
    assistant_message: dict[str, Any]
    prompt_tokens: int = 0
    completion_tokens: int = 0


class CodeGenerator(Protocol):
    def generate(self, messages: list[dict[str, Any]]) -> Generation: ...


class OpenAICompatibleGenerator:
    def __init__(self, settings: Settings) -> None:
        if not settings.api_key or settings.api_key.lower() in {
            "your_api_key",
            "your_key_here",
            "在此填写你的key",
        }:
            raise ValueError("Set API_KEY in .env before requesting an analysis")
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.api_timeout_seconds,
            max_retries=0,
        )

    def close(self) -> None:
        self.client.close()

    def generate(self, messages: list[dict[str, Any]]) -> Generation:
        request: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "tools": [EXECUTE_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "execute_python"}},
            "max_tokens": self.settings.max_tokens,
            "stream": False,
        }
        # Other OpenAI-compatible APIs may reject DeepSeek-specific request fields.
        if self.settings.provider == "deepseek":
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        try:
            response = self.client.chat.completions.create(**request)
        except APITimeoutError:
            raise ModelError(f"{self.settings.provider_name} request timed out") from None
        except APIConnectionError:
            raise ModelError(
                f"Could not connect to {self.settings.provider_name}; "
                "check API_BASE_URL and network"
            ) from None
        except APIStatusError as exc:
            hints = {
                400: "check API_MODEL and request compatibility",
                401: "check API_KEY",
                402: "check account balance",
                403: "check API permissions and region availability",
                404: "check API_BASE_URL and API_MODEL",
                429: "rate limited; retry later",
            }
            hint = hints.get(exc.status_code, "provider request failed; retry later")
            raise ModelError(
                f"{self.settings.provider_name} HTTP {exc.status_code}: {hint}"
            ) from None
        if not response.choices:
            raise ModelError(f"{self.settings.provider_name} returned no choices")
        choice = response.choices[0]
        calls = choice.message.tool_calls or []
        if not calls:
            raise ModelError(
                "Expected an execute_python tool call; "
                f"finish_reason={choice.finish_reason}. "
                "Check whether the selected model supports tools"
            )
        if len(calls) != 1 or calls[0].function.name != "execute_python":
            raise ModelError("Provider must return exactly one execute_python tool call")
        call = calls[0]
        try:
            arguments = json.loads(call.function.arguments)
        except (TypeError, json.JSONDecodeError):
            raise ModelError("Provider returned invalid tool arguments") from None
        if not isinstance(arguments, dict) or set(arguments) != {"code"}:
            raise ModelError("Tool arguments must contain only code")
        code = arguments["code"]
        if not isinstance(code, str) or not code.strip() or len(code) > 30_000:
            raise ModelError("Generated code must be nonempty and at most 30000 characters")
        usage = response.usage
        return Generation(
            code=code,
            call_id=call.id,
            assistant_message={
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": "execute_python",
                            "arguments": call.function.arguments,
                        },
                    }
                ],
            },
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


# Backward-compatible import for code written against DataTalk 0.1.0.
DeepSeekGenerator = OpenAICompatibleGenerator
