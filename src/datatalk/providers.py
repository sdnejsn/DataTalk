"""OpenAI-compatible provider presets exposed to configuration and frontends."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProviderPreset:
    id: str
    name: str
    base_url: str
    default_model: str
    note: str = ""


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    preset.id: preset
    for preset in (
        ProviderPreset(
            "deepseek",
            "DeepSeek",
            "https://api.deepseek.com",
            "deepseek-flash",
            "DataTalk disables DeepSeek thinking mode for tool calls.",
        ),
        ProviderPreset("openai", "OpenAI", "https://api.openai.com/v1", "gpt-4.1-mini"),
        ProviderPreset(
            "qwen",
            "Alibaba Cloud Model Studio (Qwen)",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "qwen-plus",
        ),
        ProviderPreset("moonshot", "Moonshot AI", "https://api.moonshot.cn/v1", "moonshot-v1-8k"),
        ProviderPreset("zhipu", "Zhipu AI", "https://open.bigmodel.cn/api/paas/v4", "glm-4.5"),
        ProviderPreset(
            "siliconflow", "SiliconFlow", "https://api.siliconflow.cn/v1", "Qwen/Qwen3-8B"
        ),
        ProviderPreset(
            "openrouter",
            "OpenRouter",
            "https://openrouter.ai/api/v1",
            "openai/gpt-4.1-mini",
        ),
        ProviderPreset(
            "gemini",
            "Google Gemini (OpenAI compatibility)",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "gemini-2.5-flash",
        ),
    )
}
SUPPORTED_PROVIDERS = (*PROVIDER_PRESETS, "custom")


def get_provider_preset(provider: str) -> ProviderPreset | None:
    return PROVIDER_PRESETS.get(provider.strip().lower())


def provider_catalog() -> list[dict[str, str]]:
    """Return non-secret provider metadata suitable for a frontend selector."""
    items = [asdict(preset) for preset in PROVIDER_PRESETS.values()]
    items.append(
        {
            "id": "custom",
            "name": "Custom OpenAI-compatible endpoint",
            "base_url": "",
            "default_model": "",
            "note": "API_BASE_URL and API_MODEL are required.",
        }
    )
    return items
