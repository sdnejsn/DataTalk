"""Environment configuration, provider resolution and input limits."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values

from .providers import SUPPORTED_PROVIDERS, get_provider_preset


@dataclass(frozen=True)
class Settings:
    provider: str = "deepseek"
    api_key: str = field(default="", repr=False)
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    api_timeout_seconds: float = 90
    max_tokens: int = 4096
    max_retries: int = 2
    execution_timeout_seconds: float = 30
    output_dir: Path = Path("outputs")
    sample_rows: int = 0
    max_csv_mb: int = 20
    max_rows: int = 100_000

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            choices = ", ".join(SUPPORTED_PROVIDERS)
            raise ValueError(f"Unsupported provider {provider!r}; choose one of: {choices}")
        object.__setattr__(self, "provider", provider)
        if not 0 <= self.max_retries <= 2:
            raise ValueError("max_retries must be between 0 and 2")
        if not 0 <= self.sample_rows <= 10:
            raise ValueError("sample_rows must be between 0 and 10")
        for name in (
            "api_timeout_seconds",
            "execution_timeout_seconds",
            "max_tokens",
            "max_csv_mb",
            "max_rows",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        parsed_url = urlparse(self.base_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError("base_url must be an absolute HTTPS URL")
        if not self.model.strip():
            raise ValueError("model cannot be blank")
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    @property
    def provider_name(self) -> str:
        preset = get_provider_preset(self.provider)
        return preset.name if preset else "Custom OpenAI-compatible endpoint"

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> Settings:
        # Process variables take precedence; configuration does not mutate os.environ.
        values = {**dotenv_values(env_file), **os.environ}

        def get(name: str, default: str = "") -> str:
            value = values.get(name)
            return default if value is None else str(value).strip()

        provider = get("API_PROVIDER", "deepseek").lower()
        preset = get_provider_preset(provider)
        if provider != "custom" and preset is None:
            choices = ", ".join(SUPPORTED_PROVIDERS)
            raise ValueError(f"Unsupported API_PROVIDER {provider!r}; choose one of: {choices}")

        # DEEPSEEK_* is a migration fallback only when API_PROVIDER is deepseek.
        legacy = provider == "deepseek"
        api_key = get("API_KEY") or (get("DEEPSEEK_API_KEY") if legacy else "")
        default_url = preset.base_url if preset else ""
        default_model = preset.default_model if preset else ""
        base_url = (
            get("API_BASE_URL") or (get("DEEPSEEK_BASE_URL") if legacy else "") or default_url
        )
        model = get("API_MODEL") or (get("DEEPSEEK_MODEL") if legacy else "") or default_model
        timeout = (
            get("API_TIMEOUT_SECONDS")
            or (get("DEEPSEEK_TIMEOUT_SECONDS") if legacy else "")
            or "90"
        )
        max_tokens = (
            get("API_MAX_TOKENS") or (get("DEEPSEEK_MAX_TOKENS") if legacy else "") or "4096"
        )
        if provider == "custom" and (not base_url or not model):
            raise ValueError("custom provider requires API_BASE_URL and API_MODEL")

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            api_timeout_seconds=float(timeout),
            max_tokens=int(max_tokens),
            max_retries=int(get("DATATALK_MAX_RETRIES", "2")),
            execution_timeout_seconds=float(get("DATATALK_EXECUTION_TIMEOUT_SECONDS", "30")),
            output_dir=Path(get("DATATALK_OUTPUT_DIR", "outputs")),
            sample_rows=int(get("DATATALK_SAMPLE_ROWS", "0")),
            max_csv_mb=int(get("DATATALK_MAX_CSV_MB", "20")),
            max_rows=int(get("DATATALK_MAX_ROWS", "100000")),
        )
