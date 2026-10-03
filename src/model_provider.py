from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_PROVIDERS = (
    "openai",
    "custom",
    "gemini",
    "anthropic",
    "ollama",
    "openrouter",
)

_PROVIDER_ALIASES = {
    "anthorpic": "anthropic",
    "claude": "anthropic",
    "open_ai": "openai",
    "open-ai": "openai",
    "google": "gemini",
    "google_genai": "gemini",
    "google-gemini": "gemini",
    "open_router": "openrouter",
    "open-router": "openrouter",
    "local": "ollama",
}


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Return a canonical provider name or fail with an actionable error."""

    if not isinstance(value, str):
        raise ValueError(
            f"Provider must be a string; supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    normalized = "_".join(value.strip().lower().replace("-", "_").split())
    normalized = _PROVIDER_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_PROVIDERS:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(f"Unknown provider {value!r}; supported providers: {supported}")
    return normalized


def build_chat_model(config: ProviderConfig):
    """Build one provider-specific LangChain chat model on demand.

    Imports stay inside each branch so offline config loading does not require
    every optional integration package.  This function constructs a client but
    does not invoke it; any network request happens only when the caller uses
    the returned model.
    """

    provider = normalize_provider(config.provider)
    common = {"temperature": config.temperature}

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Provider 'openai' requires the 'langchain-openai' package. "
                "Install it before enabling live mode."
            ) from exc
        return ChatOpenAI(**_with_optional(
            {"model": config.model_name, **common},
            api_key=config.api_key,
            base_url=config.base_url,
        ))

    if provider == "custom":
        if not config.base_url:
            raise ValueError(
                "Provider 'custom' requires base_url; set CUSTOM_BASE_URL or pass ProviderConfig.base_url."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Provider 'custom' requires the 'langchain-openai' package. "
                "Install it before enabling live mode."
            ) from exc
        return ChatOpenAI(**_with_optional(
            {"model": config.model_name, **common},
            api_key=config.api_key,
            base_url=config.base_url,
        ))

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError(
                "Provider 'gemini' requires the 'langchain-google-genai' package. "
                "Install it before enabling live mode."
            ) from exc
        return ChatGoogleGenerativeAI(**_with_optional(
            {"model": config.model_name, **common}, api_key=config.api_key
        ))

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "Provider 'anthropic' requires the 'langchain-anthropic' package. "
                "Install it before enabling live mode."
            ) from exc
        return ChatAnthropic(**_with_optional(
            {"model_name": config.model_name, **common},
            api_key=config.api_key,
            base_url=config.base_url,
        ))

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "Provider 'ollama' requires the 'langchain-ollama' package. "
                "Install it before enabling live mode."
            ) from exc
        return ChatOllama(**_with_optional(
            {"model": config.model_name, **common}, base_url=config.base_url
        ))

    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter
        except ImportError as exc:
            raise RuntimeError(
                "Provider 'openrouter' requires the 'langchain-openrouter' package. "
                "Install it before enabling live mode."
            ) from exc
        return ChatOpenRouter(**_with_optional(
            {"model": config.model_name, **common},
            api_key=config.api_key,
            base_url=config.base_url,
        ))

    # normalize_provider currently makes this unreachable; retain a defensive
    # error if a future provider is added without a corresponding constructor.
    raise ValueError(f"Provider {provider!r} has no chat-model implementation")


def _with_optional(values: dict, **optional: str | None) -> dict:
    """Add only configured optional SDK arguments."""

    return {**values, **{key: value for key, value in optional.items() if value}}
