from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider


DEFAULT_COMPACT_THRESHOLD_TOKENS = 512
DEFAULT_COMPACT_KEEP_MESSAGES = 6
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load the lab configuration without constructing a model or using a network.

    The repository root is inferred from this file when ``base_dir`` is omitted;
    callers may pass an explicit root for isolated tests.  ``.env`` is loaded
    from that root with ``override=False``, so real environment variables win.

    Supported environment variables (the ``JUDGE_`` variants configure the
    separate judge model):

    - ``LLM_PROVIDER`` / ``JUDGE_PROVIDER``: ``openai`` by default, plus
      ``custom``, ``gemini``, ``anthropic``, ``ollama`` and ``openrouter``.
    - ``LLM_MODEL`` / ``JUDGE_MODEL``: ``gpt-4o-mini`` by default.
    - ``LLM_TEMPERATURE`` / ``JUDGE_TEMPERATURE``: ``0.0`` by default.
    - ``LLM_API_KEY`` / ``JUDGE_API_KEY`` and
      ``LLM_BASE_URL`` / ``JUDGE_BASE_URL``: generic overrides.
    - Provider-specific fallbacks: ``OPENAI_API_KEY``, ``GEMINI_API_KEY``,
      ``ANTHROPIC_API_KEY``, ``OPENROUTER_API_KEY``, ``CUSTOM_API_KEY``;
      ``OPENAI_BASE_URL``, ``ANTHROPIC_BASE_URL``, ``OPENROUTER_BASE_URL``,
      ``CUSTOM_BASE_URL`` and ``OLLAMA_BASE_URL``.
    - ``COMPACT_THRESHOLD_TOKENS`` defaults to 512 and
      ``COMPACT_KEEP_MESSAGES`` defaults to 6.

    The compact default is deliberately moderate: it leaves ordinary short
    exchanges intact while allowing the longer stress conversation to trigger
    compaction under the lab's deterministic token heuristic.  Tests can lower
    it further without changing this shared production-like default.
    """

    root = (Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent.parent).resolve()
    load_dotenv(root / ".env", override=False)

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    compact_threshold_tokens = _read_int(
        "COMPACT_THRESHOLD_TOKENS", DEFAULT_COMPACT_THRESHOLD_TOKENS, minimum=1
    )
    compact_keep_messages = _read_int(
        "COMPACT_KEEP_MESSAGES", DEFAULT_COMPACT_KEEP_MESSAGES, minimum=1
    )

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    model = _build_provider_config("LLM", provider, default_model=DEFAULT_MODEL)

    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", provider))
    judge_model = _build_provider_config(
        "JUDGE", judge_provider, default_model=os.getenv("LLM_MODEL", DEFAULT_MODEL)
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model,
        judge_model=judge_model,
    )


def _read_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _read_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}, got {raw!r}")
    return value


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _build_provider_config(prefix: str, provider: str, *, default_model: str) -> ProviderConfig:
    model_name = _first_env(f"{prefix}_MODEL") or default_model
    temperature = _read_float(f"{prefix}_TEMPERATURE", DEFAULT_TEMPERATURE)

    provider_env = {
        "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
        "custom": ("CUSTOM_API_KEY", "CUSTOM_BASE_URL"),
        "gemini": ("GEMINI_API_KEY", None),
        "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"),
        "ollama": (None, "OLLAMA_BASE_URL"),
        "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"),
    }
    key_name, base_url_name = provider_env[provider]
    api_key = _first_env(f"{prefix}_API_KEY", key_name) if key_name else _first_env(f"{prefix}_API_KEY")
    base_url = _first_env(f"{prefix}_BASE_URL", base_url_name) if base_url_name else _first_env(f"{prefix}_BASE_URL")

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )
