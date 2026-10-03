from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Deterministic within-thread agent without persistent memory or compaction."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route to the deterministic offline implementation when no live agent exists."""
        if self.force_offline or self.langchain_agent is None:
            return self._reply_offline(thread_id, message)
        raise RuntimeError("Live Baseline Agent is not implemented")

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Answer from this thread's user messages and update cumulative usage."""
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt = "\n".join(f"{item['role']}: {item['content']}" for item in session.messages)
        prompt_tokens = estimate_tokens(prompt)
        answer = self._offline_response(session.messages, message)
        output_tokens = estimate_tokens(answer)
        session.messages.append({"role": "assistant", "content": answer})
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += output_tokens
        return {
            "response": answer,
            "token_usage": output_tokens,
            "prompt_tokens_processed": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Return no live agent; live provider wiring is an optional extension."""
        return None

    def _offline_response(self, messages: list[dict[str, str]], message: str) -> str:
        facts: dict[str, str] = {}
        for item in messages:
            if item["role"] == "user":
                clauses = re.split(r"(?<=[.!?])\s+|\n", item["content"])
                for clause in clauses:
                    if "?" in clause:
                        assertion = re.split(
                            r",\s*(?=(?:bạn|mình|tôi)\s+(?:có thể|nhớ|biết|giúp))",
                            clause,
                            maxsplit=1,
                            flags=re.IGNORECASE,
                        )[0]
                        if assertion == clause:
                            continue
                        clause = assertion
                    facts.update(extract_profile_updates(clause))

        question = "?" in message or any(
            phrase in message.casefold()
            for phrase in ("nhắc lại", "cho mình biết", "bạn nhớ", "mình muốn biết")
        )
        if not question:
            return "Mình đã ghi nhận thông tin trong thread này."

        query = message.casefold()
        requested = (
            ("name", ("tên",)),
            ("location", ("ở đâu", "nơi ở", "đang ở", "ở hiện tại")),
            ("profession", ("nghề", "công việc", "làm gì")),
            ("favorite_drink", ("đồ uống", "uống gì")),
            ("favorite_food", ("món ăn", "ăn gì")),
            ("response_style", ("style", "kiểu trả lời", "phong cách")),
            ("interests", ("quan tâm", "sở thích", "thích gì")),
        )
        for key, phrases in requested:
            if any(phrase in query for phrase in phrases):
                if key in facts:
                    return f"Trong thread này, bạn đã cho biết {facts[key] if key == 'name' else facts[key]}."
                return "Mình chưa biết thông tin đó trong thread này."
        if facts:
            return "Trong thread này, bạn đã chia sẻ: " + "; ".join(f"{key}: {value}" for key, value in facts.items()) + "."
        return "Mình chưa biết thông tin đó trong thread này."
