from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Offline agent with a per-user profile and compact per-thread context."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.force_offline or self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)
        raise RuntimeError("Live Advanced Agent is not implemented")

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        updates = self._asserted_updates(message)
        for key, value in updates.items():
            if key == "response_style":
                continue
            self.profile_store.upsert_fact(user_id, key, value)
        self._update_response_style(user_id, updates.get("response_style"), message)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        answer = self._offline_response(user_id, thread_id, message)
        output_tokens = estimate_tokens(answer)
        self.compact_memory.append(thread_id, "assistant", answer)

        self.thread_tokens[thread_id] = self.token_usage(thread_id) + output_tokens
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        return {
            "response": answer,
            "token_usage": output_tokens,
            "prompt_tokens_processed": prompt_tokens,
        }

    @staticmethod
    def _asserted_updates(message: str) -> dict[str, str]:
        """Ignore suggested facts in questions while retaining preceding assertions."""
        updates: dict[str, str] = {}
        for clause in re.split(r"(?<=[.!?])\s+|\n", message):
            recall = re.search(
                r"\b(?:nhắc lại|cho mình biết|bạn nhớ)\b|^\s*tóm tắt\b",
                clause,
                re.IGNORECASE,
            )
            if recall:
                clause = clause[:recall.start()].rstrip(" ,;:")
                if not clause:
                    continue
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
            clause_updates = extract_profile_updates(clause)
            if "response_style" in clause_updates and "response_style" in updates:
                clause_updates["response_style"] = AdvancedAgent._combine_style(
                    updates["response_style"], clause_updates["response_style"]
                )
            updates.update(clause_updates)
            pet = re.search(
                r"\b(?:mình|tôi)\s+nuôi\s+(?:một\s+)?(?:bé|con)\s+([^\s,.;!?]+)",
                clause,
                re.IGNORECASE,
            )
            if pet and pet.group(1).casefold() not in {"gì", "nào", "ai"}:
                updates["pet"] = pet.group(1)
        return updates

    @staticmethod
    def _combine_style(previous: str, current: str) -> str:
        traits = [trait.strip() for trait in previous.split(",") if trait.strip()]
        for trait in (part.strip() for part in current.split(",")):
            if trait == "bullet" and any(re.fullmatch(r"\d+ bullet", old) for old in traits):
                continue
            if "bullet" in trait:
                traits = [old for old in traits if "bullet" not in old and old != "dạng đoạn văn"]
            if trait and trait not in traits:
                traits.append(trait)
        return ", ".join(traits)

    def _update_response_style(self, user_id: str, update: str | None, message: str) -> None:
        correction = message if "?" not in message else ""
        prose = bool(re.search(r"\bmột đoạn văn\b", correction, re.IGNORECASE))
        stop_bullets = prose or bool(
            re.search(r"\bkhông\s+(?:dùng|sử dụng)\s+bullet\b", correction, re.IGNORECASE)
        )
        stop_tradeoff = bool(
            re.search(
                r"\bkhông\s+cần\s+(?:ưu tiên|nhấn mạnh|so sánh)\s+trade-off\b",
                correction,
                re.IGNORECASE,
            )
        )
        if not update and not stop_bullets and not stop_tradeoff:
            return

        facts = self.profile_store.facts(user_id)
        previous = facts.get("response_style", "")
        style = self._combine_style(previous, update or "")
        traits = [trait.strip() for trait in style.split(",") if trait.strip()]
        if stop_bullets:
            traits = [trait for trait in traits if "bullet" not in trait]
        if prose and "dạng đoạn văn" not in traits:
            traits.append("dạng đoạn văn")
        if stop_tradeoff:
            traits = [trait for trait in traits if "trade-off" not in trait]
        value = ", ".join(traits)
        if value and value != previous:
            self.profile_store.upsert_fact(user_id, "response_style", value)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Count the profile, summary, and role-labelled recent messages once each.

        The offline path uses no system prompt, so it has no system overhead.
        Call this after appending the user and before appending the assistant.
        """
        context = self.compact_memory.context(thread_id)
        recent = "\n".join(
            f"{item['role']}: {item['content']}" for item in context["messages"]
        )
        return (
            estimate_tokens(self.profile_store.read_text(user_id))
            + estimate_tokens(context["summary"])
            + estimate_tokens(recent)
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.facts(user_id)
        query = message.casefold()
        if "?" not in message and not any(
            phrase in query for phrase in ("nhắc lại", "cho mình biết", "bạn nhớ", "tóm tắt", "mình muốn biết")
        ):
            return "Mình đã ghi nhận thông tin bạn chia sẻ."

        requested = (
            ("name", ("tên", "mình là ai"), "Tên bạn là"),
            ("location", ("ở đâu", "nơi ở", "hiện đang ở", "còn ở"), "Bạn hiện ở"),
            ("profession", ("nghề", "công việc", "làm gì"), "Nghề hiện tại của bạn là"),
            ("interests", ("mối quan tâm", "quan tâm", "sở thích", "thích gì"), "Bạn quan tâm"),
            ("response_style", ("style", "kiểu trả lời", "phong cách"), "Bạn thích cách trả lời"),
            ("favorite_drink", ("đồ uống", "uống gì"), "Đồ uống yêu thích của bạn là"),
            ("favorite_food", ("món ăn", "ăn gì"), "Món ăn yêu thích của bạn là"),
            ("pet", ("nuôi con gì", "nuôi con", "nuôi gì"), "Bạn nuôi"),
        )
        keys = [entry for entry in requested if any(phrase in query for phrase in entry[1])]
        if not keys and "tóm tắt" in query and "mình" in query:
            keys = [entry for entry in requested if entry[0] in {"name", "profession", "interests"}]

        parts = [f"{label} {facts[key]}." for key, _, label in keys if key in facts]
        if parts:
            style = facts.get("response_style", "")
            if "bullet" in style and len(parts) > 1:
                limit = re.search(r"(\d+) bullet", style)
                count = max(1, int(limit.group(1))) if limit else len(parts)
                lines = parts[:count]
                if len(parts) > count:
                    lines[-1] = " ".join(parts[count - 1:])
                return "\n".join(f"- {part}" for part in lines)
            return " ".join(parts)

        if any(phrase in query for phrase in ("chủ đề", "đang bàn", "nói về", "thảo luận", "yêu cầu", "đang muốn")):
            context = self.compact_memory.context(thread_id)
            recent = context["messages"]
            earlier = recent[:-1] if recent and recent[-1] == {"role": "user", "content": message} else recent
            for item in reversed(earlier):
                content = item["content"]
                if item["role"] == "user" and "?" not in content and re.search(
                    r"\b(?:chủ đề|bàn về|thảo luận|tìm hiểu|so sánh|đọc về|muốn)\b", content, re.IGNORECASE
                ):
                    return f"Chủ đề trong thread này: {content}"
            topics = re.findall(r"^Topic: (.+)$", context["summary"], re.MULTILINE)
            if topics:
                return f"Chủ đề trong thread này: {topics[-1]}"

        return "Mình chưa biết thông tin đó."

    def _maybe_build_langchain_agent(self):
        """Live provider wiring is outside the offline milestone."""
        return None
