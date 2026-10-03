from __future__ import annotations

import json
import re
import tempfile
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read the unchanged UTF-8 dataset and validate its required fields."""
    try:
        conversations = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load conversations from {path}: {exc}") from exc
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"{path}: expected a nonempty list of conversations")

    seen_ids: set[str] = set()
    for index, conversation in enumerate(conversations):
        where = f"{path}: conversation {index}"
        if not isinstance(conversation, dict):
            raise ValueError(f"{where} must be an object")
        for key in ("id", "user_id"):
            if not isinstance(conversation.get(key), str) or not conversation[key].strip():
                raise ValueError(f"{where}.{key} must be a nonempty string")
        if conversation["id"] in seen_ids:
            raise ValueError(f"{where}.id is duplicated: {conversation['id']}")
        seen_ids.add(conversation["id"])

        turns = conversation.get("turns")
        if not isinstance(turns, list) or not turns or any(
            not isinstance(turn, str) or not turn.strip() for turn in turns
        ):
            raise ValueError(f"{where}.turns must be a nonempty list of nonempty strings")
        questions = conversation.get("recall_questions")
        if not isinstance(questions, list):
            raise ValueError(f"{where}.recall_questions must be a list")
        for question_index, item in enumerate(questions):
            question_where = f"{where}.recall_questions[{question_index}]"
            if not isinstance(item, dict):
                raise ValueError(f"{question_where} must be an object")
            if not isinstance(item.get("question"), str) or not item["question"].strip():
                raise ValueError(f"{question_where}.question must be a nonempty string")
            expected = item.get("expected_contains")
            if not isinstance(expected, list) or any(
                not isinstance(fact, str) or not fact.strip() for fact in expected
            ):
                raise ValueError(f"{question_where}.expected_contains must be a list of nonempty strings")
    return conversations


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _matched_count(answer: str, expected: list[str]) -> int:
    normalized_answer = _normalized(answer)
    return sum(
        bool(re.search(rf"(?<!\w){re.escape(_normalized(fact))}(?!\w)", normalized_answer))
        for fact in expected
    )


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0, 0.5, or 1 for no, partial, or complete literal fact coverage."""
    if not expected:
        return 0.0
    matched = _matched_count(answer, expected)
    return 1.0 if matched == len(expected) else 0.5 if matched else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Return literal expected-fact coverage on a 0–1 scale.

    This offline proxy is related to recall, not an independent assessment of
    fluency or factuality. Empty expected lists have no evidence and score 0.
    """
    return _matched_count(answer, expected) / len(expected) if expected else 0.0


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Score all training and recall replies with one agent instance.

    Token totals sum per-turn reply values. Recall and quality are arithmetic
    means over questions with nonempty expected facts; empty expected lists are
    still asked and counted for tokens, but excluded from score denominators.
    """
    user_ids = list(dict.fromkeys(item["user_id"] for item in conversations))
    profile_store = UserProfileStore(config.state_dir / "profiles")
    size_for = getattr(agent, "memory_file_size", profile_store.file_size)
    before_sizes = {user_id: size_for(user_id) for user_id in user_ids}

    output_tokens = 0
    prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    thread_ids: list[str] = []
    for index, conversation in enumerate(conversations):
        prefix = f"benchmark:{index}:{conversation['id']}"
        training_thread = f"{prefix}:training"
        recall_thread = f"{prefix}:recall"
        thread_ids.extend((training_thread, recall_thread))
        user_id = conversation["user_id"]

        for turn in conversation["turns"]:
            result = agent.reply(user_id, training_thread, turn)
            output_tokens += result["token_usage"]
            prompt_tokens += result["prompt_tokens_processed"]
        for question in conversation["recall_questions"]:
            result = agent.reply(user_id, recall_thread, question["question"])
            output_tokens += result["token_usage"]
            prompt_tokens += result["prompt_tokens_processed"]
            expected = question["expected_contains"]
            if expected:
                recall_scores.append(recall_points(result["response"], expected))
                quality_scores.append(heuristic_quality(result["response"], expected))

    growth = sum(size_for(user_id) - before_sizes[user_id] for user_id in user_ids)
    compactions = sum(agent.compaction_count(thread_id) for thread_id in dict.fromkeys(thread_ids))
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=output_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        memory_growth_bytes=growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format both benchmark suites with the same six metric columns."""
    lines = [
        "| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.agent_name} | {row.agent_tokens_only} | {row.prompt_tokens_processed} "
            f"| {row.recall_score:.1%} | {row.response_quality:.1%} "
            f"| {row.memory_growth_bytes} | {row.compactions} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Run both datasets offline in a fresh, inspectable state namespace."""
    config = load_config(Path(__file__).resolve().parent.parent)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="benchmark-", dir=config.state_dir))
    print(f"Run state: {run_dir}")
    print(
        "Offline deterministic; tokens are heuristic estimates from every training and recall reply. "
        "Quality is literal expected-fact coverage, not an independent judge."
    )

    suites = (
        ("Standard Benchmark", "conversations.json", "standard"),
        ("Long-Context Stress Benchmark", "advanced_long_context.json", "stress"),
    )
    for title, filename, namespace in suites:
        conversations = load_conversations(config.data_dir / filename)
        baseline_config = replace(config, state_dir=run_dir / namespace / "baseline")
        advanced_config = replace(config, state_dir=run_dir / namespace / "advanced")
        baseline = BaselineAgent(config=baseline_config, force_offline=True)
        advanced = AdvancedAgent(config=advanced_config, force_offline=True)
        rows = [
            run_agent_benchmark("Baseline", baseline, conversations, baseline_config),
            run_agent_benchmark("Advanced", advanced, conversations, advanced_config),
        ]
        print(f"\n{title} (data/{filename})")
        print(format_rows(rows))
        print(f"Advanced profiles: {advanced_config.state_dir / 'profiles'}")

    print(
        f"Compact settings: threshold={config.compact_threshold_tokens} estimated tokens, "
        f"keep_messages={config.compact_keep_messages}"
    )


if __name__ == "__main__":
    main()
