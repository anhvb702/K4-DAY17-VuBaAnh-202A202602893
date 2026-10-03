from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

import benchmark
from benchmark import heuristic_quality, load_conversations, recall_points, run_agent_benchmark
from config import load_config
from memory_store import UserProfileStore


def test_scores_use_literal_facts_with_unicode_and_empty_expectations() -> None:
    expected = ["Huế", "MLOps engineer", "AI"]
    assert recall_points("Mình chưa biết.", expected) == 0
    assert recall_points("Hiện ở Huế.", expected) == 0.5
    answer = unicodedata.normalize("NFD", "Ở Huế, làm MLOps engineer và quan tâm AI.")
    assert recall_points(answer.upper().replace(" ", "  "), expected) == 1
    assert heuristic_quality("Ở Huế và quan tâm AI.", expected) == pytest.approx(2 / 3)
    assert recall_points("Mai", ["AI"]) == heuristic_quality("Mai", ["AI"]) == 0
    assert recall_points("anything", []) == heuristic_quality("anything", []) == 0


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"id": "c", "user_id": "u", "recall_questions": []}, "turns"),
        ({"id": "c", "user_id": "u", "turns": ["hello"], "recall_questions": "bad"}, "recall_questions"),
        (
            {"id": "c", "user_id": "u", "turns": ["hello"], "recall_questions": [{"question": "Q?"}]},
            "expected_contains",
        ),
    ],
)
def test_loader_rejects_missing_or_mistyped_fields(tmp_path: Path, payload: dict, error: str) -> None:
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps([payload], ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_conversations(path)


def test_loader_preserves_valid_conversation(tmp_path: Path) -> None:
    payload = [{
        "id": "c", "user_id": "u", "turns": ["Mình tên Mai."],
        "recall_questions": [{"question": "Mình tên gì?", "expected_contains": ["Mai"]}],
    }]
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert load_conversations(path) == payload


class RecordingAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.profile_sizes: dict[str, int] = {}
        self.size_reads: list[str] = []
        self.compaction_reads: list[str] = []

    def reply(self, user_id: str, thread_id: str, message: str) -> dict:
        self.calls.append((user_id, thread_id, message))
        if message == "Mình tên Mai.":
            self.profile_sizes[user_id] = 12
        elif message == "Mình ở Huế.":
            self.profile_sizes[user_id] = 24
        return {"response": "Mai, Huế", "token_usage": 2, "prompt_tokens_processed": 3}

    def token_usage(self, thread_id: str) -> int:
        return 9999  # The runner must use the per-turn mapping instead.

    def prompt_token_usage(self, thread_id: str) -> int:
        return 9999

    def memory_file_size(self, user_id: str) -> int:
        self.size_reads.append(user_id)
        return self.profile_sizes.get(user_id, 0)

    def compaction_count(self, thread_id: str) -> int:
        self.compaction_reads.append(thread_id)
        return int(thread_id.endswith(":training"))


def test_runner_uses_fresh_recall_threads_and_per_turn_totals(tmp_path: Path) -> None:
    conversations = [
        {
            "id": "first", "user_id": "u", "turns": ["Mình tên Mai."],
            "recall_questions": [{"question": "Mình tên gì?", "expected_contains": ["Mai"]}],
        },
        {
            "id": "second", "user_id": "u", "turns": ["Mình ở Huế."],
            "recall_questions": [
                {"question": "Mình ở đâu?", "expected_contains": ["Huế"]},
                {"question": "Một câu chưa có đáp án chấm?", "expected_contains": []},
            ],
        },
    ]
    first, second = RecordingAgent(), RecordingAgent()
    config = load_config(tmp_path)
    row_a = run_agent_benchmark("A", first, conversations, config)
    row_b = run_agent_benchmark("B", second, conversations, config)

    assert first.calls == second.calls
    assert [call[2] for call in first.calls] == [
        "Mình tên Mai.", "Mình tên gì?", "Mình ở Huế.", "Mình ở đâu?",
        "Một câu chưa có đáp án chấm?",
    ]
    assert first.calls[0][1] != first.calls[1][1]
    assert first.calls[2][1] != first.calls[3][1]
    assert first.calls[0][1] != first.calls[2][1]
    assert first.calls[3][1] == first.calls[4][1]
    assert first.size_reads == second.size_reads == ["u", "u"]
    assert len(set(first.compaction_reads)) == len(first.compaction_reads) == 4
    assert row_a.agent_tokens_only == row_b.agent_tokens_only == 10
    assert row_a.prompt_tokens_processed == row_b.prompt_tokens_processed == 15
    assert row_a.memory_growth_bytes == row_b.memory_growth_bytes == 24
    assert row_a.compactions == row_b.compactions == 2
    assert row_a.recall_score == row_a.response_quality == 1


def test_main_uses_fresh_state_and_prints_both_tables(tmp_path: Path, monkeypatch, capsys) -> None:
    config = load_config(tmp_path)
    config.data_dir.mkdir(parents=True)
    for filename, name in (("conversations.json", "Mai"), ("advanced_long_context.json", "Lan")):
        payload = [{
            "id": "one", "user_id": "u", "turns": [f"Mình tên {name}."],
            "recall_questions": [{"question": "Mình tên gì?", "expected_contains": [name]}],
        }]
        (config.data_dir / filename).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    UserProfileStore(config.state_dir / "profiles").upsert_fact("u", "name", "Old")
    monkeypatch.setattr(benchmark, "load_config", lambda _: config)

    benchmark.main()
    first = capsys.readouterr().out
    benchmark.main()
    second = capsys.readouterr().out
    first_dir = Path(first.splitlines()[0].removeprefix("Run state: "))
    second_dir = Path(second.splitlines()[0].removeprefix("Run state: "))
    assert first_dir != second_dir
    for run_dir in (first_dir, second_dir):
        standard = run_dir / "standard" / "advanced" / "profiles" / "u" / "User.md"
        stress = run_dir / "stress" / "advanced" / "profiles" / "u" / "User.md"
        assert "- name: Mai" in standard.read_text(encoding="utf-8")
        assert "- name: Lan" in stress.read_text(encoding="utf-8")
        assert "Old" not in standard.read_text(encoding="utf-8")

    def numeric_output(text: str) -> list[str]:
        return [line for line in text.splitlines() if line.startswith("| ")]

    assert numeric_output(first) == numeric_output(second)
    assert first.count("| Agent | Agent tokens only | Prompt tokens processed |") == 2
    assert first.count("| Baseline |") == first.count("| Advanced |") == 2
    assert "Standard Benchmark (data/conversations.json)" in first
    assert "Long-Context Stress Benchmark (data/advanced_long_context.json)" in first
    assert "| Baseline |" in first and "| 0 | 0 |" in first
