from __future__ import annotations

from pathlib import Path

from agent_baseline import BaselineAgent
from config import load_config
from memory_store import estimate_tokens


def make_agent(tmp_path: Path) -> BaselineAgent:
    return BaselineAgent(config=load_config(tmp_path), force_offline=True)


def test_short_term_recall_and_thread_isolation(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    first = agent.reply("same-user", "thread-a", "Mình tên Mai. Bạn có thể nhớ giúp không?")
    recall = agent.reply("same-user", "thread-a", "Mình tên gì?")
    fresh_thread = agent.reply("same-user", "thread-b", "Mình tên gì?")

    assert "Mai" in recall["response"]
    assert "Mai" not in fresh_thread["response"]
    assert len(agent.sessions["thread-a"].messages) == 4
    assert len(agent.sessions["thread-b"].messages) == 2
    assert first["response"]


def test_threads_are_isolated_even_when_both_users_share_identity(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    agent.reply("same-user", "thread-a", "Mình tên Mai.")
    agent.reply("same-user", "thread-b", "Mình tên Lan.")

    assert "Mai" in agent.reply("same-user", "thread-a", "Mình tên gì?")["response"]
    assert "Lan" in agent.reply("same-user", "thread-b", "Mình tên gì?")["response"]


def test_new_agent_instance_does_not_reuse_same_state_dir_session(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    first = BaselineAgent(config=config, force_offline=True)
    second = BaselineAgent(config=config, force_offline=True)
    first.reply("same-user", "same-thread", "Mình tên Mai.")

    answer = second.reply("same-user", "same-thread", "Mình tên gì?")["response"]
    assert "Mai" not in answer
    assert "chưa biết" in answer.lower()


def test_user_facts_correct_and_ignore_questions_and_meeting_noise(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    agent.reply("u", "t", "Mình tên Mai. Mình đang ở Huế.")
    agent.reply("u", "t", "Mình vừa đi họp ở Hà Nội hai ngày.")
    agent.reply("u", "t", "Mình chuyển nơi ở từ Huế sang Đà Lạt.")

    answer = agent.reply("u", "t", "Hiện tại mình đang ở đâu?")["response"]
    assert "Đà Lạt" in answer
    assert "Huế" not in answer
    assert "Hà Nội" not in answer


def test_profession_correction_survives_joke_and_denial(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    agent.reply("u", "t", "Mình đang làm kế toán.")
    agent.reply("u", "t", "Mình không còn làm kế toán nữa, giờ chuyển sang kỹ sư dữ liệu.")
    agent.reply("u", "t", "Mình đùa hay chuyển sang đầu bếp, chứ không phải nghề hiện tại.")

    answer = agent.reply("u", "t", "Hiện tại mình làm nghề gì?")["response"]
    assert "kỹ sư dữ liệu" in answer
    assert "kế toán" not in answer
    assert "đầu bếp" not in answer


def test_unknown_facts_do_not_leak_from_other_threads_or_assistant_messages(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    agent.reply("u", "source", "Mình tên Mai.")
    target = agent.reply("u", "target", "Bạn mình tên là Nam.")
    agent.sessions["target"].messages.append(
        {"role": "assistant", "content": "Mình tên Khoa và hiện ở Đà Lạt."}
    )
    unknown = agent.reply("u", "target", "Mình tên gì?")

    assert "Mai" not in target["response"]
    assert "Nam" not in unknown["response"]
    assert "Khoa" not in unknown["response"]
    assert "chưa biết" in unknown["response"].lower()


def test_suggested_name_and_location_in_question_do_not_become_facts(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    name_answer = agent.reply("u", "name-question", "Mình tên Mai phải không?")["response"]
    place_answer = agent.reply("u", "place-question", "Mình đang ở Hà Nội phải không?")["response"]

    assert "Mai" not in name_answer
    assert "Hà Nội" not in place_answer
    assert "chưa biết" in name_answer.lower()
    assert "chưa biết" in place_answer.lower()


def test_question_about_missing_fact_does_not_become_a_fact(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    answer = agent.reply("u", "empty", "Mình làm nghề gì?")["response"]
    assert "chưa biết" in answer.lower()


def test_offline_outputs_and_counters_are_deterministic(tmp_path: Path) -> None:
    first = make_agent(tmp_path / "one")
    second = make_agent(tmp_path / "two")
    turns = ["Mình tên Mai.", "Mình tên gì?", "Mình thích Python."]

    first_results = [first.reply("u", "t", turn) for turn in turns]
    second_results = [second.reply("u", "t", turn) for turn in turns]

    assert first_results == second_results
    assert first.token_usage("t") == second.token_usage("t")
    assert first.prompt_token_usage("t") == second.prompt_token_usage("t")


def test_token_accounting_is_per_turn_then_cumulative(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    first = agent.reply("u", "t", "Mình tên Mai.")
    second = agent.reply("u", "t", "Mình tên gì?")
    messages = agent.sessions["t"].messages

    first_prompt = estimate_tokens("user: Mình tên Mai.")
    second_prompt = estimate_tokens("\n".join(f"{item['role']}: {item['content']}" for item in messages[:3]))
    assert first["prompt_tokens_processed"] == first_prompt
    assert second["prompt_tokens_processed"] == second_prompt
    assert agent.prompt_token_usage("t") == first_prompt + second_prompt
    assert first["token_usage"] == estimate_tokens(messages[1]["content"])
    assert second["token_usage"] == estimate_tokens(messages[3]["content"])
    assert agent.token_usage("t") == first["token_usage"] + second["token_usage"]
    assert agent.token_usage("missing") == agent.prompt_token_usage("missing") == 0


def test_baseline_keeps_full_history_without_profiles_or_compaction(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    for index in range(25):
        agent.reply("u", "long-thread", f"Lượt {index}: " + "trao đổi dài " * 12)

    messages = agent.sessions["long-thread"].messages
    assert len(messages) == 50
    assert [message["role"] for message in messages] == ["user", "assistant"] * 25
    assert agent.compaction_count("long-thread") == 0
    assert agent.langchain_agent is None
    assert not (tmp_path / "state" / "profiles").exists()
    assert agent.sessions["long-thread"].token_usage == agent.token_usage("long-thread")


def test_same_user_can_have_distinct_sessions_and_zero_compactions(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    agent.reply("u", "a", "hello")
    agent.reply("u", "b", "hello")
    assert agent.compaction_count("a") == agent.compaction_count("b") == 0
    assert agent.token_usage("a") == agent.token_usage("b")
    assert agent.prompt_token_usage("missing") == 0
