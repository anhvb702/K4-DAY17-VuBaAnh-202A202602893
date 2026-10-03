from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import estimate_tokens


def make_agent(tmp_path: Path, *, threshold: int = 512, keep: int = 4) -> AdvancedAgent:
    config = replace(
        load_config(tmp_path),
        compact_threshold_tokens=threshold,
        compact_keep_messages=keep,
    )
    return AdvancedAgent(config=config, force_offline=True)


def test_profile_persists_across_threads_and_instances_but_baseline_forgets(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    baseline = BaselineAgent(config=agent.config, force_offline=True)
    statement = "Mình tên Mai. Mình đang ở Huế."
    agent.reply("user-a", "thread-a", statement)
    baseline.reply("user-a", "thread-a", statement)

    profile_path = agent.profile_store.path_for("user-a")
    assert profile_path.is_file()
    assert "- name: Mai" in profile_path.read_text(encoding="utf-8")
    assert "Mai" in agent.reply("user-a", "thread-a", "Mình tên gì?")["response"]
    assert "Mai" in agent.reply("user-a", "thread-b", "Mình tên gì?")["response"]
    assert "Mai" in make_agent(tmp_path).reply("user-a", "thread-c", "Mình tên gì?")["response"]
    assert "chưa biết" in baseline.reply("user-a", "thread-b", "Mình tên gì?")["response"]
    assert agent.memory_file_size("user-a") == profile_path.stat().st_size


def test_users_are_isolated_and_questions_do_not_create_facts(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    agent.reply("user-a", "a", "Mình tên Mai.")
    assert "chưa biết" in agent.reply("user-b", "b", "Mình tên gì?")["response"]
    assert agent.memory_file_size("user-b") == 0
    agent.compact_memory.append("b", "assistant", "Mình tên Khoa và đang ở Đà Lạt.")
    assert "chưa biết" in agent.reply("user-b", "b", "Mình tên gì?")["response"]
    assert agent.memory_file_size("user-b") == 0

    before = agent.profile_store.read_text("user-a")
    before_size = agent.memory_file_size("user-a")
    agent.reply("user-a", "a", "Mình tên Lan phải không?")
    agent.reply("user-a", "a", "Mình đang ở Hà Nội phải không?")
    agent.reply("user-a", "a", "Mình tên gì?")
    assert agent.profile_store.read_text("user-a") == before
    assert agent.memory_file_size("user-a") == before_size
    assert "Mai" in agent.reply("user-a", "a", "Nhắc lại tên mình.")["response"]


def test_correction_replaces_old_fact_without_meeting_or_joke_noise(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    for turn in (
        "Mình đang ở Huế. Mình đang làm kế toán.",
        "Mình chuyển nơi ở từ Huế sang Đà Lạt.",
        "Mình vừa đi họp ở Hà Nội hai ngày.",
        "Mình không còn làm kế toán nữa, giờ chuyển sang kỹ sư dữ liệu.",
        "Mình đùa hay chuyển sang đầu bếp, chứ không phải nghề hiện tại.",
    ):
        agent.reply("u", "t", turn)
    profile = agent.profile_store.read_text("u")
    assert profile.count("- location:") == profile.count("- profession:") == 1
    assert "- location: Đà Lạt" in profile
    assert "- profession: kỹ sư dữ liệu" in profile
    assert "Huế" not in profile and "Hà Nội" not in profile
    assert "kế toán" not in profile and "đầu bếp" not in profile
    answer = agent.reply("u", "fresh", "Hiện tại mình ở đâu và làm nghề gì?")["response"]
    assert "Đà Lạt" in answer and "kỹ sư dữ liệu" in answer


def test_style_interests_and_pet_are_selected_by_question(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    agent.reply("u", "a", "Mình tên Mai. Mình thích Python và AI.")
    agent.reply("u", "a", "Mình muốn bạn trả lời ngắn gọn thành 3 bullet.")
    agent.reply("u", "a", "Mình thích cách giải thích ưu tiên so sánh trade-off.")
    agent.reply("u", "a", "Mình muốn bạn trả lời có bullet.")
    agent.reply("u", "a", "Mình nuôi một bé corgi tên Bơ.")
    answer = agent.reply("u", "b", "Nhắc lại tên, mối quan tâm và style trả lời mình thích.")["response"]
    assert all(text in answer for text in ("Mai", "Python", "AI", "3 bullet", "ngắn gọn", "trade-off"))
    assert answer.count("\n") == 2
    assert "corgi" not in answer
    assert "corgi" in agent.reply("u", "b", "Mình nuôi con gì?")["response"]
    agent.reply("u", "a", "Mình muốn bạn trả lời thành 2 bullet.")
    assert "2 bullet" in agent.profile_store.facts("u")["response_style"]
    assert "3 bullet" not in agent.profile_store.facts("u")["response_style"]

    agent.reply(
        "another", "c", "Mình muốn bạn trả lời ngắn gọn thành 3 bullet. "
        "Mình thích cách giải thích ưu tiên so sánh trade-off.",
    )
    assert agent.profile_store.facts("another")["response_style"] == (
        "3 bullet, ngắn gọn, ưu tiên trade-off"
    )


def test_explicit_style_corrections_remove_conflicting_traits_and_persist(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    agent.reply("u", "source", "Mình muốn bạn trả lời ngắn gọn thành 3 bullet, ưu tiên trade-off.")
    original_profile = agent.profile_store.read_text("u")
    recalled = agent.reply("u", "source", "Nhắc lại style trả lời mình thích.")["response"]
    assert "3 bullet" in recalled and "trade-off" in recalled
    assert agent.profile_store.read_text("u") == original_profile

    agent.reply("u", "source", "Từ giờ viết một đoạn văn, không dùng bullet nữa.")
    agent.reply("u", "source", "Không cần ưu tiên trade-off nữa.")
    corrected = agent.profile_store.read_text("u")
    style = agent.profile_store.facts("u")["response_style"]
    assert "đoạn văn" in style and "ngắn gọn" in style
    assert "bullet" not in style and "trade-off" not in style
    assert corrected.count("- response_style:") == 1

    for target in (agent, AdvancedAgent(agent.config, force_offline=True)):
        answer = target.reply("u", "fresh", "Nhắc lại style trả lời mình thích.")["response"]
        assert "đoạn văn" in answer and "ngắn gọn" in answer
        assert "bullet" not in answer and "trade-off" not in answer


def test_recall_instructions_without_question_marks_do_not_overwrite_profile(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    agent.reply("u", "source", "Mình tên Mai. Mình thích Python. Mình nuôi một bé corgi tên Bơ.")
    agent.reply("u", "source", "Mình muốn bạn trả lời ngắn gọn.")
    before = agent.profile_store.read_text("u")

    agent.reply("u", "recall", "Nhắc lại giúp mình tên và style trả lời mình thích trong cuộc trò chuyện này.")
    agent.reply("u", "recall", "Nhắc lại giúp mình: tên, món ăn yêu thích và mình nuôi con gì.")

    assert agent.profile_store.read_text("u") == before
    answer = agent.reply("u", "new", "Mình tên gì và mình nuôi con gì?")["response"]
    assert "Mai" in answer and "corgi" in answer


def test_temporary_topic_stays_in_thread_and_survives_summary(tmp_path: Path) -> None:
    agent = make_agent(tmp_path, threshold=55, keep=2)
    agent.reply("u", "topic", "Mình muốn bàn về cache invalidation trong hệ thống phân tán.")
    for index in range(8):
        agent.reply("u", "topic", f"Lượt {index}: thêm bối cảnh dài cho cuộc trao đổi hiện tại.")
    assert agent.compaction_count("topic") > 0
    assert "cache invalidation" in agent.reply("u", "topic", "Chủ đề đang bàn là gì?")["response"]
    assert "cache invalidation" in agent.reply("u", "topic", "Yêu cầu đang mở là gì?")["response"]
    assert "chưa biết" in agent.reply("u", "new", "Chủ đề đang bàn là gì?")["response"]
    assert agent.memory_file_size("u") == 0


def test_compaction_preserves_profile_correction_without_full_history_copy(tmp_path: Path) -> None:
    agent = make_agent(tmp_path, threshold=75, keep=3)
    agent.reply("u", "long", "Mình tên Mai. Mình đang ở Huế.")
    for index in range(12):
        agent.reply("u", "long", f"Lượt {index}: trao đổi thêm về thiết kế cache và các đánh đổi vận hành.")
    agent.reply("u", "long", "Mình chuyển nơi ở từ Huế sang Đà Lạt.")
    for index in range(12, 25):
        agent.reply("u", "long", f"Lượt {index}: tiếp tục trao đổi về thiết kế cache và vận hành.")

    context = agent.compact_memory.context("long")
    assert agent.compaction_count("long") > 1
    # A new append may leave one extra message when the context is below threshold.
    assert len(context["messages"]) <= agent.config.compact_keep_messages + 1
    assert "Lượt 0" not in str(context)
    assert not hasattr(agent, "sessions")
    assert "Đà Lạt" in agent.reply("u", "long", "Hiện tại mình ở đâu?")["response"]
    assert "Huế" not in agent.reply("u", "fresh", "Hiện tại mình ở đâu?")["response"]


def test_accounting_uses_profile_summary_and_recent_before_output(tmp_path: Path, monkeypatch) -> None:
    agent = make_agent(tmp_path, threshold=60, keep=3)
    agent.reply("u", "t", "Mình tên Mai. Mình muốn bàn về cache invalidation.")
    for index in range(8):
        agent.reply("u", "t", f"Lượt {index}: thêm chi tiết về cách cache hoạt động trong hệ thống.")

    captured = {}
    original = agent._offline_response

    def capture(user_id: str, thread_id: str, message: str) -> str:
        context = agent.compact_memory.context(thread_id)
        captured["profile"] = agent.profile_store.read_text(user_id)
        captured["summary"] = context["summary"]
        captured["recent"] = "\n".join(
            f"{item['role']}: {item['content']}" for item in context["messages"]
        )
        captured["last_role"] = context["messages"][-1]["role"]
        return original(user_id, thread_id, message)

    monkeypatch.setattr(agent, "_offline_response", capture)
    prior_prompt = agent.prompt_token_usage("t")
    prior_output = agent.token_usage("t")
    question = "Mình tên gì?"
    result = agent.reply("u", "t", question)
    assert captured["profile"] and captured["summary"] and captured["recent"]
    assert captured["last_role"] == "user"
    assert captured["recent"].count(f"user: {question}") == 1
    assert result["prompt_tokens_processed"] == sum(
        estimate_tokens(captured[part]) for part in ("profile", "summary", "recent")
    )
    assert result["token_usage"] == estimate_tokens(result["response"])
    assert agent.prompt_token_usage("t") == prior_prompt + result["prompt_tokens_processed"]
    assert agent.token_usage("t") == prior_output + result["token_usage"]
    assert agent.prompt_token_usage("missing") == agent.token_usage("missing") == 0
    assert agent.compaction_count("missing") == 0
    other = agent.reply("u", "other", "Mình tên gì?")
    assert agent.token_usage("other") == other["token_usage"]
    assert agent.prompt_token_usage("other") == other["prompt_tokens_processed"]
    assert agent.token_usage("t") == prior_output + result["token_usage"]


def test_determinism_and_long_thread_prompt_load(tmp_path: Path) -> None:
    first = make_agent(tmp_path / "first", threshold=85, keep=4)
    second = make_agent(tmp_path / "second", threshold=85, keep=4)
    baseline = BaselineAgent(config=first.config, force_offline=True)
    turns = ["Mình tên Mai. Mình đang ở Huế."] + [
        f"Lượt {index}: mình muốn bàn về chi phí và độ trễ của cache trong hệ thống phân tán."
        for index in range(30)
    ] + ["Hiện tại mình ở đâu?"]
    first_results = [first.reply("u", "long", turn) for turn in turns]
    second_results = [second.reply("u", "long", turn) for turn in turns]
    for turn in turns:
        baseline.reply("u", "long", turn)

    assert first_results == second_results
    assert first.token_usage("long") == second.token_usage("long")
    assert first.prompt_token_usage("long") == second.prompt_token_usage("long")
    assert first.compaction_count("long") > 0
    assert first.prompt_token_usage("long") < baseline.prompt_token_usage("long")
    assert first.memory_file_size("u") > 0
    assert first_results[-1]["response"] == second_results[-1]["response"]
