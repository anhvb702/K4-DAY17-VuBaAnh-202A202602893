from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path):
    """Build a deterministic offline config whose paths stay under tmp_path."""
    stub_model = ProviderConfig(
        provider="openai", model_name="stub-model", temperature=0.0
    )
    stub_judge = ProviderConfig(
        provider="openai", model_name="stub-judge", temperature=0.0
    )
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=80,
        compact_keep_messages=2,
        model=stub_model,
        judge_model=stub_judge,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """User profiles are UTF-8 files under the isolated profile root."""
    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / "profiles")
    user_id = "test-user"
    expected = "# Hồ sơ\n\n- name: Nguyễn An\n- place: Huế\n"

    assert store.read_text(user_id) == ""
    path = store.write_text(user_id, expected)
    assert store.read_text(user_id) == expected
    assert path == tmp_path / "state" / "profiles" / user_id / "User.md"
    assert path.is_relative_to(tmp_path)
    disk_text = path.read_bytes().decode("utf-8")
    assert disk_text.replace("\r\n", "\n") == expected

    assert store.edit_text(user_id, "Huế", "Đà Nẵng") is True
    edited = store.read_text(user_id)
    assert "Đà Nẵng" in edited
    assert "Huế" not in edited
    before_missing_edit = path.read_bytes()
    assert store.edit_text(user_id, "không có chuỗi này", "thay thế") is False
    assert path.read_bytes() == before_missing_edit


def test_compact_trigger(tmp_path: Path) -> None:
    """Compaction summarizes old content and preserves the newest messages."""
    manager = CompactMemoryManager(threshold_tokens=80, keep_messages=2)
    messages = [
        ("user", "Mình muốn thảo luận về bộ nhớ phân tán và cách đồng bộ dữ liệu."),
        ("assistant", "Ta có thể xem xét tính nhất quán và độ trễ trong ví dụ này."),
        ("user", "Mình muốn thảo luận về cách phục hồi sau khi một node gặp lỗi."),
        ("assistant", "Có thể dùng nhật ký ghi trước để khôi phục trạng thái."),
        ("user", "Mình muốn thảo luận về chiến lược kiểm tra tính toàn vẹn."),
        ("assistant", "Checksum giúp phát hiện dữ liệu bị thay đổi."),
        ("user", "Câu mới nhất cần được giữ nguyên trong phần recent messages."),
        ("assistant", "Đây là phản hồi mới nhất cần còn nguyên văn."),
    ]
    for role, content in messages:
        manager.append("thread", role, content)

    context = manager.context("thread")
    assert manager.compaction_count("thread") > 0
    assert context["summary"]
    assert context["messages"] == [
        {"role": role, "content": content} for role, content in messages[-2:]
    ]
    assert context["messages"][-1]["content"] == messages[-1][1]


def test_cross_session_recall(tmp_path: Path) -> None:
    """Only Advanced carries a directly stated profile fact across threads."""
    config = make_config(tmp_path / "advanced")
    baseline_config = make_config(tmp_path / "baseline")
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(baseline_config, force_offline=True)
    user_id = "same-user"

    advanced.reply(user_id, "advanced-training", "Mình tên là An.")
    baseline.reply(user_id, "baseline-training", "Mình tên là An.")
    advanced_answer = advanced.reply(user_id, "advanced-recall", "Mình tên gì?")["response"]
    baseline_answer = baseline.reply(user_id, "baseline-recall", "Mình tên gì?")["response"]

    profile_path = config.state_dir / "profiles" / user_id / "User.md"
    assert profile_path.is_file()
    assert profile_path.read_text(encoding="utf-8").strip()
    assert "An" in advanced_answer
    assert "chưa biết" in baseline_answer.casefold()
    assert "An" not in baseline_answer
    assert not (baseline_config.state_dir / "profiles" / user_id).exists()


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Advanced compaction lowers cumulative prompt load and retains a key fact."""
    config = make_config(tmp_path / "advanced")
    baseline_config = make_config(tmp_path / "baseline")
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(baseline_config, force_offline=True)
    user_id = "long-thread-user"
    messages = ["Mình tên là An."] + [
        (
            f"Bản ghi hệ thống số {index}: "
            "cách lưu trữ, kiểm tra và đồng bộ nhiều bản ghi trong môi trường phân tán. "
            "Ta cần xem xét độ trễ, tính nhất quán, khả năng phục hồi và chi phí vận hành."
        )
        for index in range(1, 25)
    ]

    for message in messages:
        advanced.reply(user_id, "advanced-long-thread", message)
        baseline.reply(user_id, "baseline-long-thread", message)

    advanced_answer = advanced.reply(
        user_id, "advanced-long-thread", "Mình tên gì?"
    )["response"]
    baseline.reply(user_id, "baseline-long-thread", "Mình tên gì?")

    assert advanced.compaction_count("advanced-long-thread") > 0
    assert baseline.compaction_count("baseline-long-thread") == 0
    assert advanced.prompt_token_usage("advanced-long-thread") < baseline.prompt_token_usage(
        "baseline-long-thread"
    )
    assert "An" in advanced_answer
