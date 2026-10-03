from __future__ import annotations

import pytest

from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    summarize_messages,
)


def test_token_estimator_is_deterministic() -> None:
    assert estimate_tokens("") == estimate_tokens(" \n\t") == 0
    assert estimate_tokens("a") > 0
    assert estimate_tokens("abcd " * 40) > estimate_tokens("abcd")
    assert estimate_tokens("xin chào") == estimate_tokens("xin chào")


def test_profile_roundtrip_edit_and_byte_size(tmp_path) -> None:
    store = UserProfileStore(tmp_path)
    assert store.read_text("alice") == ""
    assert store.file_size("alice") == 0
    path = store.write_text("alice", "Huế Huế ☕")
    assert path == tmp_path / "alice" / "User.md"
    assert store.read_text("alice") == "Huế Huế ☕"
    assert store.file_size("alice") == len("Huế Huế ☕".encode("utf-8"))
    assert store.edit_text("alice", "Huế", "Đà Nẵng")
    assert store.read_text("alice") == "Đà Nẵng Huế ☕"
    assert not store.edit_text("alice", "không thấy", "x")
    assert not store.edit_text("alice", "Huế", "Huế")


def test_profiles_are_isolated_and_paths_stay_inside_root(tmp_path) -> None:
    store = UserProfileStore(tmp_path)
    store.write_text("alice", "one")
    store.write_text("alice_2", "two")
    assert store.read_text("alice") == "one"
    assert store.read_text("alice_2") == "two"
    for bad_id in ("../escape", "..\\escape", "C:\\escape", "/tmp/escape", ".", "CON", "Alice"):
        with pytest.raises(ValueError):
            store.path_for(bad_id)


def test_upsert_replaces_correction_without_growth(tmp_path) -> None:
    store = UserProfileStore(tmp_path)
    store.upsert_fact("alice", "location", "Huế")
    store.upsert_fact("alice", "location", "Đà Nẵng")
    current = store.read_text("alice")
    assert store.facts("alice") == {"location": "Đà Nẵng"}
    assert "Huế" not in current
    store.upsert_fact("alice", "location", "Đà Nẵng")
    assert store.read_text("alice") == current


@pytest.mark.parametrize("message,expected", [
    ("Mình tên là An. Mình ở Huế và đang làm backend engineer.", {"name": "An", "location": "Huế", "profession": "backend engineer"}),
    ("Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.", {"profession": "MLOps engineer"}),
    ("Mình muốn bạn trả lời ngắn gọn thành 3 bullet, ưu tiên trade-off.", {"response_style": "3 bullet, ngắn gọn, ưu tiên trade-off"}),
    ("Mình cũng mong bạn ưu tiên bullet ngắn và nói thẳng trade-off. Mình muốn style trả lời có 3 bullet và ví dụ thực chiến.", {"response_style": "3 bullet, ngắn gọn, có ví dụ thực tế, ưu tiên trade-off"}),
    ("Mình thích Python và AI ứng dụng. Đồ uống yêu thích là cà phê sữa đá.", {"interests": "Python, AI ứng dụng", "favorite_drink": "cà phê sữa đá"}),
    ("Mình tên là An. Bạn nhớ tên mình không?", {"name": "An"}),
    ("Mình ở Đà Nẵng. À, đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng.", {"location": "Huế"}),
    ("Mình tên là Lan, hiện ở Cần Thơ và đang làm data engineer.", {"name": "Lan", "location": "Cần Thơ", "profession": "data engineer"}),
])
def test_extracts_asserted_facts(message: str, expected: dict[str, str]) -> None:
    assert extract_profile_updates(message).items() >= expected.items()


@pytest.mark.parametrize("message", [
    "Mình tên gì? Hiện mình ở đâu?",
    "Bạn trả lời ngắn gọn thành 3 bullet được không?",
    "Hà Nội là nơi mình vừa bay ra họp hai ngày, không phải nơi ở hiện tại.",
    "Mình đang ở Hà Nội để họp hai ngày rồi về nhà.",
    "Mình đùa rằng hay là chuyển sang product manager, nhưng đó chỉ là câu đùa.",
    "Mình không làm product manager.",
    "Bạn mình tên là Bình và đang làm bác sĩ ở Huế.",
    "Tin NASA nói phi hành gia đang ở Huế và làm kỹ sư.",
])
def test_extraction_rejects_noise(message: str) -> None:
    assert not extract_profile_updates(message)


def test_summary_preserves_user_facts_and_separates_assistant() -> None:
    summary = summarize_messages([
        {"role": "user", "content": "Mình ở Huế. Mình đang tìm hiểu về memory architecture."},
        {"role": "assistant", "content": "Bạn là bác sĩ."},
        {"role": "user", "content": "Đính chính: giờ mình đang ở Đà Nẵng. Mình muốn so sánh hai cách compact."},
    ])
    assert "Đà Nẵng" in summary
    assert "location: Huế" not in summary
    assert "bác sĩ" not in summary
    assert "compact" in summary


def test_summary_carries_correction_through_repeated_compaction() -> None:
    first = summarize_messages([
        {"role": "user", "content": "Mình tên là Mai. Mình ở Huế. Mình muốn bàn về cache."},
    ])
    second = summarize_messages([
        {"role": "summary", "content": first},
        {"role": "user", "content": "Đính chính: mình đang ở Đà Nẵng. Mình muốn so sánh cache với index."},
    ])
    third = summarize_messages([
        {"role": "summary", "content": second},
        {"role": "assistant", "content": "Bạn làm bác sĩ."},
        {"role": "user", "content": "Mình muốn tiếp tục về index."},
    ])
    assert "Fact name: Mai" in third
    assert "Fact location: Đà Nẵng" in third
    assert "Fact location: Huế" not in third
    assert "cache" in third and "index" in third
    assert "bác sĩ" not in third


def test_compact_keeps_recent_and_previous_summary() -> None:
    memory = CompactMemoryManager(80, 2)
    first = "Mình tên là An. " + "Chi tiết hội thoại " * 25
    memory.append("a", "user", first)
    memory.append("a", "assistant", "Đã hiểu. " * 20)
    memory.append("a", "user", "Mình ở Huế. " + "Bàn về memory " * 20)
    assert memory.compaction_count("a") == 1
    memory.append("a", "assistant", "Cùng xem nhé. " * 15)
    memory.append("a", "user", "Mình chuyển sang Đà Nẵng. " + "Bàn về compact " * 20)
    context = memory.context("a")
    assert memory.compaction_count("a") >= 2
    assert "name: An" in context["summary"]
    assert "location: Đà Nẵng" in context["summary"] or "Đà Nẵng" in context["messages"][-1]["content"]
    assert len(context["messages"]) == 2
    assert context["messages"][0] == {"role": "assistant", "content": "Cùng xem nhé. " * 15}
    assert context["messages"][-1] == {"role": "user", "content": "Mình chuyển sang Đà Nẵng. " + "Bàn về compact " * 20}
    assert memory.context("other") == {"messages": [], "summary": "", "compactions": 0}


def test_compact_threshold_limits_and_reduces_long_context() -> None:
    memory = CompactMemoryManager(80, 2)
    memory.append("a", "user", "short")
    assert memory.compaction_count("a") == 0
    long_message = "Mình đang nghiên cứu cách tổ chức bộ nhớ cho agent. " * 12
    for _ in range(10):
        memory.append("a", "user", long_message)
    context = memory.context("a")
    original_tokens = estimate_tokens("short") + 10 * estimate_tokens(long_message)
    compact_tokens = estimate_tokens(context["summary"]) + sum(estimate_tokens(item["content"]) for item in context["messages"])
    assert compact_tokens < original_tokens
    assert len(context["messages"]) == 2
    assert len(context["summary"]) <= 1200
    count = memory.compaction_count("a")
    memory.append("a", "user", "next")
    assert memory.compaction_count("a") == count + 1
    assert memory.context("a")["messages"][-1]["content"] == "next"


def test_invalid_compact_configuration() -> None:
    with pytest.raises(ValueError):
        CompactMemoryManager(0, 2)
    with pytest.raises(ValueError):
        CompactMemoryManager(80, 0)


def test_recent_messages_over_threshold_are_preserved() -> None:
    memory = CompactMemoryManager(80, 2)
    oversized = "nội dung dài " * 100
    memory.append("thread", "user", oversized)
    memory.append("thread", "assistant", oversized)
    assert memory.compaction_count("thread") == 0
    assert memory.context("thread")["messages"] == [
        {"role": "user", "content": oversized},
        {"role": "assistant", "content": oversized},
    ]


def test_compaction_keeps_early_fact_and_open_topic_then_summarizes_latest_correction() -> None:
    memory = CompactMemoryManager(80, 2)
    early = "Mình tên là Mai. Mình ở Cần Thơ. Mình muốn thiết kế index cho báo cáo đang mở."
    memory.append("review", "user", early)
    for index in range(16):
        memory.append("review", "assistant", f"Nội dung trao đổi phụ {index}. " * 20)
    memory.append("review", "user", "Mình chuyển nơi ở từ Cần Thơ sang Đà Lạt. " + "Trao đổi tiếp. " * 20)
    memory.append("review", "assistant", "Đã cập nhật. " * 20)
    memory.append("review", "user", "Tiếp tục phần tiếp theo. " * 20)

    context = memory.context("review")
    assert context["compactions"] >= 2
    assert "Fact name: Mai" in context["summary"]
    assert "Fact location: Đà Lạt" in context["summary"]
    assert "Fact location: Cần Thơ" not in context["summary"]
    assert "thiết kế index" in context["summary"]
    assert len(context["summary"]) <= 1200


def test_compaction_token_report_separates_summary_from_recent_messages() -> None:
    memory = CompactMemoryManager(80, 2)
    repeated = "Mình muốn bàn về cấu trúc tìm kiếm trong hệ thống. " * 12
    original_tokens = 0
    for _ in range(10):
        original_tokens += estimate_tokens(repeated)
        memory.append("tokens", "user", repeated)

    context = memory.context("tokens")
    summary_tokens = estimate_tokens(context["summary"])
    recent_tokens = sum(estimate_tokens(item["content"]) for item in context["messages"])
    assert summary_tokens + recent_tokens < original_tokens
    assert len(context["messages"]) == 2
    assert context["compactions"] > 0


def test_profile_repeated_fact_upsert_does_not_grow_file(tmp_path) -> None:
    store = UserProfileStore(tmp_path)
    store.upsert_fact("mai", "response_style", "ngắn gọn")
    original = store.read_text("mai")
    original_size = store.file_size("mai")
    for _ in range(5):
        store.upsert_fact("mai", "response_style", "ngắn gọn")
    assert store.read_text("mai") == original
    assert store.file_size("mai") == original_size


def test_extracts_name_from_statement_before_recall_question() -> None:
    assert extract_profile_updates("Mình tên Mai. Bạn có thể nhớ giúp không?") == {"name": "Mai"}


def test_correction_phrase_replaces_location_without_promoting_meeting_city() -> None:
    updates = extract_profile_updates(
        "Mình chuyển nơi ở từ Cần Thơ sang Đà Lạt. Tuần này mình ở Hà Nội để tham dự cuộc họp."
    )
    assert updates == {"location": "Đà Lạt"}


def test_example_style_phrase_does_not_hide_asserted_interests() -> None:
    updates = extract_profile_updates(
        "Mình vẫn thích Python, AI ứng dụng và cách trình bày có ví dụ thực chiến."
    )
    assert "Python" in updates["interests"]
    assert "AI ứng dụng" in updates["interests"]
    assert extract_profile_updates("Ví dụ: mình thích Python.") == {}
