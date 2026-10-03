from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Estimate one token per four stripped characters, with a minimum of one."""
    stripped = text.strip()
    return max(1, len(stripped) // 4) if stripped else 0


@dataclass
class UserProfileStore:
    """Store one UTF-8 markdown profile per user."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        if not isinstance(user_id, str) or not re.fullmatch(r"[a-z0-9_-]+", user_id):
            raise ValueError("user_id must contain only lowercase ASCII letters, digits, _ or -")
        if user_id in {"con", "prn", "aux", "nul"} or re.fullmatch(r"(?:com|lpt)[1-9]", user_id):
            raise ValueError("user_id is a reserved Windows name")
        root = self.root_dir.resolve()
        path = root / user_id / "User.md"
        if not path.resolve().is_relative_to(root):
            raise ValueError("user_id resolves outside root_dir")
        return path

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text:
            return False
        content = self.read_text(user_id)
        if search_text not in content or search_text == replacement:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.is_file() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Read simple `- key: value` profile entries."""
        return dict(re.findall(r"^- ([a-z_]+): (.+)$", self.read_text(user_id), re.MULTILINE))

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Replace a keyed fact and keep one current value for each key."""
        if not re.fullmatch(r"[a-z_]+", key) or "\n" in value or "\r" in value:
            raise ValueError("fact must be a single-line keyed value")
        facts = self.facts(user_id)
        if facts.get(key) == value:
            return
        facts[key] = value
        self.write_text(user_id, "# User profile\n\n" + "".join(f"- {name}: {fact}\n" for name, fact in facts.items()))


_PERSON = r"(?:mình|tôi)"
_PLACE = r"(?-i:[A-ZĐ][^\W\d_]*(?:\s+[A-ZĐ][^\W\d_]*){0,2})"
_PROFESSION = r"[A-Za-zÀ-ỹ][\wÀ-ỹ]*(?:\s+[A-Za-zÀ-ỹ][\wÀ-ỹ]*){0,3}"


def _clean_fact(value: str) -> str:
    return value.strip(" \t.,:;!?\"'“”")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract explicit first-person facts, ignoring questions and speculation."""
    updates: dict[str, str] = {}
    for sentence in re.split(r"[.!?\n]+", message):
        if not sentence.strip() or re.search(
            r"\b(?:nếu|giả sử|đùa|giả định)\b|^\s*ví dụ\b|\bví dụ\b(?!\s+thực\s+(?:tế|chiến|tiễn)\b)",
            sentence,
            re.IGNORECASE,
        ):
            continue
        name = re.search(rf"(?<!bạn )\b(?:{_PERSON}\s+tên(?:\s+là)?|tên\s+{_PERSON}\s+là)\s+([^,;]+?)(?=\s+(?:và|hiện|đang|cho|nhưng)\b|[,;]|$)", sentence, re.IGNORECASE)
        if name and _clean_fact(name.group(1)).casefold() not in {"gì", "ai"}:
            updates["name"] = _clean_fact(name.group(1))

        location = re.search(rf"\b{_PERSON}\s+(?:(?:hiện|đang|vẫn)\s+)?ở\s+({_PLACE})\b", sentence, re.IGNORECASE)
        if not location and re.search(rf"\b{_PERSON}\s+tên\s+là\b", sentence, re.IGNORECASE):
            location = re.search(rf"[,;]\s*hiện\s+ở\s+({_PLACE})\b", sentence, re.IGNORECASE)
        moved_location = re.search(rf"\b{_PERSON}\s+(?:đính chính[:,]?\s*)?(?:đã\s+)?(?:chuyển|dời)\s+nơi ở\s+từ\s+{_PLACE}\s+sang\s+({_PLACE})\b", sentence, re.IGNORECASE)
        if moved_location:
            location = moved_location
        if not location:
            location = re.search(rf"\bnơi ở(?: hiện tại)?(?: của {_PERSON})?\s+(?:là|đã cập nhật từ {_PLACE} sang)\s+({_PLACE})\b", sentence, re.IGNORECASE)
        temporary_visit = re.search(r"\b(?:để|tham dự|dự)\b[^.]*\b(?:họp|du lịch|công tác|hai ngày)\b", sentence[location.end():], re.IGNORECASE) if location else None
        if location and not temporary_visit and not re.search(r"\b(?:đi|bay|họp|du lịch|không phải nơi ở)\b", sentence[:location.end()], re.IGNORECASE):
            updates["location"] = _clean_fact(location.group(1))

        profession = re.search(rf"\b(?:{_PERSON}\s+(?:(?:hiện|đang|vẫn)\s+)?làm|(?:và\s+)?đang\s+làm|nghề nghiệp(?: hiện tại)?(?: của {_PERSON})?\s+(?:vẫn\s+)?là)\s+({_PROFESSION})", sentence, re.IGNORECASE)
        switched = re.search(rf"\b(?:giờ|hiện tại|nay)\s+(?:{_PERSON}\s+)?chuyển sang\s+({_PROFESSION})", sentence, re.IGNORECASE)
        if switched:
            profession = switched
        prior = sentence[:profession.start()] if profession else ""
        self_asserted = not re.search(r"\bbạn\s+mình\b", prior, re.IGNORECASE) and (not profession or not profession.group().lower().startswith("đang làm") or re.search(rf"\b{_PERSON}\b", prior, re.IGNORECASE))
        if profession and self_asserted and not re.search(r"\b(?:không|chưa|đùa|hay là)\s+(?:còn\s+)?(?:đang\s+)?làm\s*$", prior, re.IGNORECASE):
            value = _clean_fact(profession.group(1))
            value = re.split(r"\s+(?:cho|với|nhưng|chứ|nữa|để|ở)\b", value, maxsplit=1, flags=re.IGNORECASE)[0]
            if value and not value.lower().startswith(("việc", "ở ")):
                updates["profession"] = value

        interest = re.search(rf"\b{_PERSON}\s+(?:(?:đang|vẫn)\s+)?(?:thích|quan tâm(?: nhiều đến)?|học(?: thêm về)?|đọc về)\s+([^,;]+(?:,\s*[^,;]+)*)", sentence, re.IGNORECASE)
        if interest:
            value = re.split(r"\s+(?:vì|hơn|khi|nên)\b", _clean_fact(interest.group(1)), maxsplit=1, flags=re.IGNORECASE)[0]
            if value and len(value) <= 80 and not re.match(r"(?:cách|kiểu|tin|không|đồ uống|món ăn|câu trả lời)\b", value, re.IGNORECASE):
                updates["interests"] = re.sub(r"\s+và\s+", ", ", value, flags=re.IGNORECASE)

        food = re.search(r"\bmón ăn yêu thích\s+là\s+([^,;]+)", sentence, re.IGNORECASE)
        drink = re.search(r"\bđồ uống yêu thích\s+là\s+([^,;]+)", sentence, re.IGNORECASE)
        if food:
            updates["favorite_food"] = _clean_fact(food.group(1))
        if drink:
            updates["favorite_drink"] = _clean_fact(drink.group(1))

    style_sentences = [part for part in re.split(r"[.!?\n]+", message) if re.search(r"(?:mình muốn|mình (?:cũng )?mong|mình thích cách|hãy|bạn nên)", part, re.IGNORECASE) and re.search(r"(?:trả lời|giải thích|style|bullet|trade-off)", part, re.IGNORECASE)]
    style_text = " ".join(style_sentences)
    traits = []
    bullet = re.search(r"\b(\d+)\s+bullet\b", style_text, re.IGNORECASE)
    if bullet:
        traits.append(f"{bullet.group(1)} bullet")
    elif re.search(r"\bbullet\b", style_text, re.IGNORECASE):
        traits.append("bullet")
    if re.search(r"\b(?:ngắn gọn|ngắn|gọn)\b", style_text, re.IGNORECASE):
        traits.append("ngắn gọn")
    if re.search(r"\b(?:ví dụ thực tế|ví dụ thực chiến|ví dụ thực tiễn)\b", style_text, re.IGNORECASE):
        traits.append("có ví dụ thực tế")
    if re.search(r"\b(?:ưu tiên|nhấn|so sánh|bám)\b[^.]*\btrade-off\b", style_text, re.IGNORECASE):
        traits.append("ưu tiên trade-off")
    if traits:
        updates["response_style"] = ", ".join(traits)
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Carry forward keyed facts and a bounded set of user discussion topics."""
    if max_items < 1:
        raise ValueError("max_items must be positive")
    facts: dict[str, str] = {}
    topics: list[str] = []
    for item in messages:
        content = item.get("content", "")
        if item.get("role") == "summary":
            for key, value in re.findall(r"^Fact ([a-z_]+): (.+)$", content, re.MULTILINE):
                facts[key] = value
            topics.extend(re.findall(r"^Topic: (.+)$", content, re.MULTILINE))
        elif item.get("role") == "user":
            facts.update(extract_profile_updates(content))
            for sentence in re.split(r"[.!?\n]+", content):
                sentence = sentence.strip()
                if re.search(r"\b(?:tìm hiểu|so sánh|bàn về|đọc về|quan tâm|muốn|thảo luận)\b", sentence, re.IGNORECASE):
                    topic = sentence[:100].rstrip()
                    if topic and topic not in topics:
                        topics.append(topic)
    lines = [f"Fact {key}: {value[:80]}" for key, value in facts.items()]
    for topic in topics[-max_items:]:
        line = f"Topic: {topic[:80]}"
        if len("\n".join(lines)) + len(line) + 1 <= 1200:
            lines.append(line)
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.threshold_tokens) is not int or self.threshold_tokens < 1:
            raise ValueError("threshold_tokens must be a positive integer")
        if type(self.keep_messages) is not int or self.keep_messages < 1:
            raise ValueError("keep_messages must be a positive integer")

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        messages = thread["messages"]
        messages.append({"role": role, "content": content})
        total = estimate_tokens(thread["summary"]) + sum(estimate_tokens(item["content"]) for item in messages)
        if total <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return
        older = messages[:-self.keep_messages]
        previous = [{"role": "summary", "content": thread["summary"]}] if thread["summary"] else []
        thread["summary"] = summarize_messages(previous + older)
        thread["messages"] = messages[-self.keep_messages:]
        thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.get(thread_id, {"messages": [], "summary": "", "compactions": 0})
        return {"messages": [item.copy() for item in thread["messages"]], "summary": thread["summary"], "compactions": thread["compactions"]}

    def compaction_count(self, thread_id: str) -> int:
        return self.context(thread_id)["compactions"]
