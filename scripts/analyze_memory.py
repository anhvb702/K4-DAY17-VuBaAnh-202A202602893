"""Compare offline benchmark metrics with Advanced compaction enabled or disabled."""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_advanced import AdvancedAgent  # noqa: E402
from agent_baseline import BaselineAgent  # noqa: E402
from benchmark import format_rows, load_conversations, run_agent_benchmark  # noqa: E402
from config import (  # noqa: E402
    DEFAULT_COMPACT_KEEP_MESSAGES,
    DEFAULT_COMPACT_THRESHOLD_TOKENS,
    LabConfig,
)
from model_provider import ProviderConfig  # noqa: E402


NO_COMPACT_THRESHOLD = 1_000_000_000


def main() -> None:
    state_root = ROOT / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="analyze-memory-", dir=state_root))
    stub = ProviderConfig(provider="openai", model_name="offline-stub", temperature=0.0)
    base_config = LabConfig(
        base_dir=ROOT,
        data_dir=ROOT / "data",
        state_dir=run_dir,
        compact_threshold_tokens=DEFAULT_COMPACT_THRESHOLD_TOKENS,
        compact_keep_messages=DEFAULT_COMPACT_KEEP_MESSAGES,
        model=stub,
        judge_model=stub,
    )
    print(f"Run state: {run_dir.relative_to(ROOT)}")
    print(
        f"Offline heuristic tokens; default threshold={base_config.compact_threshold_tokens}, "
        f"disabled threshold={NO_COMPACT_THRESHOLD}, "
        f"keep_messages={base_config.compact_keep_messages}"
    )

    for title, filename, namespace in (
        ("Standard", "conversations.json", "standard"),
        ("Stress", "advanced_long_context.json", "stress"),
    ):
        conversations = load_conversations(base_config.data_dir / filename)
        baseline_config = replace(base_config, state_dir=run_dir / namespace / "baseline")
        compact_config = replace(base_config, state_dir=run_dir / namespace / "advanced")
        no_compact_config = replace(
            base_config,
            state_dir=run_dir / namespace / "advanced-no-compact",
            compact_threshold_tokens=NO_COMPACT_THRESHOLD,
        )
        rows = [
            run_agent_benchmark(
                "Baseline", BaselineAgent(baseline_config, force_offline=True),
                conversations, baseline_config,
            ),
            run_agent_benchmark(
                "Advanced", AdvancedAgent(compact_config, force_offline=True),
                conversations, compact_config,
            ),
            run_agent_benchmark(
                "Advanced (no compact)",
                AdvancedAgent(no_compact_config, force_offline=True),
                conversations, no_compact_config,
            ),
        ]
        if rows[-1].compactions != 0:
            raise AssertionError("Disabled compact configuration unexpectedly compacted")
        print(f"\n{title} (data/{filename})")
        print(format_rows(rows))
        print("Raw rows: " + json.dumps([asdict(row) for row in rows], ensure_ascii=False))
        print(
            "Profiles: "
            + str((no_compact_config.state_dir / "profiles").relative_to(ROOT))
        )


if __name__ == "__main__":
    main()
