"""Experiment logging.

Two complementary records:
  * ``logs/experiments.jsonl`` — one JSON line per run: config, metrics,
    environment, timing. Machine-readable; feeds the paper's tables.
  * ``LOGBOOK.md`` — human-readable dated journal of decisions and results.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import PROJECT_ROOT, Config
from .env_utils import env_report

LOGBOOK = PROJECT_ROOT / "LOGBOOK.md"


class ExperimentLogger:
    """Collects one run's record and appends it to experiments.jsonl."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.record: dict = {
            "run_name": cfg.run_name,
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": cfg.to_dict(),
            "env": env_report(),
            "metrics": {},
        }
        self._t0 = time.perf_counter()

    def log_metrics(self, **metrics) -> None:
        self.record["metrics"].update(
            {k: (round(float(v), 6) if isinstance(v, (int, float)) else v)
             for k, v in metrics.items()}
        )

    def finish(self) -> Path:
        self.record["duration_s"] = round(time.perf_counter() - self._t0, 1)
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.log_dir / "experiments.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.record, ensure_ascii=False) + "\n")
        return path


def append_logbook(title: str, body: str) -> None:
    """Append a dated entry to LOGBOOK.md."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n## {stamp} — {title}\n\n{body.strip()}\n"
    with open(LOGBOOK, "a", encoding="utf-8") as f:
        f.write(entry)


def read_experiments() -> list[dict]:
    path = PROJECT_ROOT / "logs" / "experiments.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
