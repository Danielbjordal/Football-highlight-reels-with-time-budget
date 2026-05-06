from __future__ import annotations
"""Optional lightweight pipeline activity profiling."""

import csv
import os
import threading
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Dict


CSV_FIELDNAMES = [
    "timestamp_s",
    "event_type",
    "stage",
    "detail",
    "duration_s",
    "event_id",
]


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a conventional truthy environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class _NoOpStage(AbstractContextManager["_NoOpStage"]):
    def __enter__(self) -> "_NoOpStage":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


class NoOpProfiler:
    """Disabled profiler with near-zero behavioral impact."""

    enabled = False

    def stage(self, stage: str, detail: str = "") -> _NoOpStage:
        return _NoOpStage()

    def instant(self, stage: str, detail: str = "") -> None:
        return None

    def close(self) -> None:
        return None


class _ProfileStage(AbstractContextManager["_ProfileStage"]):
    def __init__(self, profiler: "PipelineProfiler", stage: str, detail: str) -> None:
        self.profiler = profiler
        self.stage_name = stage
        self.detail = detail
        self.event_id = profiler.next_event_id()
        self.start_ts = 0.0

    def __enter__(self) -> "_ProfileStage":
        self.start_ts = self.profiler.timestamp_s()
        self.profiler.write_row(
            {
                "timestamp_s": f"{self.start_ts:.6f}",
                "event_type": "start",
                "stage": self.stage_name,
                "detail": self.detail,
                "duration_s": "",
                "event_id": str(self.event_id),
            }
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        end_ts = self.profiler.timestamp_s()
        duration_s = end_ts - self.start_ts
        detail = self.detail
        if exc is not None:
            detail = f"{detail}; error={exc}" if detail else f"error={exc}"
        self.profiler.write_row(
            {
                "timestamp_s": f"{end_ts:.6f}",
                "event_type": "end",
                "stage": self.stage_name,
                "detail": detail,
                "duration_s": f"{duration_s:.6f}",
                "event_id": str(self.event_id),
            }
        )
        return False


class PipelineProfiler:
    """CSV event logger for coarse and fine-grained pipeline activity."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.enabled = True
        self._start_time = time.perf_counter()
        self._event_counter = 0
        self._lock = threading.Lock()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.output_path.exists()
        self._fh = self.output_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            self._writer.writeheader()
            self._fh.flush()

    def timestamp_s(self) -> float:
        return time.perf_counter() - self._start_time

    def next_event_id(self) -> int:
        with self._lock:
            self._event_counter += 1
            return self._event_counter

    def write_row(self, row: Dict[str, str]) -> None:
        with self._lock:
            self._writer.writerow(row)
            self._fh.flush()

    def stage(self, stage: str, detail: str = "") -> _ProfileStage:
        return _ProfileStage(self, stage=stage, detail=detail)

    def instant(self, stage: str, detail: str = "") -> None:
        event_id = self.next_event_id()
        self.write_row(
            {
                "timestamp_s": f"{self.timestamp_s():.6f}",
                "event_type": "instant",
                "stage": stage,
                "detail": detail,
                "duration_s": "",
                "event_id": str(event_id),
            }
        )

    def close(self) -> None:
        with self._lock:
            self._fh.flush()
            self._fh.close()


_ACTIVE_PROFILER: PipelineProfiler | NoOpProfiler = NoOpProfiler()


def set_active_profiler(profiler: PipelineProfiler | NoOpProfiler) -> None:
    global _ACTIVE_PROFILER
    _ACTIVE_PROFILER = profiler


def get_profiler() -> PipelineProfiler | NoOpProfiler:
    return _ACTIVE_PROFILER
