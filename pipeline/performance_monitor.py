from __future__ import annotations
"""Lightweight benchmarking helpers for optional pipeline instrumentation."""

import csv
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore


CSV_FIELDNAMES = [
    "setup_name",
    "run_id",
    "game_id",
    "component",
    "device",
    "runtime_s",
    "throughput_fps",
    "throughput_clips_s",
    "gpu_util_avg_pct",
    "gpu_util_peak_pct",
    "gpu_mem_avg_mb",
    "gpu_mem_peak_mb",
    "cpu_util_avg_pct",
    "cpu_util_peak_pct",
    "ram_avg_mb",
    "ram_peak_mb",
    "notes",
]


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a conventional truthy/falsey environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def make_run_id() -> str:
    """Generate a timestamp-based identifier for one benchmarked pipeline run."""
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def resolve_device_label(raw_device: str | None) -> str:
    """Resolve user-facing device names into a concrete label when possible."""
    if raw_device is None:
        return "cpu"
    normalized = raw_device.strip().lower()
    if normalized not in {"auto", "cuda"}:
        return raw_device
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu" if normalized == "auto" else raw_device
    if normalized == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return raw_device


def probe_video_stats(video_path: Path) -> Dict[str, float | None]:
    """Best-effort video stats used for throughput calculations."""
    if cv2 is None:
        return {"frame_count": None, "duration_s": None}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"frame_count": None, "duration_s": None}
    try:
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        duration_s = (frame_count / fps) if frame_count > 0 and fps > 0 else None
        return {
            "frame_count": frame_count if frame_count > 0 else None,
            "duration_s": duration_s,
        }
    finally:
        cap.release()


def _safe_mean(values: List[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _safe_max(values: List[float]) -> float | None:
    return max(values) if values else None


def _fmt_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


@dataclass
class PerformanceResult:
    """One completed monitor interval."""

    setup_name: str
    run_id: str
    game_id: str
    component: str
    device: str
    runtime_s: float
    throughput_fps: float | None
    throughput_clips_s: float | None
    gpu_util_avg_pct: float | None
    gpu_util_peak_pct: float | None
    gpu_mem_avg_mb: float | None
    gpu_mem_peak_mb: float | None
    cpu_util_avg_pct: float | None
    cpu_util_peak_pct: float | None
    ram_avg_mb: float | None
    ram_peak_mb: float | None
    notes: str
    frame_count: float | None = None
    clip_count: float | None = None

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "setup_name": self.setup_name,
            "run_id": self.run_id,
            "game_id": self.game_id,
            "component": self.component,
            "device": self.device,
            "runtime_s": f"{self.runtime_s:.6f}",
            "throughput_fps": "" if self.throughput_fps is None else f"{self.throughput_fps:.6f}",
            "throughput_clips_s": "" if self.throughput_clips_s is None else f"{self.throughput_clips_s:.6f}",
            "gpu_util_avg_pct": "" if self.gpu_util_avg_pct is None else f"{self.gpu_util_avg_pct:.6f}",
            "gpu_util_peak_pct": "" if self.gpu_util_peak_pct is None else f"{self.gpu_util_peak_pct:.6f}",
            "gpu_mem_avg_mb": "" if self.gpu_mem_avg_mb is None else f"{self.gpu_mem_avg_mb:.6f}",
            "gpu_mem_peak_mb": "" if self.gpu_mem_peak_mb is None else f"{self.gpu_mem_peak_mb:.6f}",
            "cpu_util_avg_pct": "" if self.cpu_util_avg_pct is None else f"{self.cpu_util_avg_pct:.6f}",
            "cpu_util_peak_pct": "" if self.cpu_util_peak_pct is None else f"{self.cpu_util_peak_pct:.6f}",
            "ram_avg_mb": "" if self.ram_avg_mb is None else f"{self.ram_avg_mb:.6f}",
            "ram_peak_mb": "" if self.ram_peak_mb is None else f"{self.ram_peak_mb:.6f}",
            "notes": self.notes,
        }


class _ResourceSampler:
    """Background sampler for process and optional GPU metrics."""

    def __init__(self, device: str, interval_s: float) -> None:
        self.device = device
        self.interval_s = max(float(interval_s), 0.1)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

        self.cpu_samples: List[float] = []
        self.ram_samples: List[float] = []
        self.gpu_util_samples: List[float] = []
        self.gpu_mem_samples: List[float] = []

        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        self._cpu_count = float(psutil.cpu_count() or 1) if psutil is not None else 1.0
        self._gpu_handle: Any = None
        self._nvml_initialized = False
        self._gpu_notes: List[str] = []
        self._setup_gpu_handle()

    def _setup_gpu_handle(self) -> None:
        if pynvml is None:
            return
        if not self.device.lower().startswith("cuda"):
            return
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            index = 0
            if ":" in self.device:
                try:
                    index = int(self.device.split(":", 1)[1])
                except ValueError:
                    index = 0
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as exc:
            self._gpu_handle = None
            self._gpu_notes.append(f"gpu monitoring unavailable: {exc}")

    def start(self) -> None:
        if self._process is not None:
            self._process.cpu_percent(interval=None)
        self.thread = threading.Thread(target=self._run, name="performance-monitor", daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, List[float]]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=self.interval_s * 4)
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return {
            "cpu": self.cpu_samples,
            "ram": self.ram_samples,
            "gpu_util": self.gpu_util_samples,
            "gpu_mem": self.gpu_mem_samples,
        }

    def notes(self) -> str:
        return "; ".join(self._gpu_notes)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self._sample_once()
            self.stop_event.wait(self.interval_s)

    def _sample_once(self) -> None:
        if self._process is not None:
            try:
                cpu_pct = self._process.cpu_percent(interval=None) / max(self._cpu_count, 1.0)
                mem_mb = self._process.memory_info().rss / (1024 * 1024)
                self.cpu_samples.append(cpu_pct)
                self.ram_samples.append(mem_mb)
            except Exception:
                pass

        if self._gpu_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                self.gpu_util_samples.append(float(util.gpu))
                self.gpu_mem_samples.append(mem.used / (1024 * 1024))
            except Exception as exc:
                self._gpu_handle = None
                self._gpu_notes.append(f"gpu monitoring stopped: {exc}")


class PerformanceMonitor:
    """Context manager that samples resources while one component runs."""

    def __init__(
        self,
        *,
        component: str,
        metadata: Dict[str, Any],
        enabled: bool,
        sample_interval_s: float,
        collector: Any = None,
    ) -> None:
        self.component = component
        self.metadata = metadata
        self.enabled = enabled
        self.sample_interval_s = sample_interval_s
        self.collector = collector
        self._sampler: _ResourceSampler | None = None
        self._start_time: float | None = None
        self.result: PerformanceResult | None = None

    def __enter__(self) -> "PerformanceMonitor":
        if not self.enabled:
            return self
        self._start_time = time.perf_counter()
        self._sampler = _ResourceSampler(device=str(self.metadata.get("device", "cpu")), interval_s=self.sample_interval_s)
        self._sampler.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if not self.enabled:
            return False

        runtime_s = time.perf_counter() - (self._start_time or time.perf_counter())
        sample_data = self._sampler.stop() if self._sampler is not None else {"cpu": [], "ram": [], "gpu_util": [], "gpu_mem": []}
        notes = []
        base_note = str(self.metadata.get("notes", "")).strip()
        if base_note:
            notes.append(base_note)
        if self._sampler is not None and self._sampler.notes():
            notes.append(self._sampler.notes())
        if exc is not None:
            notes.append(f"error: {exc}")

        frame_count = self._coerce_float(self.metadata.get("frame_count"))
        clip_count = self._coerce_float(self.metadata.get("clip_count"))
        result = PerformanceResult(
            setup_name=str(self.metadata.get("setup_name", "")),
            run_id=str(self.metadata.get("run_id", "")),
            game_id=str(self.metadata.get("game_id", "")),
            component=self.component,
            device=str(self.metadata.get("device", "cpu")),
            runtime_s=runtime_s,
            throughput_fps=(frame_count / runtime_s) if frame_count is not None and runtime_s > 0 else None,
            throughput_clips_s=(clip_count / runtime_s) if clip_count is not None and runtime_s > 0 else None,
            gpu_util_avg_pct=_safe_mean(sample_data["gpu_util"]),
            gpu_util_peak_pct=_safe_max(sample_data["gpu_util"]),
            gpu_mem_avg_mb=_safe_mean(sample_data["gpu_mem"]),
            gpu_mem_peak_mb=_safe_max(sample_data["gpu_mem"]),
            cpu_util_avg_pct=_safe_mean(sample_data["cpu"]),
            cpu_util_peak_pct=_safe_max(sample_data["cpu"]),
            ram_avg_mb=_safe_mean(sample_data["ram"]),
            ram_peak_mb=_safe_max(sample_data["ram"]),
            notes="; ".join(n for n in notes if n),
            frame_count=frame_count,
            clip_count=clip_count,
        )
        self.result = result
        if self.collector is not None:
            self.collector(result)
        return False

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class BenchmarkSession:
    """Aggregate per-component measurements and append a stable CSV log."""

    def __init__(
        self,
        *,
        enabled: bool,
        setup_name: str,
        run_id: str,
        game_id: str,
        output_csv_path: Path,
        sample_interval_s: float,
    ) -> None:
        self.enabled = enabled
        self.setup_name = setup_name
        self.run_id = run_id
        self.game_id = game_id
        self.output_csv_path = output_csv_path
        self.sample_interval_s = sample_interval_s
        self._results: Dict[str, List[PerformanceResult]] = {}
        self._flushed = False

    def monitor(self, component: str, **metadata: Any) -> PerformanceMonitor:
        base_metadata = {
            "setup_name": self.setup_name,
            "run_id": self.run_id,
            "game_id": self.game_id,
            **metadata,
        }
        return PerformanceMonitor(
            component=component,
            metadata=base_metadata,
            enabled=self.enabled,
            sample_interval_s=self.sample_interval_s,
            collector=self._collect if self.enabled else None,
        )

    def _collect(self, result: PerformanceResult) -> None:
        self._results.setdefault(result.component, []).append(result)

    def flush(self) -> List[PerformanceResult]:
        if not self.enabled or self._flushed:
            return self.summary_rows()
        rows = self.summary_rows()
        if rows:
            self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = self.output_csv_path.exists()
            with self.output_csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                if not file_exists:
                    writer.writeheader()
                for row in rows:
                    writer.writerow(row.to_csv_row())
        self._flushed = True
        return rows

    def summary_rows(self) -> List[PerformanceResult]:
        rows: List[PerformanceResult] = []
        for component in sorted(self._results):
            rows.append(self._aggregate_component(component, self._results[component]))
        return rows

    def _aggregate_component(self, component: str, results: List[PerformanceResult]) -> PerformanceResult:
        runtime_s = sum(r.runtime_s for r in results)
        total_frames = sum(r.frame_count for r in results if r.frame_count is not None)
        frame_count = total_frames if any(r.frame_count is not None for r in results) else None
        total_clips = sum(r.clip_count for r in results if r.clip_count is not None)
        clip_count = total_clips if any(r.clip_count is not None for r in results) else None

        gpu_util_samples = [r.gpu_util_avg_pct for r in results if r.gpu_util_avg_pct is not None]
        gpu_util_peaks = [r.gpu_util_peak_pct for r in results if r.gpu_util_peak_pct is not None]
        gpu_mem_samples = [r.gpu_mem_avg_mb for r in results if r.gpu_mem_avg_mb is not None]
        gpu_mem_peaks = [r.gpu_mem_peak_mb for r in results if r.gpu_mem_peak_mb is not None]
        cpu_samples = [r.cpu_util_avg_pct for r in results if r.cpu_util_avg_pct is not None]
        cpu_peaks = [r.cpu_util_peak_pct for r in results if r.cpu_util_peak_pct is not None]
        ram_samples = [r.ram_avg_mb for r in results if r.ram_avg_mb is not None]
        ram_peaks = [r.ram_peak_mb for r in results if r.ram_peak_mb is not None]

        devices = sorted({r.device for r in results if r.device})
        notes = [r.notes for r in results if r.notes]

        return PerformanceResult(
            setup_name=self.setup_name,
            run_id=self.run_id,
            game_id=self.game_id,
            component=component,
            device=",".join(devices),
            runtime_s=runtime_s,
            throughput_fps=(frame_count / runtime_s) if frame_count is not None and runtime_s > 0 else None,
            throughput_clips_s=(clip_count / runtime_s) if clip_count is not None and runtime_s > 0 else None,
            gpu_util_avg_pct=_safe_mean(gpu_util_samples),
            gpu_util_peak_pct=_safe_max(gpu_util_peaks),
            gpu_mem_avg_mb=_safe_mean(gpu_mem_samples),
            gpu_mem_peak_mb=_safe_max(gpu_mem_peaks),
            cpu_util_avg_pct=_safe_mean(cpu_samples),
            cpu_util_peak_pct=_safe_max(cpu_peaks),
            ram_avg_mb=_safe_mean(ram_samples),
            ram_peak_mb=_safe_max(ram_peaks),
            notes="; ".join(notes),
            frame_count=frame_count,
            clip_count=clip_count,
        )

    def print_summary(self) -> None:
        if not self.enabled:
            return
        rows = self.summary_rows()
        if not rows:
            print("[benchmark] enabled, but no component runs were recorded.")
            return
        print()
        print(f"[benchmark] run_id={self.run_id} setup={self.setup_name} csv={self.output_csv_path}")
        for row in rows:
            print(
                "[benchmark] "
                f"{row.component}: runtime={row.runtime_s:.2f}s "
                f"fps={_fmt_metric(row.throughput_fps)} "
                f"clips_s={_fmt_metric(row.throughput_clips_s)} "
                f"cpu_peak={_fmt_metric(row.cpu_util_peak_pct)} "
                f"ram_peak_mb={_fmt_metric(row.ram_peak_mb)} "
                f"gpu_peak={_fmt_metric(row.gpu_util_peak_pct)} "
                f"gpu_mem_peak_mb={_fmt_metric(row.gpu_mem_peak_mb)}"
            )
