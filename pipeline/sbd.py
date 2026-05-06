from __future__ import annotations
"""Repo-side wrapper around the external SportsBD shot-boundary package."""

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import subprocess

from pipeline.profiling import get_profiler


@dataclass(frozen=True)
class Shot:
    """Simple contiguous shot interval derived from boundary timestamps."""
    shot_id: int
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def ensure_ffmpeg_on_path(ffmpeg_path: str) -> None:
    """
    sportsbd calls 'ffmpeg' internally.
    On Windows, if ffmpeg is not on PATH, we add its folder to PATH for this process.
    """
    bin_dir = str(Path(ffmpeg_path).parent)
    current = os.environ.get("PATH", "")
    if bin_dir not in current.split(";"):
        os.environ["PATH"] = bin_dir + ";" + current


def probe_duration_ms(ffprobe_path: str, video_path: Path) -> int:
    """
    Uses ffprobe to get video duration in milliseconds.
    """
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{r.stderr}")
    secs = float(r.stdout.strip())
    return int(secs * 1000)


def boundaries_to_shots(boundary_ms: List[int], duration_ms: int) -> List[Shot]:
    """
    Converts a list of boundary timestamps (ms) to contiguous shots.
    """
    cuts = [0] + sorted(set([t for t in boundary_ms if 0 < t < duration_ms])) + [duration_ms]
    shots: List[Shot] = []
    for i, (s, e) in enumerate(zip(cuts[:-1], cuts[1:])):
        if e <= s:
            continue
        shots.append(Shot(shot_id=i, start_ms=int(s), end_ms=int(e)))
    return shots


def write_shots_csv(shots: List[Shot], out_csv: Path) -> None:
    """Serialize derived shot intervals to the CSV format used by later stages."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["shot_id", "start_ms", "end_ms", "duration_ms"])
        w.writeheader()
        for sh in shots:
            w.writerow(
                {
                    "shot_id": sh.shot_id,
                    "start_ms": sh.start_ms,
                    "end_ms": sh.end_ms,
                    "duration_ms": sh.duration_ms,
                }
            )


def run_sbd(
    video_path: Path,
    out_boundaries_json: Path,
    out_shots_csv: Path,
    threshold: float,
    fps: int,
    ffmpeg_path: str,
    ffprobe_path: str,
    device: str | None = None,
) -> None:
    """
    Runs sportsbd on a video and produces:
    - boundaries.json (raw detections)
    - shots.csv (derived contiguous shots)
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")

    ensure_ffmpeg_on_path(ffmpeg_path)
    profiler = get_profiler()

    # SportsBD owns the deep model and frame-window inference; this wrapper only adapts its outputs.
    from sportsbd import download_model, run_video_inference  # type: ignore

    with profiler.stage("sbd_model_setup", detail=f"video={video_path.name}"):
        checkpoint_path = download_model()

    with profiler.stage("sbd_external_inference", detail=f"video={video_path.name}; fps={fps}; threshold={threshold}"):
        detections: Any = run_video_inference(
            video_path=str(video_path),
            checkpoint_path=checkpoint_path,
            threshold=threshold,
            stride=4,
            t_frames=16,
            fps=fps,
            device=device,
        )

    out_boundaries_json.parent.mkdir(parents=True, exist_ok=True)
    out_boundaries_json.write_text(json.dumps(detections, indent=2), encoding="utf-8")

    with profiler.stage("sbd_postprocess", detail=f"video={video_path.name}"):
        duration_ms = probe_duration_ms(ffprobe_path, video_path)

        boundary_ms: List[int] = []
        if isinstance(detections, list):
            for d in detections:
                if isinstance(d, dict):
                    if "timestamp_ms" in d:
                        boundary_ms.append(int(d["timestamp_ms"]))
                    elif "frame_idx" in d:
                        # Fall back to frame index if the library did not emit explicit timestamps.
                        boundary_ms.append(int((int(d["frame_idx"]) / fps) * 1000))

        shots = boundaries_to_shots(boundary_ms, duration_ms=duration_ms)
        write_shots_csv(shots, out_shots_csv)


if __name__ == "__main__":
    from pipeline.tooling import ffprobe_from_ffmpeg

    ffmpeg = "C:/Users/Daniel/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0.1-full_build/bin/ffmpeg.exe"
    ffprobe = ffprobe_from_ffmpeg(ffmpeg)

    video = Path("data/games/segment.mp4")
    out_dir = Path("outputs/test_sbd")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_sbd(
        video_path=video,
        out_boundaries_json=out_dir / "boundaries.json",
        out_shots_csv=out_dir / "shots.csv",
        threshold=0.7,
        fps=25,
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
    )

    print("Wrote:", out_dir / "shots.csv")
