from __future__ import annotations
"""Helpers for cutting event subclips and concatenating them into highlights."""

import csv
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Tuple

from pipeline.constants import ORDER_FALLBACK
from pipeline.profiling import get_profiler

CLIP_START_TRIM_MS = 170
CROSSFADE_DURATION_SECONDS = 0.5


def ms_to_timestamp(ms: int) -> str:
    """Convert milliseconds to an ffmpeg-friendly seconds string."""
    seconds = ms / 1000.0
    return f"{seconds:.3f}"


def ffprobe_from_ffmpeg_path(ffmpeg_path: str) -> str:
    """Resolve ffprobe next to ffmpeg when available, otherwise rely on PATH."""
    ffmpeg_bin = Path(ffmpeg_path)
    if ffmpeg_bin.exists():
        candidate = ffmpeg_bin.parent / "ffprobe.exe"
        if candidate.exists():
            return str(candidate)
    return "ffprobe"


def clip_has_audio(ffmpeg_path: str, input_video: Path) -> bool:
    """Return True when the clip contains at least one audio stream."""
    cmd = [
        ffprobe_from_ffmpeg_path(ffmpeg_path),
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(input_video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and bool(result.stdout.strip())


def probe_clip_duration_seconds(
    ffmpeg_path: str,
    input_video: Path,
 ) -> float:
    """Read clip duration via ffprobe."""
    cmd = [
        ffprobe_from_ffmpeg_path(ffmpeg_path),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(input_video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe duration failed for {input_video.name}:\n{result.stderr}")
    return float((result.stdout or "0").strip())


def normalize_clip_for_crossfade(
    ffmpeg_path: str,
    input_video: Path,
    output_video: Path,
) -> None:
    """Re-encode one clip into a consistent A/V format for reliable crossfades."""
    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_video),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "25",
    ]
    if clip_has_audio(ffmpeg_path, input_video):
        cmd.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-ar",
                "48000",
                "-ac",
                "2",
            ]
        )
    else:
        duration = probe_clip_duration_seconds(ffmpeg_path, input_video)
        cmd.extend(
            [
                "-f",
                "lavfi",
                "-t",
                f"{duration:.3f}",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-ar",
                "48000",
                "-ac",
                "2",
            ]
        )
    cmd.append(str(output_video))

    profiler = get_profiler()
    with profiler.stage("ffmpeg_normalize_clip", detail=f"input={input_video.name}; output={output_video.name}"):
        result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg normalize failed for {output_video.name}:\n{result.stderr}")


def update_intro_candidate(
    current_clip: Path | None,
    current_order: int | None,
    current_ts: float | None,
    current_event_id: str | None,
    candidate_clip: Path,
    candidate_order: int,
    candidate_ts: float,
    candidate_event_id: str,
) -> tuple[Path | None, int | None, float | None, str | None]:
    """
    Track the best intro candidate while iterating over events.
    The earliest event wins, and timestamp breaks ties within the same order slot.
    """
    if (current_clip is None) or (candidate_order < (current_order or ORDER_FALLBACK)) or (
        candidate_order == (current_order or ORDER_FALLBACK) and candidate_ts < (current_ts or float("inf"))
    ):
        return candidate_clip, candidate_order, candidate_ts, candidate_event_id
    return current_clip, current_order, current_ts, current_event_id


def update_outro_candidate(
    current_clip: Path | None,
    current_order: int | None,
    current_event_id: str | None,
    candidate_clip: Path,
    candidate_order: int,
    candidate_ts: float,
    candidate_event_id: str,
    events_metadata: dict[str, dict[str, Any]],
) -> tuple[Path | None, int | None, str | None]:
    """
    Track the best outro candidate while iterating over events.
    The latest event wins, and timestamp breaks ties within the same order slot.
    """
    current_ts = events_metadata.get(current_event_id, {}).get("from_timestamp") if current_event_id else None
    if (current_clip is None) or (candidate_order > (current_order or -1)) or (
        candidate_order == (current_order or -1) and candidate_ts > (current_ts or -1)
    ):
        return candidate_clip, candidate_order, candidate_event_id
    return current_clip, current_order, current_event_id


def cut_clip(
    ffmpeg_path: str,
    input_video: Path,
    output_video: Path,
    start_ms: int,
    end_ms: int,
) -> None:
    """
    Cuts a subclip from input_video and saves it as output_video.
    We re-encode instead of stream-copying to make boundary cuts more reliable.
    """
    output_video.parent.mkdir(parents=True, exist_ok=True)

    # Trim a few frames from both ends of every assembled clip to hide boundary flash frames.
    if end_ms - start_ms > (2 * CLIP_START_TRIM_MS):
        start_ms += CLIP_START_TRIM_MS
        end_ms -= CLIP_START_TRIM_MS

    start_ts = ms_to_timestamp(start_ms)
    end_ts = ms_to_timestamp(end_ms)

    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(input_video),
        "-ss", start_ts,
        "-to", end_ts,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_video),
    ]

    profiler = get_profiler()
    with profiler.stage("ffmpeg_cut_clip", detail=f"input={input_video.name}; output={output_video.name}"):
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg cut failed for {output_video.name}:\n{result.stderr}")


def read_selected_shots(selected_shots_csv: Path) -> List[Tuple[int, int, int]]:
    """
    Returns list of (shot_id, start_ms, end_ms) for all kept shots, sorted by start_ms.
    """
    if not selected_shots_csv.exists():
        return []
    kept: List[Tuple[int, int, int]] = []
    with selected_shots_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                keep = int(row.get("keep", 0))
            except (TypeError, ValueError):
                keep = 0
            if keep != 1:
                continue
            try:
                shot_id = int(row["shot_id"])
                start_ms = int(row["start_ms"])
                end_ms = int(row["end_ms"])
            except (KeyError, ValueError, TypeError):
                continue
            kept.append((shot_id, start_ms, end_ms))
    kept.sort(key=lambda t: t[1])
    return kept


def concat_clips(
    ffmpeg_path: str,
    clip_paths: list[Path],
    output_video: Path,
    work_dir: Path,
) -> None:
    """
    Concatenates multiple mp4 clips into one output video.
    """
    if not clip_paths:
        raise ValueError("No clips to concatenate")

    work_dir.mkdir(parents=True, exist_ok=True)
    concat_txt = work_dir / "concat.txt"

    with concat_txt.open("w", encoding="utf-8") as f:
        for clip in clip_paths:
            # ffmpeg concat expects one resolved file path per line.
            f.write(f"file '{clip.resolve().as_posix()}'\n")

    cmd = [
        ffmpeg_path,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_txt),
        "-c", "copy",
        str(output_video),
    ]

    profiler = get_profiler()
    with profiler.stage("ffmpeg_concat_clips", detail=f"output={output_video.name}; clips={len(clip_paths)}"):
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed:\n{result.stderr}")


def crossfade_clips(
    ffmpeg_path: str,
    clip_paths: List[Path],
    output_video: Path,
    work_dir: Path,
) -> None:
    """Render the final highlight with audio+video crossfades between clips."""
    if not clip_paths:
        raise ValueError("No clips to crossfade")

    work_dir.mkdir(parents=True, exist_ok=True)

    if len(clip_paths) == 1:
        shutil.copy2(clip_paths[0], output_video)
        return

    with tempfile.TemporaryDirectory(dir=work_dir) as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        normalized_paths: List[Path] = []
        durations: List[float] = []

        for idx, clip in enumerate(clip_paths):
            normalized = tmpdir / f"normalized_{idx:03d}.mp4"
            normalize_clip_for_crossfade(ffmpeg_path=ffmpeg_path, input_video=clip, output_video=normalized)
            normalized_paths.append(normalized)
            durations.append(probe_clip_duration_seconds(ffmpeg_path=ffmpeg_path, input_video=normalized))

        cmd: List[str] = [ffmpeg_path, "-y"]
        for path in normalized_paths:
            cmd.extend(["-i", str(path)])

        filter_parts: List[str] = []
        cumulative_duration = durations[0]
        current_video = "[0:v]"
        current_audio = "[0:a]"

        for idx in range(1, len(normalized_paths)):
            transition_duration = min(CROSSFADE_DURATION_SECONDS, durations[idx - 1], durations[idx])
            offset = max(cumulative_duration - transition_duration, 0.0)
            next_video = f"[{idx}:v]"
            next_audio = f"[{idx}:a]"
            video_out = f"[vxf{idx}]"
            audio_out = f"[axf{idx}]"
            filter_parts.append(
                f"{current_video}{next_video}xfade=transition=fade:duration={transition_duration}:offset={offset}{video_out}"
            )
            filter_parts.append(
                f"{current_audio}{next_audio}acrossfade=d={transition_duration}{audio_out}"
            )
            current_video = video_out
            current_audio = audio_out
            cumulative_duration = cumulative_duration + durations[idx] - transition_duration

        cmd.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                current_video,
                "-map",
                current_audio,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(output_video),
            ]
        )

        profiler = get_profiler()
        with profiler.stage("ffmpeg_crossfade_clips", detail=f"output={output_video.name}; clips={len(clip_paths)}"):
            result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg crossfade failed:\n{result.stderr}")


def assemble_selected_clips(
    ffmpeg_path: str,
    input_video: Path,
    kept_shots: List[Tuple[int, int, int]],
    output_video: Path,
    work_dir: Path,
    debug: bool = False,
) -> None:
    """
    Cuts all kept shots for one event and concatenates them into output_video.
    """
    if not kept_shots:
        raise ValueError("No kept shots to assemble")

    work_dir.mkdir(parents=True, exist_ok=True)

    # Fast path: single shot -> cut directly
    if len(kept_shots) == 1:
        shot_id, start_ms, end_ms = kept_shots[0]
        if debug:
            print(f"  [assembly] single kept shot {shot_id} ({start_ms}-{end_ms} ms) -> {output_video.name}")
        cut_clip(
            ffmpeg_path=ffmpeg_path,
            input_video=input_video,
            output_video=output_video,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        return

    # Multiple shots: cut to temp files then concat
    temp_clips: list[Path] = []
    for idx, (shot_id, start_ms, end_ms) in enumerate(kept_shots):
        temp_path = work_dir / f"part_{idx:02d}_shot_{shot_id}.mp4"
        if debug:
            print(f"  [assembly] cutting shot {shot_id} ({start_ms}-{end_ms} ms) -> {temp_path.name}")
        cut_clip(
            ffmpeg_path=ffmpeg_path,
            input_video=input_video,
            output_video=temp_path,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        temp_clips.append(temp_path)

    if debug:
        kept_ids = [str(sid) for sid, _, _ in kept_shots]
        print(f"  [assembly] concatenating kept shots {kept_ids} into {output_video.name}")

    concat_clips(
        ffmpeg_path=ffmpeg_path,
        clip_paths=temp_clips,
        output_video=output_video,
        work_dir=work_dir / "_concat",
    )


def read_replay_shots(replay_csv: Path) -> List[Tuple[int, int, int]]:
    """
    Parse replay_shots.csv into the tuple shape used by the generic assembler.
    """
    if not replay_csv.exists():
        return []
    rows: List[Tuple[int, int, int]] = []
    with replay_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sid = int(row.get("shot_id", -1))
                s_ms = int(row.get("start_ms", -1))
                e_ms = int(row.get("end_ms", -1))
            except (TypeError, ValueError):
                continue
            if s_ms < 0 or e_ms <= s_ms:
                continue
            rows.append((sid, s_ms, e_ms))
    return rows


def read_reaction_shots(reaction_csv: Path) -> List[Tuple[int, int, int]]:
    """
    Parse reaction_shots.csv into the tuple shape used by the generic assembler.
    """
    if not reaction_csv.exists():
        return []
    rows: List[Tuple[int, int, int]] = []
    with reaction_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sid = int(row.get("shot_id", -1))
                s_ms = int(row.get("start_ms", -1))
                e_ms = int(row.get("end_ms", -1))
            except (TypeError, ValueError):
                continue
            if s_ms < 0 or e_ms <= s_ms:
                continue
            rows.append((sid, s_ms, e_ms))
    return rows


def assemble_replay_clip(
    ffmpeg_path: str,
    input_video: Path,
    replay_shots: List[Tuple[int, int, int]],
    output_video: Path,
    work_dir: Path,
    debug: bool = False,
) -> None:
    """
    Reuse the generic cut/concat path for replay-only clips.
    """
    if not replay_shots:
        return
    assemble_selected_clips(
        ffmpeg_path=ffmpeg_path,
        input_video=input_video,
        kept_shots=replay_shots,
        output_video=output_video,
        work_dir=work_dir,
        debug=debug,
    )
