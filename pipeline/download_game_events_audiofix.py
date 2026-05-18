"""
Download and filter football event clips from the Eliteserien highlights API.
This variant keeps the current downloader behavior, but adds an audio-recovery
fallback for clips where direct HLS download produces a video-only MP4.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests

FFMPEG_PATH = "ffmpeg"
API_BASE_ENV = "ELITESERIEN_API_BASE"
API_BASE_PLACEHOLDER = "ELITESERIEN_API_BASE_PLACEHOLDER"
GAME_IDS = [4407]
FILTERED_TYPES = {"goal", "red_card", "penalty", "start_phase", "end_of_game", "shot", "free_kick"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_api_base() -> str:
    api_base = os.environ.get(API_BASE_ENV, API_BASE_PLACEHOLDER).strip().rstrip("/")
    if api_base == API_BASE_PLACEHOLDER:
        raise ValueError(f"Set {API_BASE_ENV} to the Eliteserien API base URL before downloading events.")
    return api_base


def get_ffmpeg_cmd() -> str | None:
    return FFMPEG_PATH if Path(FFMPEG_PATH).exists() else shutil.which("ffmpeg")


def get_ffprobe_cmd(ffmpeg_cmd: str) -> str:
    ffmpeg_path = Path(ffmpeg_cmd)
    if ffmpeg_path.exists():
        candidate = ffmpeg_path.parent / "ffprobe.exe"
        if candidate.exists():
            return str(candidate)
    ffprobe_cmd = shutil.which("ffprobe")
    if not ffprobe_cmd:
        raise FileNotFoundError("ffprobe not found; install FFmpeg or update FFMPEG_PATH")
    return ffprobe_cmd


def fetch_text(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def has_audio_stream(video_path: Path, ffprobe_cmd: str) -> bool:
    cmd = [
        ffprobe_cmd,
        "-v",
        "error",
        "-show_streams",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams", [])
    return any(isinstance(s, dict) and s.get("codec_type") == "audio" for s in streams)


def parse_stream_inf(line: str) -> Dict[str, str]:
    """Parse one EXT-X-STREAM-INF attribute line into a simple dict."""
    attrs: Dict[str, str] = {}
    _, _, raw_attrs = line.partition(":")
    for part in raw_attrs.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        attrs[key.strip().upper()] = value.strip().strip('"')
    return attrs


def parse_resolution(value: str) -> int:
    """Convert RESOLUTION=WxH into total pixel count for easy ranking."""
    try:
        width_str, height_str = value.lower().split("x", 1)
        return int(width_str) * int(height_str)
    except Exception:
        return 0


def extract_variant_playlist_url(master_url: str) -> str:
    text = fetch_text(master_url)

    variants: List[tuple[int, int, str]] = []
    pending_attrs: Dict[str, str] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#EXT-X-STREAM-INF:"):
            pending_attrs = parse_stream_inf(line)
            continue
        if line.startswith("#"):
            continue

        if pending_attrs is not None:
            bandwidth = int(pending_attrs.get("BANDWIDTH", "0") or "0")
            resolution_score = parse_resolution(pending_attrs.get("RESOLUTION", ""))
            variants.append((bandwidth, resolution_score, urljoin(master_url, line)))
            pending_attrs = None

    if not variants:
        raise ValueError(f"No variant playlist found in master manifest: {master_url}")

    # Prefer the highest-bandwidth stream and use resolution as a tie-breaker.
    variants.sort(key=lambda item: (item[0], item[1]))
    return variants[-1][2]


def extract_segment_urls(variant_url: str) -> List[str]:
    text = fetch_text(variant_url)
    segments: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ".ts" in line:
            segments.append(urljoin(variant_url, line))
    if not segments:
        raise ValueError(f"No TS segments found in variant manifest: {variant_url}")
    return segments


def download_clip_direct(hls_url: str, output_path: Path, ffmpeg_cmd: str) -> None:
    cmd = [ffmpeg_cmd, "-y", "-i", hls_url, "-c", "copy", str(output_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def download_clip_via_segments(hls_url: str, output_path: Path, ffmpeg_cmd: str) -> None:
    variant_url = extract_variant_playlist_url(hls_url)
    segment_urls = extract_segment_urls(variant_url)

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        concat_list = tmpdir / "files.txt"
        transport_stream = tmpdir / "output.ts"

        with concat_list.open("w", encoding="utf-8") as f:
            for url in segment_urls:
                f.write(f"file '{url}'\n")

        concat_cmd = [
            ffmpeg_cmd,
            "-y",
            "-protocol_whitelist",
            "file,http,https,tcp,tls,crypto",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(transport_stream),
        ]
        subprocess.run(concat_cmd, check=True, capture_output=True)

        remux_cmd = [
            ffmpeg_cmd,
            "-y",
            "-i",
            str(transport_stream),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(remux_cmd, check=True, capture_output=True)


def download_events(game_id: int) -> List[Dict[str, Any]]:
    url = f"{get_api_base()}/game/{game_id}/events?count=999"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("events", "results", "items", "data"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
    raise ValueError(f"Unexpected events payload for game {game_id}: {type(payload)}")


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def extract_event_type(event: Dict[str, Any]) -> Optional[str]:
    for key in ("event_type", "type", "name"):
        if key in event and isinstance(event[key], str):
            return event[key]
    tag = event.get("tag")
    if isinstance(tag, dict):
        action = tag.get("action")
        if isinstance(action, str):
            return action
    return None


def normalize_type(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value.strip().lower().replace(" ", "_")


def filter_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = []
    for ev in events:
        etype = normalize_type(extract_event_type(ev))
        if etype and etype in FILTERED_TYPES:
            ev = ev.copy()
            ev["_event_type"] = etype
            filtered.append(ev)
    return filtered


def find_video_url(event: Dict[str, Any]) -> Optional[str]:
    candidates = ("video_url", "video", "url", "playback", "stream", "hls")
    for key in candidates:
        val = event.get(key)
        if isinstance(val, str) and val.startswith("http"):
            return val
    playlist = event.get("playlist")
    if isinstance(playlist, dict):
        for key in candidates:
            val = playlist.get(key)
            if isinstance(val, str) and val.startswith("http"):
                return val
    return None


def download_clip(hls_url: str, output_path: Path) -> bool:
    if output_path.exists():
        print(f"  Skipping existing clip: {output_path.name}")
        return True

    ffmpeg_cmd = get_ffmpeg_cmd()
    if not ffmpeg_cmd:
        print("  ffmpeg not found; please install or update FFMPEG_PATH")
        return False

    try:
        ffprobe_cmd = get_ffprobe_cmd(ffmpeg_cmd)
    except FileNotFoundError as exc:
        print(f"  {exc}")
        return False

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        direct_output = tmpdir / "direct.mp4"
        fallback_output = tmpdir / "fallback.mp4"

        try:
            download_clip_direct(hls_url, direct_output, ffmpeg_cmd)
            if has_audio_stream(direct_output, ffprobe_cmd):
                shutil.move(str(direct_output), str(output_path))
                print(f"  Downloaded {output_path.name}")
                return True

            print(f"  Direct download missing audio for {output_path.name}, trying segment fallback")
            download_clip_via_segments(hls_url, fallback_output, ffmpeg_cmd)

            if has_audio_stream(fallback_output, ffprobe_cmd):
                shutil.move(str(fallback_output), str(output_path))
                print(f"  Downloaded {output_path.name} via audio-recovery fallback")
                return True

            shutil.move(str(direct_output), str(output_path))
            print(f"  Warning: {output_path.name} still has no audio after fallback")
            return True
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr)
            print(f"  Failed to download {output_path.name}: {stderr}")
            return False
        except Exception as exc:
            print(f"  Failed to download {output_path.name}: {exc}")
            return False


def process_game(game_id: int) -> None:
    game_dir = Path("data") / "games" / str(game_id)
    clips_dir = game_dir

    ensure_dir(game_dir)
    ensure_dir(clips_dir)

    print(f"\nGame {game_id}:")
    events_path = game_dir / "events.json"
    if events_path.exists():
        events = json.loads(events_path.read_text(encoding="utf-8"))
        print("  Loaded cached events.json")
    else:
        try:
            events = download_events(game_id)
            save_json(events_path, events)
        except Exception as exc:
            print(f"  Failed to download events: {exc}")
            return

    unique_types = sorted({extract_event_type(ev) for ev in events if extract_event_type(ev)})
    print(f"  Unique event types: {unique_types}")
    print(f"  Total events: {len(events)}")

    filtered = filter_events(events)
    save_json(game_dir / "target_events.json", filtered)
    print(f"  Filtered events: {len(filtered)}")

    downloaded = 0
    metadata_rows = []
    for idx, ev in enumerate(filtered):
        url = find_video_url(ev)
        if not url:
            print(f"  No video URL for filtered event {idx} ({ev.get('_event_type')})")
        etype = ev.get("_event_type", "event")

        if etype == "goal" and isinstance(url, str):
            try:
                parts = url.split("/")
                if len(parts) >= 2 and parts[-1].lower().startswith("manifest"):
                    id_start_end = parts[-2]
                    segments = id_start_end.split(":")
                    if len(segments) == 3:
                        clip_id, start_ms, end_ms = segments
                        new_end = str(int(end_ms) + 60000)
                        parts[-2] = ":".join([clip_id, start_ms, new_end])
                        url = "/".join(parts)
            except Exception:
                pass

        raw_event_id = ev.get("id")
        event_id = str(raw_event_id if raw_event_id is not None else f"{game_id}-{idx}")
        event_id = event_id.replace("_", "-")

        out_path = clips_dir / f"{event_id}_{etype}.mp4"
        playlist_events = ev.get("playlist", {}).get("events") if isinstance(ev.get("playlist"), dict) else None
        if isinstance(playlist_events, list) and playlist_events:
            first_ev = playlist_events[0] if isinstance(playlist_events[0], dict) else {}
            from_ts = first_ev.get("from_timestamp")
            to_ts = first_ev.get("to_timestamp")
        else:
            from_ts = ev.get("from_timestamp")
            to_ts = ev.get("to_timestamp")
        metadata_rows.append(
            {
                "event_id": event_id,
                "filename": out_path.name,
                "tag": etype,
                "from_timestamp": from_ts if from_ts is not None else "",
                "to_timestamp": to_ts if to_ts is not None else "",
                "game_id": game_id,
            }
        )

        if not url:
            continue
        if download_clip(url, out_path):
            downloaded += 1

    print(f"  Downloaded clips: {downloaded}")

    meta_path = game_dir / "events_metadata.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["event_id", "filename", "tag", "from_timestamp", "to_timestamp", "game_id"]
        )
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"  Wrote metadata: {meta_path}")


def main() -> None:
    for gid in GAME_IDS:
        process_game(gid)


if __name__ == "__main__":
    main()
