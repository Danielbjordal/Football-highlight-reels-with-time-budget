"""
Download and filter football event clips from the Eliteserien highlights API.
Hardcoded for a small set of game IDs, easy to extend later.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

FFMPEG_PATH = r"C:/Users/Daniel/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0.1-full_build/bin/ffmpeg.exe"
GAME_IDS = [4400]
FILTERED_TYPES = {"goal", "free_kick", "red_card", "penalty", "start_phase", "end_of_game", "shot"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_events(game_id: int) -> List[Dict[str, Any]]:
    url = f"https://api.highlights.eliteserien.no/eliteserien/game/{game_id}/events?count=999"
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
    # Prefer explicit fields
    for key in ("event_type", "type", "name"):
        if key in event and isinstance(event[key], str):
            return event[key]
    # Tag dict with action
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
    ffmpeg_cmd = FFMPEG_PATH if Path(FFMPEG_PATH).exists() else shutil.which("ffmpeg")
    if not ffmpeg_cmd:
        print("  ffmpeg not found; please install or update FFMPEG_PATH")
        return False
    cmd = [ffmpeg_cmd, "-y", "-i", hls_url, "-c", "copy", str(output_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  Downloaded {output_path.name}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  Failed to download {output_path.name}: {exc.stderr.decode('utf-8', errors='ignore')}")
        return False
    except FileNotFoundError:
        print("  ffmpeg executable not found; update FFMPEG_PATH")
        return False


def process_game(game_id: int) -> None:
    game_dir = Path("data") / "games" / str(game_id)
    # Save clips directly in game folder to match pipeline expectations
    clips_dir = game_dir

    # Ensure target directories exist before writing json/mp4/csv
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

        # If goal, extend HLS end_ms by +60000 while keeping id/start_ms intact
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

        # Use a stable event id in filenames so pipeline.parse_event_filename works.
        raw_event_id = ev.get("id")
        event_id = str(raw_event_id if raw_event_id is not None else f"{game_id}-{idx}")
        # Ensure we don't split the id when the pipeline splits on the first underscore.
        event_id = event_id.replace("_", "-")

        out_path = clips_dir / f"{event_id}_{etype}.mp4"
        # collect metadata regardless of download success
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

    # write metadata
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
