from __future__ import annotations
"""Reaction-shot selection and assembly helpers."""

import csv
from pathlib import Path
from typing import Any, List

from pipeline.assembly import assemble_replay_clip

REACTION_LABELS = {"public"}


def write_reaction_csv(rows: List[dict[str, Any]], out_csv: Path) -> None:
    """Persist the reaction rows selected for one event."""
    fieldnames = ["shot_id", "start_ms", "end_ms", "duration_ms", "label", "reaction"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_reaction_shots(
    selected_shots_csv: Path,
    shot_predictions_csv: Path,
    event_tag: str = "goal",
) -> List[dict[str, Any]]:
    """
    Return reaction shot rows after the anchor.
    - goal: first close_up_player_or_field_referee, first public
    - red_card/redcard: first shot after the anchor
    """
    if not selected_shots_csv.exists() or not shot_predictions_csv.exists():
        return []
    anchor_end_ms: int | None = None
    with selected_shots_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = (row.get("selected_role") or "").lower()
            try:
                keep = int(row.get("keep", 0))
            except (TypeError, ValueError):
                keep = 0
            if role == "anchor" and keep == 1:
                try:
                    anchor_end_ms = int(row.get("end_ms", -1))
                except (TypeError, ValueError):
                    anchor_end_ms = None
                break
    if anchor_end_ms is None:
        return []

    found: dict[str, dict[str, Any]] = {}
    with shot_predictions_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = sorted(reader, key=lambda r: int(r.get("start_ms", 0)))

    normalized_tag = event_tag.lower()
    if normalized_tag in {"red_card", "redcard"}:
        # Red cards currently use the very first post-anchor shot as their reaction clip.
        for row in rows:
            try:
                start_ms = int(row["start_ms"])
                end_ms = int(row["end_ms"])
            except (KeyError, ValueError, TypeError):
                continue
            if start_ms <= anchor_end_ms:
                continue
            return [
                {
                    "shot_id": row.get("shot_id", ""),
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": row.get("duration_ms", ""),
                    "label": (row.get("label") or "").lower(),
                    "reaction": 1,
                }
            ]
        return []

    for row in rows:
        label = (row.get("label") or "").lower()
        if label not in REACTION_LABELS:
            continue
        try:
            start_ms = int(row["start_ms"])
            end_ms = int(row["end_ms"])
        except (KeyError, ValueError, TypeError):
            continue
        if start_ms <= anchor_end_ms:
            continue
        if label in found:
            continue
        found[label] = {
            "shot_id": row.get("shot_id", ""),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": row.get("duration_ms", ""),
            "label": label,
            "reaction": 1,
        }
        if len(found) == 1:
            break

    return sorted(found.values(), key=lambda r: r["start_ms"])


def assemble_reaction_clip(
    ffmpeg_path: str,
    input_video: Path,
    reaction_rows: List[dict[str, Any]],
    output_video: Path,
    work_dir: Path,
    debug: bool = False,
) -> None:
    """
    Assemble reaction shots into a single clip using the same cut/concat path as replays.
    """
    if not reaction_rows:
        return
    kept = [(int(r["shot_id"]), int(r["start_ms"]), int(r["end_ms"])) for r in reaction_rows]
    assemble_replay_clip(
        ffmpeg_path=ffmpeg_path,
        input_video=input_video,
        replay_shots=kept,
        output_video=output_video,
        work_dir=work_dir,
        debug=debug,
    )
