from __future__ import annotations
"""Greedy time-budget planning for already assembled event clips."""

import subprocess
from pathlib import Path
from typing import Any, Dict, List

# Priority tiers for optional units (lower number means higher priority).
PRIORITY = {
    "goal_anchor": 1,
    "penalty_main": 2,
    "free_kick_main": 3,
    "red_card_main": 4,
    "red_card_reaction": 5,
    "goal_replay": 6,
    "goal_reaction": 7,
    "shot_main": 8,
}

INTRO_OUTRO_MIN_BUDGET_SECONDS = 121.0


def get_clip_duration_seconds(ffprobe_path: str, video_path: Path) -> float:
    """Read clip duration from the finished mp4 so budgeting uses real runtimes."""
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return float(out.strip())
    except Exception:
        return 0.0


def build_clip_plan(
    units: List[Dict[str, Any]],
    *,
    ffprobe_path: str,
    time_budget_seconds: float,
    debug: bool = False,
) -> List[Path]:
    """
    Greedy time-budgeting over already-assembled units.
    Each unit dict requires: type, event_id, order_idx, ts, path, intra_order.
    """
    # Filter existing files and add durations
    valid_units: List[Dict[str, Any]] = []
    for u in units:
        path = u.get("path")
        if not isinstance(path, Path) or not path.exists():
            if debug:
                print(f"[budget] skip missing {u.get('type')} {path}")
            continue
        dur = get_clip_duration_seconds(ffprobe_path, path)
        if dur <= 0:
            if debug:
                print(f"[budget] skip zero-duration {u.get('type')} {path}")
            continue
        uu = dict(u)
        uu["duration"] = dur
        valid_units.append(uu)

    if not valid_units:
        return []

    # Goal reactions/replays are only valid if their parent goal anchor exists.
    goal_anchor_ids = {u["event_id"] for u in valid_units if u.get("type") == "goal_anchor"}

    # Intro/outro are handled outside the normal priority list.
    guaranteed_extras: List[Dict[str, Any]] = []
    if time_budget_seconds >= INTRO_OUTRO_MIN_BUDGET_SECONDS:
        guaranteed_extras = [u for u in valid_units if u.get("type") in {"intro", "outro"}]

    # Goal anchors are mandatory and can push the total above the requested budget.
    selected: List[Dict[str, Any]] = [u for u in valid_units if u.get("type") == "goal_anchor"]
    selected.extend(guaranteed_extras)
    selected_paths = {u["path"] for u in selected}

    # Remaining budget after mandatories (can go negative)
    used = sum(u["duration"] for u in selected)
    remaining = time_budget_seconds - used

    # Everything else competes for the remaining budget.
    optional: List[Dict[str, Any]] = []
    for u in valid_units:
        t = u.get("type")
        if t in {"goal_anchor", "intro", "outro"}:
            continue
        if t in {"goal_replay", "goal_reaction"} and u.get("event_id") not in goal_anchor_ids:
            continue
        optional.append(u)

    # Sort optional by priority tier then chronological
    optional.sort(key=lambda u: (PRIORITY.get(u.get("type"), 99), u.get("order_idx", 1e9), u.get("ts", 1e9), u.get("intra_order", 0)))

    # Group optional units by priority and select alternately within each group.
    # Replays are intentionally uncapped here: they should be included whenever
    # budget remains after higher-priority candidates.
    from collections import defaultdict
    priority_groups = defaultdict(list)
    for u in optional:
        pri = PRIORITY.get(u.get("type"), 99)
        priority_groups[pri].append(u)

    for pri in sorted(priority_groups.keys()):
        group = priority_groups[pri]
        # Sort group by timestamp
        group.sort(key=lambda u: u.get("ts", 1e9))
        # Alternate selection: earliest, latest, second earliest, second latest, etc.
        selected_indices = []
        left, right = 0, len(group) - 1
        while left <= right:
            if left <= right:
                selected_indices.append(left)
                left += 1
            if left <= right:
                selected_indices.append(right)
                right -= 1
        for idx in selected_indices:
            u = group[idx]
            if u.get("path") in selected_paths:
                continue
            dur = u["duration"]
            if dur <= remaining:
                selected.append(u)
                selected_paths.add(u["path"])
                remaining -= dur
            else:
                if debug:
                    print(f"[budget] skip {u.get('type')} (dur={dur:.2f}s, remaining={remaining:.2f}s)")
                continue

    # Intro/outro should stay at the outer edges even in the budgeted path.
    selected.sort(
        key=lambda u: (
            0 if u.get("type") == "intro" else 2 if u.get("type") == "outro" else 1,
            u.get("order_idx", 1e9),
            u.get("ts", 1e9),
            u.get("intra_order", 0),
        )
    )
    return [u["path"] for u in selected]
