from __future__ import annotations
"""Rule-based anchor and continuity selection for each event clip."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Tunables used by the current rule-based selector.
MIN_SHOT_DURATION_MS = 1000
MAX_EVENT_DURATION_MS = 10_000
SHOT_EVENT_MAX_END_MS = 16_000
ALLOWED_ANCHOR_LABELS = {"main_camera_left", "main_camera_center", "main_camera_right"}
# Window and duration thresholds used by event-specific anchor heuristics.
MAIN_CAMERA_LABELS = {"main_camera_left", "main_camera_center", "main_camera_right"}
WINDOW_START_MS = 5000
WINDOW_END_MS = 11000
MIN_DURATION_MS = 3000
MIN_OVERLAP_MS = 2000

# Camera class priors for anchor scoring
CLASS_PRIORS: Dict[str, float] = {
    "main_camera_center": 1.00,
    "main_camera_left": 0.95,
    "main_camera_right": 0.95,
    "behind_the_goal": 0.80,
    "close_up_player_or_field_referee": 0.45,
    "public": 0.15,
    "close_up_side_staff": 0.10,
    "close_up_corner": 0.05,
}

def is_goal_event(video_name: str) -> bool:
    """Legacy helper retained for compatibility with earlier experimentation."""
    return "_goal" in video_name


def load_shots(shots_csv: Path) -> List[Dict[str, Any]]:
    """Load shot intervals and coerce numeric CSV columns to integers."""
    if not shots_csv.exists():
        print(f"[selection] missing shots file: {shots_csv}")
        return []
    with shots_csv.open("r", encoding="utf-8") as f:
        return [
            {
                **row,
                "shot_id": int(row["shot_id"]),
                "start_ms": int(row["start_ms"]),
                "end_ms": int(row["end_ms"]),
                "duration_ms": int(row["duration_ms"]),
            }
            for row in csv.DictReader(f)
        ]


def load_predictions(pred_csv: Path) -> Dict[int, Dict[str, Any]]:
    """Load classifier output rows keyed by shot id for fast lookup during selection."""
    if not pred_csv.exists():
        print(f"[selection] missing predictions file: {pred_csv}, treating all labels as unknown")
        return {}
    preds: Dict[int, Dict[str, Any]] = {}
    with pred_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                shot_id = int(row["shot_id"])
            except (KeyError, ValueError):
                continue
            preds[shot_id] = row
    return preds


def duration_score_ms(duration_ms: int) -> float:
    """Legacy scoring helper kept in the CSV output for analysis/debugging."""
    d = duration_ms
    if d < 800:
        return 0.0
    if 800 <= d < 1500:
        return min(1.0, (d - 800) / 700)
    if 1500 <= d <= 8000:
        return 1.0
    # Lightly penalize extremely long only beyond 12s
    if 8000 < d <= 12000:
        return max(0.0, (12000 - d) / 4000)
    return 0.0


def compute_overlap(start_ms: int, end_ms: int, window_start_ms: int, window_end_ms: int) -> int:
    """Measure how much of a shot overlaps a target decision window."""
    return max(0, min(end_ms, window_end_ms) - max(start_ms, window_start_ms))


def class_score(label: str) -> float:
    """Legacy scoring helper kept in the CSV output for analysis/debugging."""
    return CLASS_PRIORS.get(label, 0.0)


def compute_scores(shots: List[Dict[str, Any]], preds: Dict[int, Dict[str, Any]]) -> None:
    """Attach legacy score fields used for inspection, even though anchoring is rule-based."""
    if not shots:
        return
    total_duration = max(s["end_ms"] for s in shots)
    for sh in shots:
        pred = preds.get(sh["shot_id"], {})
        label = pred.get("label", "unknown")
        try:
            confidence = float(pred.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0

        midpoint_ms = (sh["start_ms"] + sh["end_ms"]) / 2.0
        midpoint_rel = midpoint_ms / total_duration if total_duration > 0 else 0.0

        cls_score = class_score(label)
        pos_score = early_position_score(midpoint_rel)
        dur_score = duration_score_ms(sh["duration_ms"])

        anchor_score = (
            0.60 * cls_score
            + 0.20 * dur_score
            + 0.15 * pos_score
            + 0.05 * confidence
        )

        sh.update(
            {
                "label": label,
                "confidence": confidence,
                "midpoint_ms": midpoint_ms,
                "midpoint_rel": midpoint_rel,
                "class_score": cls_score,
                "position_score": pos_score,
                "duration_score": dur_score,
                "anchor_score": anchor_score,
            }
        )


def select_first_shot(
    shots_csv: Path,
    out_csv: Path,
    *,
    debug: bool = False,
    event_name: str | None = None,
    video_path: Path | None = None,
) -> None:
    """
    Rule-based anchor selector with event-specific logic.
    """
    shots = load_shots(shots_csv)
    pred_csv = shots_csv.parent / "shot_predictions.csv"
    preds = load_predictions(pred_csv)

    if not shots:
        print(f"[selection] no shots found in {shots_csv}, writing empty selection file.")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "shot_id",
                    "start_ms",
                    "end_ms",
                    "duration_ms",
                    "label",
                    "confidence",
                    "keep",
                    "selected_role",
                    "reason",
                    "class_score",
                    "position_score",
                    "duration_score",
                    "anchor_score",
                ],
            )
            writer.writeheader()
        return

    # Merge classifier output into the raw shot list before applying event-specific rules.
    for sh in shots:
        pred = preds.get(sh["shot_id"], {})
        sh["label"] = pred.get("label", sh.get("label", "unknown"))
        try:
            sh["confidence"] = float(pred.get("confidence", sh.get("confidence", 0.0)))
        except (TypeError, ValueError):
            sh["confidence"] = 0.0

    event_label = event_name or shots_csv.parent.name
    stem = event_label.lower()
    is_goal = stem.endswith("_goal")
    is_shot = stem.endswith("_shot")
    is_penalty = stem.endswith("_penalty")
    is_red = stem.endswith("_red_card") or stem.endswith("_redcard")
    is_free_kick = stem.endswith("_free_kick")
    is_intro = stem.endswith("_start_phase")
    is_outro = stem.endswith("_end_of_game")
    # start_phase bypasses shot-based anchor selection and always uses a fixed intro window.
    if is_intro:
        rows = [
            {
                "shot_id": 0,
                "start_ms": 0,
                "end_ms": 15000,
                "duration_ms": 15000,
                "label": "start_phase",
                "confidence": 0.0,
                "keep": 1,
                "selected_role": "anchor",
                "reason": "fixed intro clip",
                "class_score": 0.0,
                "position_score": 0.0,
                "duration_score": 0.0,
                "anchor_score": 0.0,
            }
        ]
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "shot_id",
                    "start_ms",
                    "end_ms",
                    "duration_ms",
                    "label",
                    "confidence",
                    "keep",
                    "selected_role",
                    "reason",
                    "class_score",
                    "position_score",
                    "duration_score",
                    "anchor_score",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        return

    # end_of_game bypasses shot-based anchor selection and keeps the full event clip.
    if is_outro:
        min_start = min(sh["start_ms"] for sh in shots) if shots else 0
        max_end = max(sh["end_ms"] for sh in shots) if shots else 0
        rows = [
            {
                "shot_id": shots[0]["shot_id"] if shots else 0,
                "start_ms": min_start,
                "end_ms": 20000,
                "duration_ms": 20000,
                "label": "end_of_game",
                "confidence": 0.0,
                "keep": 1,
                "selected_role": "anchor",
                "reason": "full outro clip",
                "class_score": 0.0,
                "position_score": 0.0,
                "duration_score": 0.0,
                "anchor_score": 0.0,
            }
        ]
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "shot_id",
                    "start_ms",
                    "end_ms",
                    "duration_ms",
                    "label",
                    "confidence",
                    "keep",
                    "selected_role",
                    "reason",
                    "class_score",
                    "position_score",
                    "duration_score",
                    "anchor_score",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        return

    short_ids = [sh["shot_id"] for sh in shots if sh["duration_ms"] < MIN_SHOT_DURATION_MS]

    def select_anchor(event_type: str) -> Optional[int]:
        """Choose one anchor index using event-type-specific rule sets."""
        candidates: List[Tuple[int, int, int, int]] = []  # idx, overlap, start, duration
        if event_type in {"goal", "shot", "penalty", "free_kick"}:
            # Prefer valid main-camera gameplay shots overlapping the decisive-action window.
            for idx, sh in enumerate(shots):
                if sh.get("label") not in MAIN_CAMERA_LABELS:
                    continue
                if sh["duration_ms"] < MIN_DURATION_MS:
                    continue
                overlap = compute_overlap(sh["start_ms"], sh["end_ms"], WINDOW_START_MS, WINDOW_END_MS)
                if overlap >= MIN_OVERLAP_MS:
                    candidates.append((idx, overlap, sh["start_ms"], sh["duration_ms"]))
            if not candidates:
                # Relax the duration threshold before falling back to the first main-camera shot.
                for idx, sh in enumerate(shots):
                    if sh.get("label") not in MAIN_CAMERA_LABELS:
                        continue
                    if sh["duration_ms"] < 2000:
                        continue
                    overlap = compute_overlap(sh["start_ms"], sh["end_ms"], WINDOW_START_MS, WINDOW_END_MS)
                    candidates.append((idx, overlap, sh["start_ms"], sh["duration_ms"]))
            if candidates:
                candidates.sort(key=lambda t: (-t[1], t[2], -t[3]))
                return candidates[0][0]
            for idx, sh in enumerate(shots):
                if sh.get("label") in MAIN_CAMERA_LABELS:
                    return idx
            return None
        if event_type in {"red_card", "redcard"}:
            # Red cards allow any non-public shot, still centered on the same time window.
            for idx, sh in enumerate(shots):
                if sh.get("label") == "public":
                    continue
                if sh["duration_ms"] < MIN_DURATION_MS:
                    continue
                overlap = compute_overlap(sh["start_ms"], sh["end_ms"], WINDOW_START_MS, WINDOW_END_MS)
                if overlap >= MIN_OVERLAP_MS:
                    candidates.append((idx, overlap, sh["start_ms"], sh["duration_ms"]))
            if not candidates:
                # Final fallback: any non-public shot, even if overlap is weak.
                for idx, sh in enumerate(shots):
                    if sh.get("label") == "public":
                        continue
                    candidates.append((idx, compute_overlap(sh["start_ms"], sh["end_ms"], WINDOW_START_MS, WINDOW_END_MS), sh["start_ms"], sh["duration_ms"]))
            if candidates:
                candidates.sort(key=lambda t: (-t[1], t[2], -t[3]))
                return candidates[0][0]
            return None
        return None

    selection_type = (
        "goal"
        if is_goal
        else ("shot" if is_shot else ("penalty" if is_penalty else ("free_kick" if is_free_kick else ("red_card" if is_red else ""))))
    )

    anchor_idx = select_anchor(selection_type)

    kept_indices: List[int] = []
    if anchor_idx is not None:
        kept_indices.append(anchor_idx)

    initial_selected = sorted(set(kept_indices))

    # Continuity fill: include all valid shots between earliest and latest selected shot indices.
    continuity_added: set[int] = set()
    if initial_selected:
        span_start = min(initial_selected)
        span_end = max(initial_selected)
        for idx in range(span_start, span_end + 1):
            if idx in initial_selected:
                continue
            if shots[idx]["duration_ms"] >= MIN_SHOT_DURATION_MS:
                kept_indices.append(idx)
                continuity_added.add(idx)

    # Apply a hard per-event duration cap while keeping the kept block contiguous around the anchor.
    over_budget_removed: set[int] = set()
    if kept_indices:
        block = sorted(set(kept_indices))
        total_duration = sum(shots[i]["duration_ms"] for i in block)
        if total_duration > MAX_EVENT_DURATION_MS and anchor_idx is not None:
            while total_duration > MAX_EVENT_DURATION_MS and len(block) > 1:
                left_dist = anchor_idx - block[0]
                right_dist = block[-1] - anchor_idx
                if right_dist > left_dist:
                    remove_idx = block.pop()
                else:
                    remove_idx = block.pop(0)
                if remove_idx == anchor_idx:
                    block.insert(0, remove_idx)
                    break
                over_budget_removed.add(remove_idx)
                total_duration = sum(shots[i]["duration_ms"] for i in block)
        kept_indices = block

    if debug:
        event_label = event_name or shots_csv.parent.name
        print(f"[selection] event={event_label} type={selection_type or 'other'} anchor_idx={anchor_idx}")
        # Dump all shots with overlap values so mis-selections are easier to diagnose offline.
        cand_lines = []
        for idx, sh in enumerate(shots):
            overlap = compute_overlap(sh["start_ms"], sh["end_ms"], WINDOW_START_MS, WINDOW_END_MS)
            cand_lines.append(f"id={sh['shot_id']} label={sh.get('label')} dur={sh['duration_ms']} overlap={overlap}")
        print(f"[selection] candidates: {' | '.join(cand_lines)}")
        kept_ids = [shots[i]["shot_id"] for i in kept_indices]
        initial_ids = [shots[i]["shot_id"] for i in initial_selected]
        print(f"[selection] kept={kept_ids} initial={initial_ids}")
        if short_ids:
            print(f"[selection] rejected too short: {short_ids}")

    # Write the selector decision back out for later inspection and assembly.
    rows = []
    for idx, sh in enumerate(shots):
        row_start_ms = sh["start_ms"]
        row_end_ms = sh["end_ms"]
        row_duration_ms = sh["duration_ms"]
        role = "none"
        reason = "not_selected"
        keep = 0

        if sh["duration_ms"] < MIN_SHOT_DURATION_MS:
            reason = "rejected: too short"
        elif idx in kept_indices:
            keep = 1
            if idx == anchor_idx:
                role = "anchor"
                reason = "selected as anchor"
                if is_shot and row_end_ms > SHOT_EVENT_MAX_END_MS:
                    # Long shot-event anchors are trimmed so they do not dominate the final reel.
                    row_end_ms = SHOT_EVENT_MAX_END_MS
                    row_duration_ms = max(0, row_end_ms - row_start_ms)
            elif idx in continuity_added:
                role = "continuity"
                reason = "selected for continuity"
            else:
                reason = "selected for continuity"
        else:
            if idx in over_budget_removed:
                reason = "rejected: over budget"
            else:
                reason = "rejected: not selected"

        rows.append(
            {
                "shot_id": sh["shot_id"],
                "start_ms": row_start_ms,
                "end_ms": row_end_ms,
                "duration_ms": row_duration_ms,
                "label": sh.get("label", "unknown"),
                "confidence": sh.get("confidence", 0.0),
                "keep": keep,
                "selected_role": role,
                "reason": reason,
                "class_score": sh.get("class_score", 0.0),
                "position_score": sh.get("position_score", 0.0),
                "duration_score": sh.get("duration_score", 0.0),
                "anchor_score": sh.get("anchor_score", 0.0),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "shot_id",
                "start_ms",
                "end_ms",
                "duration_ms",
                "label",
                "confidence",
                "keep",
                "selected_role",
                "reason",
                "class_score",
                "position_score",
                "duration_score",
                "anchor_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[selection] anchor={shots[anchor_idx]['shot_id'] if anchor_idx is not None else 'None'}")
