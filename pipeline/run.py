from __future__ import annotations
"""Main CLI entrypoint that orchestrates the end-to-end highlight pipeline."""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict
import json
import os
import time

import yaml
from pipeline.constants import ALLOWED_TAGS, ORDER_FALLBACK, REACTION_ORDER_OFFSET, REPLAY_ORDER_OFFSET
from pipeline.performance_monitor import (
    BenchmarkSession,
    env_flag,
    make_run_id,
    probe_video_stats,
    resolve_device_label,
)
from pipeline.profiling import (
    NoOpProfiler,
    PipelineProfiler,
    env_flag as profiling_env_flag,
    set_active_profiler,
)
from pipeline.replay import extract_replay_shots
from pipeline.reaction import select_reaction_shots, write_reaction_csv, assemble_reaction_clip
from pipeline.time_budgeting import build_clip_plan

from pipeline.classify_event import classify_event_shots
from pipeline.sbd import run_sbd
from pipeline.selection import select_first_shot
from pipeline.assembly import (
    assemble_selected_clips,
    assemble_replay_clip,
    crossfade_clips,
    read_selected_shots,
    read_replay_shots,
    read_reaction_shots,
    update_intro_candidate,
    update_outro_candidate,
)
from pipeline.tooling import ffprobe_from_ffmpeg


def parse_event_filename(video_path: Path) -> tuple[str, str]:
    """
    Example:
    a1w29ambahuu2_goal.mp4
    -> ("a1w29ambahuu2", "goal")
    """
    stem = video_path.stem
    parts = stem.split("_", 1)

    if len(parts) != 2:
        raise ValueError(f"Unexpected filename format: {video_path.name}")

    event_id, event_tag = parts
    return event_id, event_tag


def debug_print(enabled: bool, message: str) -> None:
    """Small helper so optional debug logging stays centralized."""
    if enabled:
        print(message)


def format_elapsed(seconds: float) -> str:
    """Format stage timings consistently for console output."""
    return f"{seconds:.2f}s"


def event_unit_metadata(event_tag: str) -> tuple[str, int]:
    """
    Maps event tags to unit type and intra-order for assembly/budgeting.
    """
    if event_tag == "goal":
        return "goal_anchor", 0
    if event_tag == "penalty":
        return "penalty_main", 0
    if event_tag in {"red_card", "redcard"}:
        return "red_card_main", 0
    if event_tag == "free_kick":
        return "free_kick_main", 0
    if event_tag == "shot":
        return "shot_main", 0
    return "other", 0


def load_events_metadata(metadata_path: Path) -> dict[str, dict[str, Any]]:
    """Load API timestamps used later for event ordering in the final reel."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}. Run download_events to generate it.")

    metadata: dict[str, dict[str, Any]] = {}
    with metadata_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_id = row.get("event_id")
            if not event_id:
                continue
            try:
                from_ts = int(row.get("from_timestamp", ""))
            except (TypeError, ValueError):
                from_ts = None
            metadata[event_id] = {"from_timestamp": from_ts}
    return metadata


def build_final_manifest_entries(
    clip_paths: list[Path],
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build ordered manifest entries for the final highlight clip list."""
    unit_by_path = {u["path"]: u for u in units}
    entries: list[dict[str, Any]] = []
    for idx, clip_path in enumerate(clip_paths):
        unit = unit_by_path.get(clip_path, {})
        entries.append(
            {
                "clip_index": idx,
                "clip_path": str(clip_path),
                "event_id": unit.get("event_id"),
                "clip_type": unit.get("type"),
            }
        )
    return entries


def write_final_assembly_manifest(
    work_dir: Path,
    clip_paths: list[Path],
    units: list[dict[str, Any]],
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_txt = work_dir / "final_manifest.txt"
    manifest_json = work_dir / "final_manifest.json"
    with manifest_txt.open("w", encoding="utf-8") as f:
        for clip_path in clip_paths:
            f.write(f"{clip_path}\n")
    entries = build_final_manifest_entries(clip_paths, units)
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def check_selection_schema(selected_shots_csv: Path) -> None:
    """Warn when cached selection files predate the current output schema."""
    required_cols = {
        "selected_role",
        "class_score",
        "position_score",
        "duration_score",
        "anchor_score",
    }
    if not selected_shots_csv.exists():
        return
    with selected_shots_csv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print(f"[selection] warning: {selected_shots_csv} is empty.")
            return
    missing = required_cols.difference(header)
    if missing:
        print(f"[selection] warning: {selected_shots_csv} missing new columns {sorted(missing)}; likely generated by old selector. Use --force-selection.")
    else:
        print(f"[selection] using selection file with new schema: {selected_shots_csv}")


def log_selection_summary(selected_shots_csv: Path, debug: bool) -> None:
    """Print a concise debug summary of the selector output for one event."""
    if not debug or not selected_shots_csv.exists():
        return
    kept_ids: list[int] = []
    with selected_shots_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for row in reader:
            try:
                if int(row.get("keep", 0)) == 1:
                    kept_ids.append(int(row.get("shot_id", -1)))
            except (ValueError, TypeError):
                continue
    print(f"  Debug: selection file {selected_shots_csv.name} keep_count={len(kept_ids)} keep_ids={kept_ids}")
    required = {"selected_role", "class_score", "position_score", "duration_score", "anchor_score"}
    if required.issubset(set(header)):
        print("  Debug: selection file has new schema columns")
    else:
        print("  Debug: selection file missing new schema columns; consider --force-selection")


def load_config(path: Path) -> Dict[str, Any]:
    """Load the YAML config file used to parameterize the pipeline run."""
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_event_order(game_dir: Path) -> dict[str, int]:
    """
    Returns mapping event_id -> index based on target_events.json if present.
    """
    target = game_dir / "target_events.json"
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    order: dict[str, int] = {}
    if isinstance(data, list):
        for idx, ev in enumerate(data):
            if isinstance(ev, dict):
                ev_id = ev.get("id")
                if isinstance(ev_id, str):
                    order[ev_id] = idx
    return order


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the main pipeline CLI."""
    parser = argparse.ArgumentParser(description="Run video highlight pipeline.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config YAML.")
    parser.add_argument("--game-id", type=str, help="Game ID to process (overrides config).")
    parser.add_argument("--ffmpeg", type=str, help="Path to ffmpeg binary (overrides config).")
    parser.add_argument("--weights", type=str, help="Path to classifier weights (overrides config).")
    parser.add_argument("--device", type=str, help="Device for classifier, e.g. cpu or cuda.")
    parser.add_argument("--force-selection", action="store_true", help="Regenerate selection and selected clips.")
    parser.add_argument("--force-assembly", action="store_true", help="Regenerate per-event selected clips and final highlight.")
    parser.add_argument("--debug-selection", action="store_true", help="Print selection/assembly debug info.")
    parser.add_argument("--time_budget", type=float, help="Highlight time budget in seconds.")
    parser.add_argument("--benchmark", action="store_true", help="Enable component-level benchmarking and CSV logging.")
    parser.add_argument("--benchmark-setup-name", type=str, help="Friendly hardware/setup name stored in the benchmark CSV.")
    parser.add_argument("--benchmark-csv", type=Path, help="Path to the benchmark CSV log.")
    parser.add_argument("--benchmark-sample-interval", type=float, help="Sampling interval in seconds for CPU/RAM/GPU monitoring.")
    parser.add_argument("--profile", action="store_true", help="Enable lightweight pipeline activity profiling.")
    parser.add_argument("--profile-output", type=Path, help="Path to the pipeline profiling CSV.")
    return parser


def main() -> None:
    """Run all configured stages for one game and optionally assemble a final highlight reel."""
    pipeline_started = time.perf_counter()
    args = build_arg_parser().parse_args()

    cfg = load_config(Path(args.config))
    defaults = cfg.get("defaults", {})
    paths_cfg = cfg.get("paths", {})
    sbd_cfg = cfg.get("sbd", {})
    classifier_cfg = cfg.get("classifier", {})
    tools_cfg = cfg.get("tools", {})
    benchmark_cfg = cfg.get("benchmark", {})
    profiling_cfg = cfg.get("profiling", {})

    game_id = args.game_id or defaults.get("game_id") or "4418"
    ffmpeg_path = args.ffmpeg or tools_cfg.get("ffmpeg_path") or "ffmpeg"
    ffprobe_path = ffprobe_from_ffmpeg(ffmpeg_path)
    weights_path = args.weights or defaults.get("weights_path") or "weights/resnet50_forzasys_soccer_camera_zoom_v2.pth"
    device = args.device or classifier_cfg.get("device") or defaults.get("device") or "auto"
    sbd_threshold = sbd_cfg.get("threshold", 0.7)
    sbd_fps = sbd_cfg.get("fps", 25)
    num_samples = classifier_cfg.get("num_samples", 3)
    classifier_batch_size = int(classifier_cfg.get("batch_size", 16))
    force_selection = bool(args.force_selection)
    force_assembly = bool(args.force_assembly)
    debug_selection = bool(args.debug_selection)
    time_budget = args.time_budget
    benchmark_enabled = bool(
        args.benchmark
        or env_flag("ENABLE_BENCHMARK", default=bool(benchmark_cfg.get("enabled", False)))
    )
    benchmark_setup_name = (
        args.benchmark_setup_name
        or os.environ.get("BENCHMARK_SETUP_NAME")
        or benchmark_cfg.get("setup_name")
        or os.environ.get("COMPUTERNAME")
        or "default_setup"
    )
    benchmark_csv_path = Path(
        args.benchmark_csv
        or os.environ.get("BENCHMARK_CSV_PATH")
        or benchmark_cfg.get("output_csv", "benchmark_results/performance_log.csv")
    )
    benchmark_sample_interval = float(
        args.benchmark_sample_interval
        or os.environ.get("BENCHMARK_SAMPLE_INTERVAL")
        or benchmark_cfg.get("sampling_interval_s", 0.5)
    )
    resolved_device = resolve_device_label(device)
    benchmark_session = BenchmarkSession(
        enabled=benchmark_enabled,
        setup_name=benchmark_setup_name,
        run_id=make_run_id(),
        game_id=str(game_id),
        output_csv_path=benchmark_csv_path,
        sample_interval_s=benchmark_sample_interval,
    )
    profiling_enabled = bool(
        args.profile
        or profiling_env_flag("PIPELINE_PROFILE", default=bool(profiling_cfg.get("enabled", False)))
    )
    profiling_output_path = Path(
        args.profile_output
        or os.environ.get("PIPELINE_PROFILE_OUTPUT")
        or profiling_cfg.get("output_csv", "outputs/profiling_run.csv")
    )
    profiler = PipelineProfiler(profiling_output_path) if profiling_enabled else NoOpProfiler()
    set_active_profiler(profiler)

    game_dir = Path("data/games") / game_id
    output_root = Path(paths_cfg.get("output_dir", "outputs")) / game_id

    if not game_dir.exists():
        raise FileNotFoundError(f"Missing game folder: {game_dir}")

    print(f"Pipeline device setting: {device}")

    try:
        with profiler.stage("pipeline_total", detail=f"game_id={game_id}"):
            metadata_path = game_dir / "events_metadata.csv"
            events_metadata = load_events_metadata(metadata_path)
            event_order = load_event_order(game_dir)

            videos = sorted(game_dir.glob("*.mp4"))
            full_pipeline_clip_count = len(videos) if videos else 0

            with benchmark_session.monitor(
                "FULL_PIPELINE",
                device=resolved_device,
                clip_count=full_pipeline_clip_count,
            ):
                if not videos:
                    print("No event clips found.")
                    print(f"Total pipeline runtime: {format_elapsed(time.perf_counter() - pipeline_started)}")
                    return

                print(f"Found {len(videos)} event clips in {game_dir}")
                print()
                selected_clips: list[tuple[int, float, Path]] = []
                unit_records: list[dict[str, Any]] = []
                intro_clip: Path | None = None
                intro_unit: dict[str, Any] | None = None
                intro_order: int | None = None
                intro_ts: float | None = None
                intro_event_id: str | None = None
                outro_clip: Path | None = None
                outro_unit: dict[str, Any] | None = None
                outro_order: int | None = None
                outro_event_id: str | None = None

                allowed_tags = ALLOWED_TAGS

                for video_path in videos:
                    event_id, event_tag = parse_event_filename(video_path)
                    video_stats = probe_video_stats(video_path)
                    frame_count = video_stats.get("frame_count")

                    event_out_dir = output_root / video_path.stem
                    event_out_dir.mkdir(parents=True, exist_ok=True)

                    boundaries_json = event_out_dir / "boundaries.json"
                    shots_csv = event_out_dir / "shots.csv"
                    shot_predictions_csv = event_out_dir / "shot_predictions.csv"
                    selected_shots_csv = event_out_dir / "selected_shots.csv"
                    selected_clip_mp4 = event_out_dir / "selected_clip.mp4"

                    print(f"Event: {video_path.name}")
                    print(f"  event_id: {event_id}")
                    print(f"  event_tag: {event_tag}")
                    print(f"  input: {video_path}")
                    print(f"  output_dir: {event_out_dir}")
                    print()

                    if shots_csv.exists():
                        print("  SBD: skipping, shots.csv already exists")
                    else:
                        print("  SBD: running")
                        sbd_started = time.perf_counter()
                        with profiler.stage("shot_boundary_detection", detail=f"event_id={event_id}; video={video_path.name}"):
                            with benchmark_session.monitor(
                                "SBD",
                                device=resolved_device,
                                frame_count=frame_count,
                                clip_count=1,
                                notes=f"event={video_path.stem}",
                            ):
                                run_sbd(
                                    video_path=video_path,
                                    out_boundaries_json=boundaries_json,
                                    out_shots_csv=shots_csv,
                                    threshold=sbd_threshold,
                                    fps=sbd_fps,
                                    ffmpeg_path=ffmpeg_path,
                                    ffprobe_path=ffprobe_path,
                                    device=device,
                                )
                        print(f"  SBD: finished in {format_elapsed(time.perf_counter() - sbd_started)}")

                    if shot_predictions_csv.exists():
                        print("  Classification: skipping, shot_predictions.csv already exists")
                    else:
                        print("  Classification: running")
                        classify_started = time.perf_counter()
                        with profiler.stage("shot_classification", detail=f"event_id={event_id}; video={video_path.name}"):
                            with benchmark_session.monitor(
                                "CLASSIFICATION",
                                device=resolved_device,
                                frame_count=frame_count,
                                clip_count=1,
                                notes=f"event={video_path.stem}",
                            ):
                                classify_event_shots(
                                    video_path=video_path,
                                    shots_csv=shots_csv,
                                    out_csv=shot_predictions_csv,
                                    weights_path=weights_path,
                                    device=device,
                                    num_samples=num_samples,
                                    batch_size=classifier_batch_size,
                                )
                        print(f"  Classification: finished in {format_elapsed(time.perf_counter() - classify_started)}")
                    
                    if event_tag in allowed_tags:
                        selection_needed = force_selection or not selected_shots_csv.exists()
                        if selection_needed:
                            print(f"  Selection: running heuristic selector (force={force_selection})")
                            with profiler.stage("selection_logic", detail=f"event_id={event_id}; video={video_path.name}"):
                                with benchmark_session.monitor(
                                    "SELECTION_LOGIC",
                                    device="cpu",
                                    clip_count=1,
                                    notes=f"event={video_path.stem}",
                                ):
                                    select_first_shot(
                                        shots_csv=shots_csv,
                                        out_csv=selected_shots_csv,
                                        debug=debug_selection,
                                        event_name=video_path.stem,
                                        video_path=video_path,
                                    )
                        else:
                            print("  Selection: reusing cached selected_shots.csv")
                            check_selection_schema(selected_shots_csv)
                        log_selection_summary(selected_shots_csv, debug_selection)
                    else:
                        print("  Selection: skipped (unsupported event tag)")

                    if event_tag in allowed_tags:
                        kept_shots = read_selected_shots(selected_shots_csv)
                        debug_print(debug_selection, f"  Debug: kept shots count={len(kept_shots)} ids={[k[0] for k in kept_shots]}")
                        if not kept_shots:
                            print("  Assembly: skipped, no kept shots found")
                        else:
                            assembly_needed = force_assembly or force_selection or not selected_clip_mp4.exists()
                            if assembly_needed:
                                print(f"  Assembly: cutting selected clip (force_selection={force_selection}, force_assembly={force_assembly})")
                                with profiler.stage("assembly_selected_clip", detail=f"event_id={event_id}; clips={len(kept_shots)}"):
                                    with benchmark_session.monitor(
                                        "VIDEO_ASSEMBLY",
                                        device="cpu",
                                        clip_count=len(kept_shots),
                                        notes=f"event={video_path.stem}; kind=selected_clip",
                                    ):
                                        assemble_selected_clips(
                                            ffmpeg_path=ffmpeg_path,
                                            input_video=video_path,
                                            kept_shots=kept_shots,
                                            output_video=selected_clip_mp4,
                                            work_dir=event_out_dir / "_selected",
                                            debug=debug_selection,
                                        )
                            else:
                                print("  Assembly: reusing cached selected_clip.mp4")
                            ts = events_metadata.get(event_id, {}).get("from_timestamp")
                            if ts is None:
                                print("  Assembly: missing metadata timestamp; clip will be placed last")
                                ts_value = float("inf")
                            else:
                                ts_value = ts
                            order_idx = event_order.get(event_id, ORDER_FALLBACK)

                            unit_type, intra_order = event_unit_metadata(event_tag)
                            unit_records.append(
                                {
                                    "type": unit_type,
                                    "event_id": event_id,
                                    "order_idx": order_idx,
                                    "ts": ts_value,
                                    "path": selected_clip_mp4,
                                    "intra_order": intra_order,
                                }
                            )

                            if event_tag in {"goal", "red_card", "redcard"}:
                                reaction_csv = event_out_dir / "reaction_shots.csv"
                                reaction_rows = select_reaction_shots(
                                    selected_shots_csv,
                                    shot_predictions_csv,
                                    event_tag=event_tag,
                                )
                                write_reaction_csv(reaction_rows, reaction_csv)
                                if reaction_rows:
                                    reaction_clip_mp4 = event_out_dir / "reaction_clip.mp4"
                                    with profiler.stage("assembly_reaction_clip", detail=f"event_id={event_id}; clips={len(reaction_rows)}"):
                                        with benchmark_session.monitor(
                                            "VIDEO_ASSEMBLY",
                                            device="cpu",
                                            clip_count=len(reaction_rows),
                                            notes=f"event={video_path.stem}; kind=reaction_clip",
                                        ):
                                            assemble_reaction_clip(
                                                ffmpeg_path=ffmpeg_path,
                                                input_video=video_path,
                                                reaction_rows=reaction_rows,
                                                output_video=reaction_clip_mp4,
                                                work_dir=event_out_dir / "_reaction",
                                                debug=debug_selection,
                                            )
                                    selected_clips.append((order_idx + REACTION_ORDER_OFFSET, ts_value + REACTION_ORDER_OFFSET, reaction_clip_mp4))
                                    reaction_unit_type = "goal_reaction" if event_tag == "goal" else "red_card_reaction"
                                    unit_records.append(
                                        {
                                            "type": reaction_unit_type,
                                            "event_id": event_id,
                                            "order_idx": order_idx,
                                            "ts": ts_value,
                                            "path": reaction_clip_mp4,
                                            "intra_order": 1,
                                        }
                                    )

                            if event_tag == "goal":
                                replay_csv = event_out_dir / "replay_shots.csv"
                                replay_started = time.perf_counter()
                                with profiler.stage("replay_logo_detection", detail=f"event_id={event_id}; video={video_path.name}"):
                                    with benchmark_session.monitor(
                                        "LOGO_DETECTION",
                                        device=resolved_device,
                                        frame_count=frame_count,
                                        clip_count=1,
                                        notes=f"event={video_path.stem}",
                                    ):
                                        extract_replay_shots(
                                            video_path=video_path,
                                            shots_csv=shots_csv,
                                            out_csv=replay_csv,
                                            weights_path=Path("weights/logo_ntf_2024_resnet50.pth"),
                                            threshold=0.8,
                                            device=device,
                                            max_replay_shots=2 if time_budget is None or time_budget > 120.0 else 1,
                                            debug=debug_selection,
                                        )
                                print(f"  Replay detection: finished in {format_elapsed(time.perf_counter() - replay_started)}")
                                replay_kept = read_replay_shots(replay_csv)
                                if replay_kept:
                                    replay_clip_mp4 = event_out_dir / "replay_clip.mp4"
                                    with profiler.stage("assembly_replay_clip", detail=f"event_id={event_id}; clips={len(replay_kept)}"):
                                        with benchmark_session.monitor(
                                            "VIDEO_ASSEMBLY",
                                            device="cpu",
                                            clip_count=len(replay_kept),
                                            notes=f"event={video_path.stem}; kind=replay_clip",
                                        ):
                                            assemble_replay_clip(
                                                ffmpeg_path=ffmpeg_path,
                                                input_video=video_path,
                                                replay_shots=replay_kept,
                                                output_video=replay_clip_mp4,
                                                work_dir=event_out_dir / "_replay",
                                                debug=debug_selection,
                                            )
                                    selected_clips.append((order_idx + REPLAY_ORDER_OFFSET, ts_value + REPLAY_ORDER_OFFSET, replay_clip_mp4))
                                    unit_records.append(
                                        {
                                            "type": "goal_replay",
                                            "event_id": event_id,
                                            "order_idx": order_idx,
                                            "ts": ts_value,
                                            "path": replay_clip_mp4,
                                            "intra_order": 2,
                                        }
                                    )

                            if event_tag == "start_phase":
                                prev_intro_id = intro_event_id
                                intro_clip, intro_order, intro_ts, intro_event_id = update_intro_candidate(
                                    intro_clip,
                                    intro_order,
                                    intro_ts,
                                    intro_event_id,
                                    selected_clip_mp4,
                                    order_idx,
                                    ts_value,
                                    event_id,
                                )
                                if intro_event_id != prev_intro_id:
                                    intro_unit = {
                                        "type": "intro",
                                        "event_id": event_id,
                                        "order_idx": order_idx,
                                        "ts": ts_value,
                                        "path": selected_clip_mp4,
                                        "intra_order": -1,
                                    }
                            elif event_tag == "end_of_game":
                                prev_outro_id = outro_event_id
                                outro_clip, outro_order, outro_event_id = update_outro_candidate(
                                    outro_clip,
                                    outro_order,
                                    outro_event_id,
                                    selected_clip_mp4,
                                    order_idx,
                                    ts_value,
                                    event_id,
                                    events_metadata,
                                )
                                if outro_event_id != prev_outro_id:
                                    outro_unit = {
                                        "type": "outro",
                                        "event_id": event_id,
                                        "order_idx": order_idx,
                                        "ts": ts_value,
                                        "path": selected_clip_mp4,
                                        "intra_order": 99,
                                    }
                            else:
                                selected_clips.append((order_idx, ts_value, selected_clip_mp4))
                    else:
                        print("  Assembly: skipped (unsupported event tag)")
                
                final_highlight = output_root / "highlight_reel.mp4"
                assembly_work_dir = output_root / "_assembly"

                units_for_budget = list(unit_records)
                if intro_unit:
                    units_for_budget.append(intro_unit)
                if outro_unit:
                    units_for_budget.append(outro_unit)

                if time_budget is None:
                    if selected_clips or intro_clip or outro_clip:
                        need_final = force_selection or force_assembly or not final_highlight.exists()
                        if not need_final:
                            print(f"Final highlight exists and force not set; reusing {final_highlight}")
                        else:
                            selected_clips.sort(key=lambda pair: (pair[0], pair[1]))
                            ordered_clips = [p for _, _, p in selected_clips]
                            final_list: list[Path] = []
                            if intro_clip is not None:
                                final_list.append(intro_clip)
                                debug_print(debug_selection, f"[assembly] intro clip added (event={intro_event_id})")
                            final_list.extend(ordered_clips)
                            if outro_clip is not None:
                                final_list.append(outro_clip)
                                debug_print(debug_selection, f"[assembly] outro clip added (event={outro_event_id})")
                            if not final_list:
                                print("No selected clips to assemble.")
                                print(f"Total pipeline runtime: {format_elapsed(time.perf_counter() - pipeline_started)}")
                                return
                            print("Creating final MVP highlight (chronological order)...")
                            with profiler.stage("final_video_creation", detail=f"kind=chronological; clips={len(final_list)}"):
                                with benchmark_session.monitor(
                                    "VIDEO_ASSEMBLY",
                                    device="cpu",
                                    clip_count=len(final_list),
                                    notes="kind=final_highlight",
                                ):
                                    write_final_assembly_manifest(
                                        assembly_work_dir,
                                        final_list,
                                        units_for_budget,
                                    )
                                    crossfade_clips(
                                        ffmpeg_path=ffmpeg_path,
                                        clip_paths=final_list,
                                        output_video=final_highlight,
                                        work_dir=assembly_work_dir,
                                    )
                            print(f"Final highlight written to: {final_highlight}")
                    else:
                        print("No selected clips found, skipping final highlight assembly.")
                else:
                    plan = build_clip_plan(
                        units_for_budget,
                        ffprobe_path=ffprobe_path,
                        time_budget_seconds=time_budget,
                        debug=debug_selection,
                    )
                    if not plan:
                        print("No clips fit within the time budget; skipping final assembly.")
                        print(f"Total pipeline runtime: {format_elapsed(time.perf_counter() - pipeline_started)}")
                        return
                    print(f"Creating budgeted highlight (time_budget={time_budget}s)...")
                    with profiler.stage("final_video_creation", detail=f"kind=budgeted; clips={len(plan)}; time_budget={time_budget}"):
                        with benchmark_session.monitor(
                            "VIDEO_ASSEMBLY",
                            device="cpu",
                            clip_count=len(plan),
                            notes="kind=final_highlight_budgeted",
                        ):
                            write_final_assembly_manifest(
                                assembly_work_dir,
                                plan,
                                units_for_budget,
                            )
                            crossfade_clips(
                                ffmpeg_path=ffmpeg_path,
                                clip_paths=plan,
                                output_video=final_highlight,
                                work_dir=assembly_work_dir,
                            )
                    print(f"Final highlight written to: {final_highlight}")

                print(f"Total pipeline runtime: {format_elapsed(time.perf_counter() - pipeline_started)}")
    finally:
        benchmark_session.flush()
        benchmark_session.print_summary()
        try:
            profiler.close()
        finally:
            set_active_profiler(NoOpProfiler())


if __name__ == "__main__":
    main()
