from __future__ import annotations
"""Replay detection based on logo transitions in longer goal clips."""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision
from torchvision import transforms
import cv2

from pipeline.profiling import get_profiler


WEIGHTS_PATH_DEFAULT = Path("weights/logo_ntf_2024_resnet50.pth")
LOGO_CLASS_IDX = 1
LOGO_THRESH = 0.8
LOGO_BATCH_SIZE = 16
MAX_REPLAY_SHOT_MS = 15_000

# Cache loaded models per (weights, device) so multiple goal events reuse one model instance.
_LOGO_MODEL_CACHE: Dict[tuple[Path, str], torch.nn.Module] = {}


def get_inference_device(device: str | None = None) -> torch.device:
    """Resolve replay inference device while keeping CPU fallback safe."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def load_logo_model(weights_path: Path | None = None, device: str | None = None) -> torch.nn.Module:
    """Load the binary logo detector, reusing a cached copy when possible."""
    weights_path = weights_path or WEIGHTS_PATH_DEFAULT
    dev = get_inference_device(device)
    cache_key = (weights_path.resolve(), str(dev))
    cached = _LOGO_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    model = torchvision.models.resnet50(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.BatchNorm1d(model.fc.in_features),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(model.fc.in_features, 512),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(512),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 2),
    )
    checkpoint = torch.load(weights_path, map_location=dev, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(dev)
    if dev.type == "cuda":
        # Channels-last can help throughput for standard image backbones on CUDA.
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    _LOGO_MODEL_CACHE[cache_key] = model
    return model


def detect_first_logo_end_ms(
    video_path: Path,
    model: torch.nn.Module,
    threshold: float = LOGO_THRESH,
    batch_size: int = LOGO_BATCH_SIZE,
) -> Optional[float]:
    """
    Returns end time (ms) of first grouped logo event, or None if no logo.
    Frames are streamed; detection starts at 25s to skip early content.
    Group positives separated by <= 5 frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame_threshold = int(25.0 * fps)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # OpenCV gives HWC uint8, so convert to normalized tensor first.
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    positives: List[int] = []
    batch: List[torch.Tensor] = []
    idxs: List[int] = []
    frame_idx = 0
    model_device = next(model.parameters()).device
    use_amp = model_device.type == "cuda"
    profiler = get_profiler()

    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx < start_frame_threshold:
                frame_idx += 1
                continue

            # OpenCV decodes frames as BGR; the classifier was trained on RGB input.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(frame_rgb)
            batch.append(tensor)
            idxs.append(frame_idx)

            if len(batch) == batch_size:
                detail = f"frames={len(batch)}; device={model_device}"
                with profiler.stage("replay_logo_batch_inference", detail=detail):
                    stacked = torch.stack(batch, dim=0)
                    if model_device.type == "cuda":
                        stacked = stacked.to(device=model_device, memory_format=torch.channels_last, non_blocking=True)
                    else:
                        stacked = stacked.to(device=model_device)
                    with torch.autocast(device_type=model_device.type, enabled=use_amp):
                        logits = model(stacked)
                    probs = torch.softmax(logits, dim=1)[:, LOGO_CLASS_IDX]
                    for fi, p in zip(idxs, probs):
                        if p.item() >= threshold:
                            positives.append(fi)
                batch.clear()
                idxs.clear()

            frame_idx += 1

        # flush remaining
        if batch:
            detail = f"frames={len(batch)}; device={model_device}; tail=1"
            with profiler.stage("replay_logo_batch_inference", detail=detail):
                stacked = torch.stack(batch, dim=0)
                if model_device.type == "cuda":
                    stacked = stacked.to(device=model_device, memory_format=torch.channels_last, non_blocking=True)
                else:
                    stacked = stacked.to(device=model_device)
                with torch.autocast(device_type=model_device.type, enabled=use_amp):
                    logits = model(stacked)
                probs = torch.softmax(logits, dim=1)[:, LOGO_CLASS_IDX]
                for fi, p in zip(idxs, probs):
                    if p.item() >= threshold:
                        positives.append(fi)

    cap.release()

    if not positives:
        return None

    # group positives separated by at most 5 frames
    gap = 5
    positives.sort()
    groups: List[Tuple[int, int]] = []
    start = positives[0]
    end = positives[0]
    for idx in positives[1:]:
        if idx - end <= gap:
            end = idx
        else:
            groups.append((start, end))
            start = idx
            end = idx
    groups.append((start, end))

    # first grouped logo event in processed window
    g_start, g_end = groups[0]
    end_ms = (g_end + 1) * 1000.0 / fps
    return end_ms


def select_replay_shots(
    shots_csv: Path,
    logo_end_ms: float,
    max_replay_shots: int = 2,
    max_replay_shot_ms: int = MAX_REPLAY_SHOT_MS,
) -> List[Dict[str, str | int]]:
    """Pick the first shot boundaries that begin after the first replay-logo transition."""
    with shots_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        shots = list(reader)

    # Ensure the candidate shots are considered in chronological order.
    shots.sort(key=lambda row: int(row.get("start_ms", 0)))

    replay_rows: List[Dict[str, str | int]] = []
    for row in shots:
        try:
            start_ms = int(row["start_ms"])
            end_ms = int(row["end_ms"])
        except (KeyError, ValueError, TypeError):
            continue
        duration_ms = end_ms - start_ms
        if duration_ms > max_replay_shot_ms:
            continue
        if start_ms > logo_end_ms:
            replay_rows.append(
                {
                    "shot_id": row.get("shot_id", ""),
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": row.get("duration_ms", duration_ms),
                    "label": row.get("label", ""),
                    "replay": 1,
                }
            )
        if len(replay_rows) >= max_replay_shots:
            break
    return replay_rows


def write_replay_csv(replay_rows: List[Dict[str, str | int]], out_csv: Path) -> None:
    """Persist the replay shot rows selected for one event."""
    fieldnames = ["shot_id", "start_ms", "end_ms", "duration_ms", "label", "replay"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(replay_rows)


def extract_replay_shots(
    video_path: Path,
    shots_csv: Path,
    out_csv: Path,
    weights_path: Path | None = None,
    threshold: float = LOGO_THRESH,
    device: str | None = None,
    max_replay_shots: int = 2,
    max_replay_shot_ms: int = MAX_REPLAY_SHOT_MS,
    debug: bool = False,
) -> None:
    """
    Run logo detection and pick the first replay shots after the first logo event.
    Goal events only (caller ensures). Safe no-op on failure.
    """
    try:
        model = load_logo_model(weights_path, device=device)
        logo_end_ms = detect_first_logo_end_ms(video_path, model, threshold=threshold)
        if logo_end_ms is None:
            if debug:
                print(f"[replay] no logo found for {video_path.name}")
            write_replay_csv([], out_csv)
            return
        replay_rows = select_replay_shots(
            shots_csv,
            logo_end_ms,
            max_replay_shots=max_replay_shots,
            max_replay_shot_ms=max_replay_shot_ms,
        )
        if debug:
            print(f"[replay] logo_end_ms={logo_end_ms:.1f} selected {len(replay_rows)} shots")
        write_replay_csv(replay_rows, out_csv)
    except Exception as exc:
        if debug:
            print(f"[replay] failed for {video_path.name}: {exc}")
        write_replay_csv([], out_csv)
