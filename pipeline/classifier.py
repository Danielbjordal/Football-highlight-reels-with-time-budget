from __future__ import annotations
"""Camera-shot classification helpers used during per-event labeling."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2

from pipeline.profiling import get_profiler


CLASS_NAMES = [
    "close_up_player_or_field_referee",
    "close_up_side_staff",
    "close_up_corner",
    "main_camera_center",
    "main_camera_left",
    "main_camera_right",
    "behind_the_goal",
    "public",
]


@dataclass(frozen=True)
class CameraClassifier:
    """Container for the loaded model and the preprocessing it expects."""
    model: torch.nn.Module
    transform: transforms.Compose
    device: torch.device
    use_amp: bool


def resolve_torch_device(device: str | None = None) -> torch.device:
    """Resolve a user/config device string while keeping CPU fallback safe."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def load_camera_classifier(weights_path: str | Path, device: str | None = None) -> CameraClassifier:
    """
    Loads the pretrained ResNet50 camera classifier for inference only.

    weights_path: path to .pth state_dict
    device: "cpu" or "cuda" (optional). If None, auto-detect.
    """
    weights_path = Path(weights_path)

    dev = resolve_torch_device(device)

    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    state_dict = torch.load(weights_path, map_location=dev)
    if isinstance(state_dict, dict):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(dev)
    if dev.type == "cuda":
        # Channels-last can improve throughput for 2D CNN inference on CUDA.
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return CameraClassifier(
        model=model,
        transform=transform,
        device=dev,
        use_amp=(dev.type == "cuda"),
    )


def predict_frame(clf: CameraClassifier, image_rgb: np.ndarray) -> Tuple[str, float, List[float]]:
    """
    Predict on ONE frame (RGB numpy array HxWx3 uint8).
    Returns: (top_class, top_confidence, probs_list)
    """
    probs = predict_frames_batched(clf, [image_rgb], batch_size=1)[0]

    top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    return CLASS_NAMES[top_idx], float(probs[top_idx]), probs


def sample_frames_from_shot(
    video_path: str,
    start_ms: int,
    end_ms: int,
    num_samples: int = 3,
    cap: cv2.VideoCapture | None = None,
) -> List[np.ndarray]:
    """
    Returns a list of RGB frames sampled uniformly from [start_ms, end_ms].
    Uses OpenCV seeking by milliseconds and can reuse an existing capture handle.
    """
    owns_cap = cap is None
    if cap is None:
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    duration = max(1, end_ms - start_ms)
    times = [
        start_ms + int(duration * (i + 1) / (num_samples + 1))
        for i in range(num_samples)
    ]

    frames: List[np.ndarray] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    if owns_cap:
        cap.release()
    return frames


def predict_frames_batched(
    clf: CameraClassifier,
    frames_rgb: List[np.ndarray],
    batch_size: int = 16,
) -> List[List[float]]:
    """
    Predict probabilities for multiple RGB frames using batched inference.
    """
    if not frames_rgb:
        return []

    tensors = [clf.transform(frame) for frame in frames_rgb]
    probs_out: List[List[float]] = []
    profiler = get_profiler()

    with torch.inference_mode():
        for start in range(0, len(tensors), max(1, batch_size)):
            batch_end = start + max(1, batch_size)
            detail = f"frames={len(tensors[start:batch_end])}; device={clf.device}"
            with profiler.stage("classification_batch_inference", detail=detail):
                batch = torch.stack(tensors[start:batch_end], dim=0)
                if clf.device.type == "cuda":
                    batch = batch.to(device=clf.device, memory_format=torch.channels_last, non_blocking=True)
                else:
                    batch = batch.to(device=clf.device)
                with torch.autocast(device_type=clf.device.type, enabled=clf.use_amp):
                    logits = clf.model(batch)
                    probs = F.softmax(logits, dim=1)
                probs_out.extend(probs.detach().cpu().numpy().tolist())

    return probs_out


def predict_shot(
    clf: CameraClassifier,
    video_path: str,
    start_ms: int,
    end_ms: int,
    num_samples: int = 3,
    batch_size: int = 16,
    cap: cv2.VideoCapture | None = None,
) -> Tuple[str, float, List[float]]:
    """
    Predict class for a shot by averaging probabilities across sampled frames.
    Returns: (top_class, top_confidence, probs_avg_list).
    """
    frames = sample_frames_from_shot(video_path, start_ms, end_ms, num_samples=num_samples, cap=cap)
    if not frames:
        return "unknown", 0.0, [0.0] * len(CLASS_NAMES)

    probs_acc = np.zeros((len(CLASS_NAMES),), dtype=np.float64)
    for probs in predict_frames_batched(clf, frames, batch_size=batch_size):
        probs_acc += np.array(probs, dtype=np.float64)

    probs_avg = (probs_acc / len(frames)).tolist()
    top_idx = int(max(range(len(probs_avg)), key=lambda i: probs_avg[i]))
    return CLASS_NAMES[top_idx], float(probs_avg[top_idx]), probs_avg


if __name__ == "__main__":
    # Smoke test: load model and run one shot prediction.
    clf = load_camera_classifier("weights/resnet50_forzasys_soccer_camera_zoom_v2.pth", device="cpu")
    print("Loaded camera classifier")
    print(f"  device: {clf.device}")
    print(f"  num_classes: {clf.model.fc.out_features}")

    # Adjust path + times as needed
    video_path = "data/events/event_01.mp4"
    label, conf, _ = predict_shot(clf, video_path, start_ms=2000, end_ms=6000, num_samples=3)
    print(f"Shot prediction: {label} (conf={conf:.3f})")
