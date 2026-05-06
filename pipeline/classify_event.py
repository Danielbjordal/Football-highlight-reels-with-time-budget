from __future__ import annotations
"""Apply the camera classifier to every shot in one event clip."""

import csv
from pathlib import Path

import cv2

from pipeline.classifier import CLASS_NAMES, load_camera_classifier, predict_shot


def classify_event_shots(
    video_path: Path,
    shots_csv: Path,
    out_csv: Path,
    weights_path: str,
    device: str = "cpu",
    num_samples: int = 3,
    batch_size: int = 16,
) -> None:
    """
    Reads shots.csv, classifies each shot, and writes shot_predictions.csv.
    """
    clf = load_camera_classifier(weights_path=weights_path, device=device)

    rows = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        # Reuse one capture handle for the whole event to avoid reopening the file per shot.
        with shots_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                shot_id = int(row["shot_id"])
                start_ms = int(row["start_ms"])
                end_ms = int(row["end_ms"])

                label, confidence, probs = predict_shot(
                    clf,
                    video_path=str(video_path),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    num_samples=num_samples,
                    batch_size=batch_size,
                    cap=cap,
                )

                out_row = {
                    "shot_id": shot_id,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "label": label,
                    "confidence": confidence,
                }

                for class_name, prob in zip(CLASS_NAMES, probs):
                    out_row[f"prob_{class_name}"] = prob

                rows.append(out_row)
    finally:
        cap.release()

    fieldnames = [
        "shot_id",
        "start_ms",
        "end_ms",
        "label",
        "confidence",
    ] + [f"prob_{name}" for name in CLASS_NAMES]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
