from __future__ import annotations
"""Small PATH/binary helpers shared by external video tools."""

import os
from pathlib import Path


def add_bin_to_path(bin_dir: str) -> None:
    """
    Adds a directory to PATH for the current Python process.
    This does not modify the system PATH permanently.
    """
    current = os.environ.get("PATH", "")
    parts = current.split(";") if current else []
    if bin_dir not in parts:
        os.environ["PATH"] = bin_dir + ";" + current


def ensure_ffmpeg_on_path(ffmpeg_path: str) -> None:
    """
    Some libraries (like sportsbd) call 'ffmpeg' directly.
    If ffmpeg is not on PATH, add its folder for this process.
    """
    bin_dir = str(Path(ffmpeg_path).parent)
    add_bin_to_path(bin_dir)


def ffprobe_from_ffmpeg(ffmpeg_path: str) -> str:
    """
    If ffmpeg.exe exists, ffprobe.exe is usually in the same folder.
    Returns the correct ffprobe path.
    """
    p = Path(ffmpeg_path)

    if p.suffix.lower() == ".exe" and p.exists():
        candidate = p.parent / "ffprobe.exe"
        if candidate.exists():
            return str(candidate)

    return "ffprobe"
