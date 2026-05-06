# Roadmap Overview

Purpose: track current pipeline state and next steps for the master thesis on automated video highlight generation for football.

## Quick start

- Ensure FFmpeg is available at the configured path in `config.yaml` (`tools.ffmpeg_path`).
- Ensure ResNet weights are present: `weights/resnet50_forzasys_soccer_camera_zoom_v2.pth`.
- Default game is set in `config.yaml` (`defaults.game_id`, currently `4418`); override with `--game-id <id>` if needed.
- Run pipeline:

```bash
python -m pipeline.run            # uses defaults in config.yaml
python -m pipeline.run --game-id 4418  # explicit game id (optional while default is 4418)
python -m pipeline.run --force-selection --force-assembly --debug-selection
```

(Uses CPU; will skip steps when outputs already exist.)

## Current pipeline at a glance

1. Shot boundary detection ? `boundaries.json`, `shots.csv` per event.
2. Camera viewpoint classification (ResNet50) ? `shot_predictions.csv`.
3. MVP selection (first shot of goal events) ? `selected_shots.csv`.
4. Cut selected clips ? `selected_clip.mp4`.
5. Concatenate all selected goal clips ? `outputs/4418/highlight_mvp.mp4`.

See `pipeline.md` for step details and `backlog.md` for planned improvements.
