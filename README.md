# Football Highlight Generation Pipeline

This repository contains a Python pipeline for generating football highlight reels from Eliteserien event clips. It downloads and filters game events, runs shot boundary detection, classifies camera shot types with a ResNet50 model, selects useful moments, detects replay-logo transitions for goals, and assembles the selected clips into a final highlight video.

## What the pipeline does

For each event clip in `data/games/<game_id>/`, the pipeline runs these stages:

1. Shot boundary detection, producing `boundaries.json` and `shots.csv`.
2. Shot type classification, producing `shot_predictions.csv`.
3. Event-specific selection, producing `selected_shots.csv`.
4. Optional reaction and replay extraction for goal/red-card style events.
5. FFmpeg assembly of selected clips into `outputs/<game_id>/highlight_reel.mp4`.

The pipeline reuses existing outputs by default. Use the force flags described below when you want to regenerate cached stages.

## Private files not included

The repository intentionally does not include private API endpoints or trained model weights.

Contact my supervisor, `[INSERT SUPERVISOR NAME]`, for:

- The Eliteserien API base endpoint.
- The ResNet50 shot type classifier weights.
- The ResNet50 logo detection weights.

Place the model files in the existing `weights/` folder:

```text
weights/
  resnet50_forzasys_soccer_camera_zoom_v2.pth
  logo_ntf_2024_resnet50.pth
```

The `weights/` directory is kept in git with `.gitkeep`, but checkpoint files are ignored and should not be committed.

## Installation

Use Python 3.10 or newer. The project was developed on Windows, but the commands are standard Python/FFmpeg commands and should also work on Linux/macOS with path adjustments.

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install FFmpeg and FFprobe:

- Windows: install FFmpeg with winget, Chocolatey, or from the official FFmpeg builds.
- macOS: `brew install ffmpeg`
- Linux: use your distribution package manager, for example `sudo apt install ffmpeg`.

Make sure `ffmpeg` and `ffprobe` are available on `PATH`, or set `tools.ffmpeg_path` in `config.yaml` to the full path of the FFmpeg executable.

4. Create a local config:

```bash
copy config.example.yaml config.yaml
```

On macOS/Linux:

```bash
cp config.example.yaml config.yaml
```

Then edit `config.yaml` if needed. Common values are:

- `defaults.game_id`: game folder to process.
- `defaults.weights_path`: shot type classifier checkpoint path.
- `classifier.device`: `auto`, `cpu`, or `cuda`.
- `tools.ffmpeg_path`: `ffmpeg` or a full executable path.

5. Configure the private API endpoint when downloading events:

```powershell
$env:ELITESERIEN_API_BASE = "INSERT_ELITESERIEN_API_BASE_HERE"
```

On macOS/Linux:

```bash
export ELITESERIEN_API_BASE="INSERT_ELITESERIEN_API_BASE_HERE"
```

The endpoint value should include the base path used before `/game/<game_id>/events`.

## Downloading event clips

The main pipeline expects this structure:

```text
data/games/<game_id>/
  events.json
  target_events.json
  events_metadata.csv
  <event_id>_<event_type>.mp4
```

To fetch clips with the audio-recovery downloader:

```bash
python -m pipeline.download_game_events_audiofix
```

The downloader uses the `GAME_IDS` list inside `pipeline/download_game_events_audiofix.py`. Edit that list for the game IDs you want to fetch. Downloaded MP4 files are ignored by git.

## Using the CLI

The main command-line interface is:

```bash
python -m pipeline.run
```

Show the full CLI help menu:

```bash
python -m pipeline.run --help
```

Common commands:

```bash
python -m pipeline.run
python -m pipeline.run --game-id 4407
python -m pipeline.run --game-id 4407 --device cpu
python -m pipeline.run --game-id 4407 --weights weights/resnet50_forzasys_soccer_camera_zoom_v2.pth
python -m pipeline.run --game-id 4407 --force-selection --force-assembly
python -m pipeline.run --game-id 4407 --time_budget 120
python -m pipeline.run --game-id 4407 --debug-selection
python -m pipeline.run --game-id 4407 --benchmark
python -m pipeline.run --game-id 4407 --profile
```

Important options:

- `--config`: path to the YAML config file. Defaults to `config.yaml`.
- `--game-id`: overrides `defaults.game_id`.
- `--ffmpeg`: overrides `tools.ffmpeg_path`.
- `--weights`: overrides the shot type classifier checkpoint path.
- `--device`: `auto`, `cpu`, or `cuda`.
- `--force-selection`: regenerates selection CSVs and selected clips.
- `--force-assembly`: regenerates assembled clips and final highlight video.
- `--time_budget`: creates a final reel constrained to a target number of seconds.
- `--debug-selection`: prints extra selection and assembly details.
- `--benchmark`: records component runtime/resource measurements.
- `--profile`: writes lightweight pipeline stage timings.

## Outputs

Per-event outputs are written to:

```text
outputs/<game_id>/<event_id>_<event_type>/
```

The final highlight video is written to:

```text
outputs/<game_id>/highlight_reel.mp4
```

Generated video files, benchmark logs, local config, downloaded event clips, and model weights are intentionally ignored by git.
