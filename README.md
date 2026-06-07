# music-reel

Automatically create Instagram-ready vertical videos from music performances.

Analyzes audio to find the most interesting solo moments, then converts the segment to vertical format with song title overlay.

## Requirements

- Python 3.11+ (uses `tomllib` from stdlib)
- `pip install librosa numpy scipy` (or direnv allow to use the nix shell if you use NixOS)
- `ffmpeg` + `ffprobe` on `PATH`

## Usage

```bash
python music-reel.py "Alice In Wonderland.mp4"
```

Output is written to `<input>_reel.mp4` in the same directory.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--duration` / `-d` | 30 | Target clip length in seconds |
| `--min-duration` / `--max-duration` | — | Enables smart boundary snapping to nearby valleys so cuts don't land mid-phrase |
| `--format` / `-f` | `crop` | `crop` (center 9:16) or `blur` (blurred background, preserves full frame) |
| `--full-video` | off | Skips audio analysis and keeps the entire video. Auto-defaults `--format` to `blur` |
| `--start` / `--end` | — | Manual segment in seconds (skips auto-detection) |
| `--title` | auto | Custom title (default: derived from filename) |
| `--subtitle` | — | Second text line below the title |
| `--output` / `-o` | auto | Output file path |
| `--batch` | off | Process all videos in a directory |

### Optional config

Place a `.toml` file next to the video with the same filename stem:

```toml
title = "Alice In Wonderland"
subtitle = "solo transcription"
min_duration = 25
max_duration = 35
```

CLI flags override those values.

### Examples

```bash
# Auto-detect the best 30s and center-crop to 9:16
python music-reel.py "My Solo.mp4"

# 45-second clip with blurred background
python music-reel.py "My Solo.mp4" --duration 45 --format blur

# Manually choose the segment
python music-reel.py "Concert.mp4" --start 120 --end 165

# Keep the whole video (no cutting)
python music-reel.py "Full Set.mp4" --full-video

# Process everything in a folder
python music-reel.py ./recordings/ --batch

# Custom title and subtitle
python music-reel.py "take2.mp4" --title "Cherokee" --subtitle "🎹 chorusing"
```

## How it works

1. Extracts audio and computes RMS energy, onset strength, spectral centroid, and zero-crossing rate
2. Combines these into an "interest" score and smooths it
3. Slides a window across the score to find the most interesting segment
4. If `--min-duration` / `--max-duration` are given, snaps boundaries to the nearest low-activity valleys to avoid cutting mid-phrase
5. Trims the video, converts to 1080×1920, and overlays the song title
