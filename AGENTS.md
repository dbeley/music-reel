# music-reel

Single-file Python script that makes Instagram-ready vertical videos from music performances.

## Running

```bash
python music-reels path/to/video.mp4
```

No package or build step — just run the script. The `.py` extension is omitted from the filename.

## Dependencies

- **Python**: `librosa`, `numpy`, `scipy` — install with `pip install librosa numpy scipy`
- **System**: `ffmpeg` + `ffprobe` (must be on `PATH`)

## Sidecar config

Place a `.toml` file next to the video with the same stem to set `title`, `subtitle`, `min_duration`, `max_duration` — these merge with (and are overridden by) CLI flags.

## Key CLI flags

| Flag | Default | Notes |
|------|---------|-------|
| `--duration` | 30 | Target clip length in seconds |
| `--min-duration` / `--max-duration` | — | Enables smart boundary snapping to valleys |
| `--format` | `crop` | `crop` (center 9:16) or `blur` (blurred background, preserves full frame) |
| `--full-video` | off | Skips audio analysis, defaults `--format` to `blur` |
| `--start` / `--end` | — | Manual segment (skips auto-detection) |
| `--batch` | off | Process all videos in a directory |
| `--subtitle` | — | Second text line below title |

## No tests, no CI, no linters

Single-script repo with zero test suite, no formatter config, no CI. Manual testing only.
