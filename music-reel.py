#!/usr/bin/env python3
"""
Music Reels Generator - Automatically create Instagram-ready vertical videos from music performances.

Analyzes audio to find the most interesting solo moments, then converts to vertical format
with song title overlay.

Usage:
    python music-reels.py "path/to/video.mp4"
    python music-reels.py "path/to/video.mp4" --duration 45 --output "my_reel.mp4"
    python music-reels.py "path/to/video.mp4" --format blur  # blurred background instead of crop
"""

import argparse
import subprocess
import sys
from pathlib import Path
import tempfile
import json
import tomllib

try:
    import librosa
    import numpy as np
except ImportError:
    print("Error: This tool requires librosa and numpy.")
    print("Install with: pip install librosa numpy")
    print(
        "Or run with: nix run nixpkgs#python3Packages.librosa nixpkgs#python3Packages.numpy -- python music-reels.py"
    )
    sys.exit(1)


def get_video_duration(video_path: str) -> float:
    """Get the total duration of a video file in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Could not get video duration: {result.stderr}")
    return float(result.stdout.strip())


def analyze_audio(
    audio_path: str,
    duration: int = 30,
    min_duration: int | None = None,
    max_duration: int | None = None,
) -> tuple[float, float]:
    """
    Analyze audio to find the most interesting section, then snap
    boundaries to nearby low-activity valleys to avoid cutting mid-phrase.

    Returns: (start_time, end_time) in seconds
    """
    print(f"Analyzing audio from {audio_path}...")

    # Load audio
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Compute features that indicate "interesting" music solo sections
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]

    rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-10)
    onset_norm = (onset_env - onset_env.min()) / (
        onset_env.max() - onset_env.min() + 1e-10
    )
    centroid_norm = (spectral_centroid - spectral_centroid.min()) / (
        spectral_centroid.max() - spectral_centroid.min() + 1e-10
    )
    zcr_norm = (zcr - zcr.min()) / (zcr.max() - zcr.min() + 1e-10)

    interest = 0.4 * onset_norm + 0.3 * rms_norm + 0.2 * centroid_norm + 0.1 * zcr_norm

    from scipy.ndimage import uniform_filter1d

    interest_smooth = uniform_filter1d(interest, size=50)

    times = librosa.frames_to_time(
        np.arange(len(interest_smooth)), sr=sr, hop_length=512
    )

    duration_frames = int(duration * sr / 512)

    best_score = -1
    best_start = 0

    for i in range(len(interest_smooth) - duration_frames):
        segment_score = np.mean(interest_smooth[i : i + duration_frames])
        if segment_score > best_score:
            best_score = segment_score
            best_start = i

    best_end = best_start + duration_frames

    if min_duration is not None or max_duration is not None:
        min_dur = min_duration or duration
        max_dur = max_duration or duration
        min_frames = int(min_dur * sr / 512)
        max_frames = int(max_dur * sr / 512)

        budget = max_frames - duration_frames
        search_before = min(budget // 2, best_start)
        search_after = min(budget - search_before, len(interest_smooth) - best_end)

        if search_before > 0:
            window = interest_smooth[best_start - search_before : best_start + 1]
            snap_start = best_start - search_before + int(np.argmin(window))
        else:
            snap_start = best_start

        if search_after > 0:
            window = interest_smooth[best_end : best_end + search_after + 1]
            snap_end = best_end + int(np.argmin(window))
        else:
            snap_end = best_end

        seg_len = snap_end - snap_start
        if seg_len > max_frames:
            excess = seg_len - max_frames
            snap_start += excess // 2
            snap_end -= excess - excess // 2
        if seg_len < min_frames:
            deficit = min_frames - seg_len
            snap_start = max(0, snap_start - deficit // 2)
            snap_end = min(len(interest_smooth) - 1, snap_end + deficit - deficit // 2)

        best_start = snap_start
        best_end = snap_end

    start_time = times[best_start]
    end_time = times[min(best_end, len(times) - 1)]

    print(
        f"Best segment found: {start_time:.1f}s - {end_time:.1f}s (score: {best_score:.3f}, duration: {end_time - start_time:.1f}s)"
    )

    return start_time, end_time


def extract_song_title(filename: str) -> str:
    """Extract and format song title from filename."""
    # Remove extension and clean up
    title = Path(filename).stem

    # Handle common patterns
    title = title.replace("_", " ").replace("-", " ")

    # Capitalize properly
    title = " ".join(word.capitalize() for word in title.split())

    return title


def load_sidecar_config(video_path: Path) -> dict:
    toml_path = video_path.with_suffix(".toml")
    if toml_path.exists():
        print(f"Loading sidecar config: {toml_path}")
        with open(toml_path, "rb") as f:
            return tomllib.load(f)
    return {}


def process_video(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    title: str,
    format_type: str = "crop",
    subtitle: str = "",
):
    """
    Process video: trim, convert to vertical, add text overlay.

    format_type: "crop" (center crop) or "blur" (blurred background)
    """
    print(f"Processing video: {input_path}")
    print(f"Output: {output_path}")
    print(f"Segment: {start_time:.1f}s - {end_time:.1f}s")
    print(f"Title: {title}")
    if subtitle:
        print(f"Subtitle: {subtitle}")

    # Escape special characters for ffmpeg drawtext
    title_escaped = title.replace("'", "'\\''").replace(":", "\\:")

    if format_type == "crop":
        # Center crop to 9:16 aspect ratio
        # Assuming input is 16:9 (1920x1080), crop to 607x1080 (centered)
        video_filter = (
            f"crop=ih*9/16:ih,"
            f"scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
        )
    elif format_type == "blur":
        # Blurred background with centered video
        video_filter = (
            f"split[original][blurred];"
            f"[blurred]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,gblur=sigma=20[bg];"
            f"[original]scale=1080:608:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )
    else:
        raise ValueError(f"Unknown format type: {format_type}")

    # Text overlay filter
    text_filter = (
        f"drawtext=text='{title_escaped}':"
        f"fontsize=48:fontcolor=white:"
        f"x=(w-text_w)/2:y=h*0.25:"
        f"borderw=3:bordercolor=black:"
        f"font=sans-serif"
    )

    if subtitle:
        subtitle_escaped = subtitle.replace("'", "'\\''").replace(":", "\\:")
        text_filter += (
            f",drawtext=text='{subtitle_escaped}':"
            f"fontsize=32:fontcolor=white:"
            f"x=(w-text_w)/2:y=h*0.25+70:"
            f"borderw=2:bordercolor=black:"
            f"font=sans-serif"
        )

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-ss",
        str(start_time),
        "-i",
        input_path,
        "-t",
        str(end_time - start_time),
        "-vf",
        f"{video_filter},{text_filter}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",  # Optimize for web streaming
        "-y",  # Overwrite output
        output_path,
    ]

    print(f"Running ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error processing video:")
        print(result.stderr)
        sys.exit(1)

    print(f"✓ Created: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create Instagram-ready vertical videos from music performances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Alice In Wonderland.mp4"
  %(prog)s "My Solo.mp4" --duration 45 --format blur
  %(prog)s "Performance.mp4" --output "instagram_reel.mp4"
  %(prog)s "Full Set.mp4" --full-video
        """,
    )

    parser.add_argument("input", help="Input video file")
    parser.add_argument(
        "-o", "--output", help="Output video file (default: <input>_reel.mp4)"
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=30,
        help="Target duration of clip in seconds (default: 30)",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        help="Minimum clip duration in seconds (enables smart boundaries)",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        help="Maximum clip duration in seconds (enables smart boundaries)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["crop", "blur"],
        default="crop",
        help="Vertical format: crop (center crop) or blur (blurred background)",
    )
    parser.add_argument(
        "--title",
        help="Custom title (default: extracted from filename or sidecar .toml)",
    )
    parser.add_argument(
        "--subtitle", help="Subtitle text (e.g. '🎹 chorusing on Cherokee')"
    )
    parser.add_argument(
        "--start", type=float, help="Manual start time in seconds (skip auto-detection)"
    )
    parser.add_argument(
        "--end", type=float, help="Manual end time in seconds (skip auto-detection)"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Process all videos in the input directory"
    )
    parser.add_argument(
        "--full-video",
        action="store_true",
        help="Process entire video without cutting/analysis",
    )

    args = parser.parse_args()

    # If --full-video is used without explicit --format, use blur to preserve full frame
    if args.full_video and "--format" not in sys.argv and "-f" not in sys.argv:
        args.format = "blur"

    # Batch mode
    if args.batch:
        input_dir = Path(args.input)
        if not input_dir.is_dir():
            print(f"Error: With --batch, input must be a directory")
            sys.exit(1)

        video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV"}
        videos = [f for f in input_dir.iterdir() if f.suffix in video_extensions]

        if not videos:
            print(f"No video files found in {input_dir}")
            sys.exit(1)

        print(f"Found {len(videos)} videos to process\n")

        for video in videos:
            print(f"\n{'=' * 60}")
            print(f"Processing: {video.name}")
            print("=" * 60)

            output_path = video.parent / f"{video.stem}_reel.mp4"
            config = load_sidecar_config(video)
            title = config.get("title", extract_song_title(video.name))
            subtitle = config.get("subtitle", "")
            min_dur = args.min_duration or config.get("min_duration")
            max_dur = args.max_duration or config.get("max_duration")

            try:
                if args.full_video:
                    duration = get_video_duration(str(video))
                    start_time = 0
                    end_time = duration
                    print(f"Using full video: {duration:.1f}s")
                else:
                    start_time, end_time = analyze_audio(
                        str(video),
                        duration=args.duration,
                        min_duration=min_dur,
                        max_duration=max_dur,
                    )
                process_video(
                    str(video),
                    str(output_path),
                    start_time,
                    end_time,
                    title,
                    args.format,
                    subtitle,
                )
            except Exception as e:
                print(f"Error processing {video.name}: {e}")
                continue

        print(f"\n{'=' * 60}")
        print(f"✓ Batch processing complete!")
        return

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_reel.mp4"

    # Get title, subtitle, and duration overrides (CLI args override sidecar config)
    config = load_sidecar_config(input_path)
    title = args.title or config.get("title", extract_song_title(input_path.name))
    subtitle = args.subtitle or config.get("subtitle", "")
    min_duration = args.min_duration or config.get("min_duration")
    max_duration = args.max_duration or config.get("max_duration")

    # Determine segment
    if args.full_video:
        duration = get_video_duration(str(input_path))
        start_time = 0
        end_time = duration
        print(f"Using full video: {duration:.1f}s")
    elif args.start is not None and args.end is not None:
        start_time = args.start
        end_time = args.end
        print(f"Using manual segment: {start_time:.1f}s - {end_time:.1f}s")
    else:
        # Auto-detect best segment
        try:
            start_time, end_time = analyze_audio(
                str(input_path),
                duration=args.duration,
                min_duration=min_duration,
                max_duration=max_duration,
            )
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            print("Falling back to first 30 seconds...")
            start_time = 0
            end_time = args.duration

    # Process video
    process_video(
        str(input_path),
        str(output_path),
        start_time,
        end_time,
        title,
        args.format,
        subtitle,
    )

    print(f"\n✓ Done! Your Instagram reel is ready: {output_path}")


if __name__ == "__main__":
    main()
