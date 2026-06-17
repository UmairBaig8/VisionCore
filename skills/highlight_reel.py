"""
Highlight reel generator — extracts clips around key events and stitches them.
Uses ffmpeg if available, falls back to OpenCV.
"""

import json
import subprocess
import tempfile
from pathlib import Path

from core.paths import output_dir
from skills.frame_sampler import count_frames as _count


def _ffmpeg_available():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return True
    except Exception:
        return False


def _extract_clip_ffmpeg(video_path, start_sec, duration, out_path):
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
        "-c", "copy",
        str(out_path),
    ], check=True)


def _concat_clips_ffmpeg(clip_paths, out_path):
    concat_file = out_path.with_suffix(".txt")
    lines = [f"file '{p.resolve()}'" for p in clip_paths]
    concat_file.write_text("\n".join(lines))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(out_path),
    ], check=True)
    concat_file.unlink(missing_ok=True)


def _extract_clip_cv2(video_path, start_sec, duration, out_path):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
    frames_to_write = int(duration * fps)
    for _ in range(frames_to_write):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()


def generate_reel(video_path, key_events, video_name, clip_before=5.0, clip_after=3.0):
    """
    Generate a highlight reel from key_events timestamps.

    Args:
        video_path: path to source video
        key_events: list of dicts with 'timestamp' field (e.g. "57.2s")
        video_name: output filename stem
        clip_before: seconds before each event to include
        clip_after: seconds after each event to include
    Returns:
        Path to generated reel, or None if no events
    """
    if not key_events:
        return None

    use_ffmpeg = _ffmpeg_available()
    out_dir = output_dir() / "reels"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── extract timestamps ──
    timestamps = []
    for ev in key_events:
        ts = ev.get("timestamp", ev.get("global_time", ""))
        ts = ts.replace("s", "")
        try:
            timestamps.append(float(ts))
        except (ValueError, TypeError):
            continue

    if not timestamps:
        return None

    timestamps.sort()

    # ── merge overlapping events ──
    merged = []
    window = clip_before + clip_after
    for t in timestamps:
        if merged and t - merged[-1] < window:
            merged[-1] = t
        else:
            merged.append(t)

    # ── extract clips ──
    temp_dir = Path(tempfile.mkdtemp())
    clips = []
    for i, t in enumerate(merged):
        start = max(0, t - clip_before)
        duration = clip_before + clip_after
        clip_path = temp_dir / f"clip_{i:04d}.mp4"

        if use_ffmpeg:
            try:
                _extract_clip_ffmpeg(video_path, start, duration, clip_path)
                clips.append(clip_path)
            except Exception:
                continue
        else:
            _extract_clip_cv2(video_path, start, duration, clip_path)
            clips.append(clip_path)

    if not clips:
        return None

    # ── concat ──
    reel_path = out_dir / f"{video_name}_reel.mp4"

    if use_ffmpeg and len(clips) > 1:
        _concat_clips_ffmpeg(clips, reel_path)
    elif len(clips) == 1:
        clips[0].rename(reel_path)
    else:
        # cv2 concat fallback
        import cv2
        cap = cv2.VideoCapture(str(clips[0]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(reel_path), fourcc, fps, (w, h))
        for cp in clips:
            cap = cv2.VideoCapture(str(cp))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
            cap.release()
        writer.release()

    # clean up
    for cp in clips:
        cp.unlink(missing_ok=True)
    temp_dir.rmdir()

    return reel_path
