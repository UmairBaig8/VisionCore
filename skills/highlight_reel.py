"""
Highlight reel generator — extracts clips around key events, filters by type/team,
supports landscape (16:9) and social/vertical (9:16) formats.
"""

import json
import subprocess
import tempfile
from pathlib import Path

from core.paths import output_dir

FLAVORS = {
    "all":    {"label": "Full Highlights",       "types": None, "format": "landscape"},
    "goals":  {"label": "Goals Only",            "types": {"GOAL"}, "format": "landscape"},
    "cards":  {"label": "Cards & Fouls",         "types": {"YELLOW_CARD", "RED_CARD", "FOUL", "PENALTY"}, "format": "landscape"},
    "drama":  {"label": "Drama Moments",         "types": {"GOAL", "YELLOW_CARD", "RED_CARD", "PENALTY", "VAR_CHECK", "INJURY"}, "format": "landscape"},
    "saves":  {"label": "Saves & Blocks",        "types": {"SAVE", "GOAL_ATTEMPT", "BLOCK"}, "format": "landscape"},
    "social": {"label": "Social (9:16)",         "types": None, "format": "vertical"},
    "social_goals": {"label": "Goal Reel (9:16)", "types": {"GOAL"}, "format": "vertical"},
}


def _ffmpeg_available():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return True
    except Exception:
        return False


_FFMPEG_CHECKED = False
_FFMPEG_OK = False


def _check_ffmpeg():
    global _FFMPEG_CHECKED, _FFMPEG_OK
    if not _FFMPEG_CHECKED:
        _FFMPEG_OK = _ffmpeg_available()
        _FFMPEG_CHECKED = True
        if not _FFMPEG_OK:
            print("\n  ⚠ ffmpeg not found — reels may be unplayable."
                  "\n  Install: apt install ffmpeg  (Ubuntu)  or  brew install ffmpeg  (macOS)")
    return _FFMPEG_OK


def _extract_clip_ffmpeg(video_path, start_sec, duration, out_path, crop_vertical=False):
    vf = ""
    if crop_vertical:
        # crop center 9:16 from source — use 60% width to capture more frame
        vf = "crop=iw*0.60:ih,scale=1080:1920"

    args = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start_sec),
        "-i", str(video_path),
        "-t", str(duration),
    ]
    if vf:
        args += ["-vf", vf]
    args += ["-c:a", "copy", str(out_path)]
    subprocess.run(args, check=True)


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


def _filter_events(key_events, event_types=None, team=None, player=None):
    filtered = []
    for ev in key_events:
        et = ev.get("type", "")
        if event_types and et not in event_types:
            continue
        t = ev.get("team", ev.get("batsman", ev.get("player", ""))).lower()
        if team and team.lower() not in t:
            continue
        if player and player.lower() not in t:
            continue
        filtered.append(ev)
    return filtered


def _parse_timestamps(key_events):
    timestamps = []
    for ev in key_events:
        ts = ev.get("timestamp", ev.get("global_time", ""))
        ts = ts.replace("s", "")
        try:
            timestamps.append(float(ts))
        except (ValueError, TypeError):
            continue
    return sorted(timestamps)


def _merge_overlapping(timestamps, window=8.0):
    merged = []
    for t in timestamps:
        if merged and t - merged[-1] < window:
            merged[-1] = t
        else:
            merged.append(t)
    return merged


def _add_title_card(reel_path, title, subtitle="", duration=3.0):
    """Overlay title text on first N seconds of reel via ffmpeg drawtext."""
    import tempfile
    if not _ffmpeg_available():
        return reel_path

    tmp = Path(tempfile.mktemp(suffix=".mp4"))
    title_esc = title.replace("'", "'\\\\''").replace(":", "\\:")
    sub_esc = subtitle.replace("'", "'\\\\''").replace(":", "\\:")

    vf = (
        f"drawtext=fontsize=42:fontcolor=white:"
        f"text='{title_esc}':x=(w-text_w)/2:y=(h-text_h)/2-25:"
        f"enable='between(t,0,{duration})',"
        f"drawtext=fontsize=24:fontcolor=#aaaaaa:"
        f"text='{sub_esc}':x=(w-text_w)/2:y=(h-text_h)/2+25:"
        f"enable='between(t,0,{duration})'"
    )

    try:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(reel_path),
            "-vf", vf,
            "-c:a", "copy",
            str(tmp),
        ], check=True)
        tmp.replace(reel_path)
    except Exception:
        pass
    return reel_path


def generate_reel(video_path, key_events, video_name,
                  flavor="all", clip_before=8.0, clip_after=5.0,
                  event_types=None, player=None, team=None,
                  title=None, subtitle=None):
    """Generate a single highlight reel.
    
    Goals get +2s extra before (to capture build-up play).
    event_types overrides flavor defaults. player/team filter events."""
    if isinstance(key_events, str):
        try:
            key_events = json.loads(key_events)
        except json.JSONDecodeError:
            return None

    flavor_cfg = FLAVORS.get(flavor, FLAVORS["all"])
    types = event_types or flavor_cfg.get("types")

    attack_types = {"GOAL", "GOAL_ATTEMPT", "PENALTY", "DUNK"}
    goal_boost = 2.0 if types and (types & attack_types) else 0

    filtered = _filter_events(key_events, types, team=team, player=player)
    if not filtered:
        return None

    timestamps = _merge_overlapping(_parse_timestamps(filtered),
                                    window=clip_before + clip_after + goal_boost + 2)
    if not timestamps:
        return None

    use_ffmpeg = _check_ffmpeg()
    is_vertical = flavor_cfg["format"] == "vertical"

    out_dir = output_dir() / "reels"
    out_dir.mkdir(parents=True, exist_ok=True)
    reel_path = out_dir / f"{video_name}_{flavor}.mp4"

    temp_dir = Path(tempfile.mkdtemp())
    clips = []

    for i, t in enumerate(timestamps):
        start = max(0, t - clip_before - goal_boost)
        duration = clip_before + clip_after + goal_boost
        clip_path = temp_dir / f"clip_{i:04d}.mp4"

        try:
            if use_ffmpeg:
                _extract_clip_ffmpeg(video_path, start, duration, clip_path,
                                     crop_vertical=is_vertical)
            else:
                _extract_clip_cv2(video_path, start, duration, clip_path)
            clips.append(clip_path)
        except Exception:
            continue

    if not clips:
        return None

    if use_ffmpeg:
        _concat_clips_ffmpeg(clips, reel_path)
    else:
        _concat_clips_cv2(clips, reel_path)

    for cp in clips:
        cp.unlink(missing_ok=True)
    temp_dir.rmdir()

    if title:
        _add_title_card(reel_path, title, subtitle or "")
    return reel_path


def generate_all_reels(video_path, key_events, video_name, flavors=None,
                       clip_before=8.0, clip_after=5.0,
                       player=None, team=None):
    """Generate multiple reels from one analysis pass.
    If player or team is set, generates a single filtered reel."""
    if player or team:
        results = {}
        label = player or team or "custom"
        print(f"  {label.capitalize()} reel", end="", flush=True)
        path = generate_reel(video_path, key_events, f"{video_name}_{label}",
                             flavor="all", clip_before=clip_before,
                             clip_after=clip_after,
                             event_types=None, player=player, team=team,
                             title=f"{label.upper()} Highlights",
                             subtitle=f"{video_name} | VidCore AI")
        if path:
            print(f" → {path.name}")
            results[label] = str(path)
        else:
            print(" → no matching events")
        return results

    results = {}
    for flavor in flavors:
        print(f"  {FLAVORS[flavor]['label']}", end="", flush=True)
        cfg = FLAVORS[flavor]
        path = generate_reel(video_path, key_events, video_name, flavor=flavor,
                               clip_before=clip_before, clip_after=clip_after,
                               title=cfg['label'],
                               subtitle=f"{video_name} | VidCore AI")
        if path:
            print(f" → {path.name}")
            results[flavor] = str(path)
        else:
            print(" → no events")
    return results


# ── cv2 fallbacks ──

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


def _concat_clips_cv2(clip_paths, out_path):
    import cv2
    cap = cv2.VideoCapture(str(clip_paths[0]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for cp in clip_paths:
        cap = cv2.VideoCapture(str(cp))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
    writer.release()
