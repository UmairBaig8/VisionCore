import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import requests

from core.agent_loader import AgentLoader
from core.agent_router import AgentRouter
from core.config import load_config
from core.context import MatchContext
from core.emitter import EventEmitter
from core.llm_client import VLLMClient
from core.paths import agents_dir

from skills.frame_sampler import sample_frames, count_frames
from skills.live_sampler import sample_live, count_live_frames
from skills.frame_encoder import encode_frame
from skills.timeline import Timeline
from skills.video_loader import open_video
from skills.report_generator import save_report
from skills.csv_writer import save_csv
from skills.highlight_reel import generate_all_reels
from skills.live_reel import LiveReelBuilder

# lazy YOLO model — loaded once, runs on every frame for VLM cross-validation
_yolo_model = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO("yolo11n.pt")
        except Exception:
            _yolo_model = False
    return _yolo_model if _yolo_model is not False else None


def _analyze_frame_yolo(frame):
    """Run YOLO on frame, return structured data for VLM cross-check.
    Returns dict with: ball_xy, ball_zone, player_count, players_in_box, phase_hint."""
    model = _get_yolo()
    if model is None:
        return None
    try:
        results = model(frame, verbose=False)
        h, w = frame.shape[:2]
        ball_cx, ball_cy = -1, -1
        total_players = 0
        players_bottom = 0
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                y1 = float(box.xyxy[0][1])
                x1 = float(box.xyxy[0][0])
                x2 = float(box.xyxy[0][2])
                if cls == 0:  # person
                    total_players += 1
                    if y1 > h * 0.55:
                        players_bottom += 1
                elif cls == 32:  # sports ball
                    ball_cx = (x1 + x2) / 2
                    ball_cy = y1
        # ball zone
        if ball_cy < 0:
            zone = "not_visible"
        elif ball_cy < h * 0.25:
            zone = "far_end"
        elif ball_cy < h * 0.55:
            zone = "midfield"
        elif ball_cx > 0 and ball_cx < w * 0.35:
            zone = "left_box"
        elif ball_cx > 0 and ball_cx > w * 0.65:
            zone = "right_box"
        else:
            zone = "center_attacking"
        # phase hint from yolo
        if zone in ("left_box", "right_box"):
            phase_hint = "attack_final_third"
        elif zone == "midfield" and total_players >= 6:
            phase_hint = "open_play"
        elif total_players <= 3:
            phase_hint = "commercial_or_replay"
        else:
            phase_hint = "unknown"
        return {
            "ball_xy": f"({ball_cx:.0f},{ball_cy:.0f})" if ball_cy >= 0 else "not_found",
            "ball_zone": zone,
            "player_count": total_players,
            "players_in_box": players_bottom,
            "phase_hint": phase_hint,
            "has_goal_activity": zone in ("left_box", "right_box") and players_bottom >= 1,
        }
    except Exception:
        return None

logger = logging.getLogger("orchestrator")

MAX_RETRIES = 3
RETRY_BACKOFF = 2
CLASSIFY_FRAMES = 3


def _ask_with_retry(client, prompt, image_b64=None, label="llm"):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            result = client.ask(prompt, image_b64)
            elapsed = time.time() - t0
            logger.debug("  %s call took %.1fs", label, elapsed)
            return result
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF ** attempt
                print(f"  [retry {attempt}/{MAX_RETRIES} in {wait}s]", file=sys.stderr)
                time.sleep(wait)
        except (KeyError, requests.HTTPError) as exc:
            last_exc = exc
            break
    return None


def _format_events_for_summary(timeline):
    lines = []
    for i, ev in enumerate(timeline.events, 1):
        t = ev.get("timestamp", "?")
        result = ev.get("result", ev.get("description", ""))
        lines.append(f"## Event {i} [{t}]\n{result}\n")
    return "\n---\n".join(lines)


def _detect_geo(client, video_path):
    geo_prompt_path = agents_dir() / "geo_agent.md"
    if not geo_prompt_path.exists():
        return None
    geo_prompt = geo_prompt_path.read_text()
    # sample from middle of video, not the intro graphics
    cap = open_video(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    mid_frame = min(total // 2, int(fps * 30))
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    b64 = encode_frame(frame)
    if not b64:
        return None
    result = _ask_with_retry(client, geo_prompt, b64, label="geo")
    if result:
        return _parse_json_safe(result)
    return None


def _load_type_prompts():
    path = agents_dir() / "type_prompts.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _detect_sport(client, video_path):
    sport_prompt_path = agents_dir() / "sport_classifier.md"
    if not sport_prompt_path.exists():
        return {"sport": "generic", "confidence": 0.0}
    sport_prompt = sport_prompt_path.read_text()
    cap = open_video(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    off = min(total // 2, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, off)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return {"sport": "generic", "confidence": 0.0}
    b64 = encode_frame(frame)
    if not b64:
        return {"sport": "generic", "confidence": 0.0}
    result = _ask_with_retry(client, sport_prompt, b64, label="sport")
    if result:
        parsed = _parse_json_safe(result)
        return {"sport": parsed.get("sport", "generic"), "confidence": 0.8}
    return {"sport": "generic", "confidence": 0.0}


def _load_sport_events_prompt(sport):
    path = agents_dir() / "sports" / f"{sport}_events.md"
    if path.exists():
        return path.read_text()
    return None


def _parse_json_safe(raw_text):
    try:
        text = raw_text.strip()
        if "```" in text:
            for part in text.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{") and part.endswith("}"):
                    text = part
                    break
        return json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        return {}


def _classify_video(client, video_path, interval):
    classifier_prompt = (agents_dir() / "video_classifier.md").read_text()
    cap = open_video(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # sample 3 frames spread across the video for reliable classification
    positions = [
        int(fps * 15),                 # early (skip intro)
        int(total * 0.50),             # mid
        max(total - int(fps * 30), 0), # late (avoid credits)
    ]
    frames_b64 = []
    for pos in positions:
        pos = max(0, min(pos, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            b64 = encode_frame(frame)
            if b64:
                frames_b64.append(b64)
    cap.release()

    if not frames_b64:
        return {"video_type": "full_match", "confidence": 0.0,
                "evidence": "no frames extracted"}

    # send all sampled frames to classifier
    multi_prompt = (f"{classifier_prompt}\n\n"
                    f"You are shown {len(frames_b64)} frames from different timestamps. "
                    f"Consider patterns across ALL frames to classify the video type. "
                    f"Reply with JSON only.")
    result = _ask_with_retry(client, multi_prompt, frames_b64, label="classify")
    if result:
        parsed = _parse_json_safe(result)
        return {"video_type": parsed.get("video_type", "full_match"),
                "confidence": parsed.get("confidence", 0.8),
                "evidence": parsed.get("evidence", "")}
    return {"video_type": "full_match", "confidence": 0.0,
            "evidence": "no response"}


def _build_report_header(ctx, video_stem):
    if ctx is None:
        return f"# {video_stem}\n\n"

    header = [f"# {video_stem}\n"]
    header.append(f"**Type:** {ctx.video_type}")
    header.append(f"**Sport:** {ctx.sport}")
    header.append(f"**Score:** {ctx.score_string()}")

    if ctx.location:
        header.append(f"**Location:** {ctx.location}")
    if ctx.league:
        header.append(f"**League:** {ctx.league}")
    if ctx.teams:
        header.append(f"**Teams:** {', '.join(ctx.teams)}")

    header.append("")

    if ctx.key_events:
        header.append(f"## Key Events ({len(ctx.key_events)})\n")
        for ev in ctx.key_events:
            ts = ev.get("timestamp", "?")
            et = ev.get("type", "event")
            team = ev.get("team", ev.get("batsman", ev.get("player", "")))
            runs = ev.get("runs", ev.get("points", ""))
            detail = f" — {team}" if team else ""
            detail += f" ({runs})" if runs else ""
            header.append(f"- **[{ts}]** {et}{detail}")
        header.append("")

    return "\n".join(header)


def _update_context_from_scene(ctx, scene_desc, skip_score=False):
    if not scene_desc or not ctx:
        return

    # ── phase detection from keywords ──
    if any(w in scene_desc.lower() for w in ("graphic", "promotional", "abstract",
                                                "static", "triangles", "geometric",
                                                "subscribe")):
        if ctx.phase != "commercial":
            ctx.update_phase("commercial")
    elif any(w in scene_desc.lower() for w in ("goal scored", "celebrating",
                                                  "goalkeeper diving", "shot on goal",
                                                  "goal attempt")):
        if ctx.phase != "attack_final_third":
            ctx.update_phase("attack_final_third")
    elif any(w in scene_desc.lower() for w in ("half time", "halftime")):
        ctx.update_phase("half_time")
    elif any(w in scene_desc.lower() for w in ("dribbling", "passing", "midfield",
                                                  "possession", "advancing")):
        if ctx.phase in ("kickoff", "commercial", "unknown", "attack_final_third"):
            ctx.update_phase("open_play")

    if skip_score:
        return

    # ── score extraction: match score-like patterns in scene description ──
    score_patterns = [
        r'\bscore\S*\s*[:=]?\s*(\d+)\s*[-–]\s*(\d+)\b',
        r'\b(?:leads?|leading|winning|trailing|tied)\s+(\d+)\s*[-–]\s*(\d+)\b',
        r'\b([A-Z]{2,4})\s+(\d+)\s*[-–]\s*(\d+)\s+\1\b',
        r'\b(?:currently|now|still)\s+(\d+)\s*[-–]\s*(\d+)\b',
    ]
    for pat in score_patterns:
        m = re.search(pat, scene_desc, re.IGNORECASE)
        if m:
            try:
                groups = m.groups()
                if len(groups) == 3:
                    h, a = int(groups[1]), int(groups[2])
                else:
                    h, a = int(groups[0]), int(groups[1])
                if (h, a) != (ctx.home_score, ctx.away_score):
                    if h >= ctx.home_score and a >= ctx.away_score:
                        ctx.home_score = h
                        ctx.away_score = a
                        ctx.last_score_change = "detected_from_scene"
            except (ValueError, IndexError):
                pass
            break


class VideoOrchestrator:

    def __init__(self, video_path, sample_interval=0.5,
                 depth="full", stream_mode=False, report_only=False,
                 live=False, classify=True, location=None,
                 verbose=False, generate_reel_flag=False,
                 clip_before=8.0, clip_after=5.0,
                 player_filter=None, team_filter=None,
                 emitter=None):
        self.video_path = video_path
        self.sample_interval = sample_interval
        self.depth = depth
        self.stream_mode = stream_mode
        self.report_only = report_only
        self.live = live
        self.classify = classify
        self.location = location
        self.verbose = verbose
        self.generate_reel = generate_reel_flag
        self.clip_before = clip_before
        self.clip_after = clip_after
        self.player_filter = player_filter
        self.team_filter = team_filter
        self.emitter = emitter or EventEmitter()
        self.ctx = None

    def _run_parallel(self, client, tasks):
        results = {}
        if not tasks:
            return results
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(_ask_with_retry, client, p, i): k
                       for k, (p, i) in tasks.items()}
            for f in as_completed(futures):
                key = futures[f]
                try:
                    results[key] = f.result()
                except Exception as e:
                    results[key] = None
        return results

    def _vprint(self, msg):
        if self.verbose:
            print(f"  [router] {msg}", file=sys.stderr)

    def _emit_agent(self, name):
        self.emitter.on_agent_active(name)

    def run(self):
        cfg = load_config()
        agents = AgentLoader().load()
        client = VLLMClient(cfg["vllm_endpoint"], cfg["model"])
        timeline = Timeline()

        # ── count frames first (instant, no VLM) — emit progress bar immediately ──
        if self.live:
            total_frames = count_live_frames(self.video_path, self.sample_interval)
        else:
            total_frames = count_frames(self.video_path, self.sample_interval)
        video_stem = Path(self.video_path).stem
        self.emitter.on_progress(0, total_frames, 0)
        logger.info("video: %s, %d frames to process", video_stem, total_frames)

        # ── detection phase (parallel — 3x faster) ──
        from concurrent.futures import ThreadPoolExecutor as TPE, as_completed as ac

        def _run_classify():
            if self.classify:
                return _classify_video(client, self.video_path,
                                       max(self.sample_interval * 4, 2.0))
            return {"video_type": "full_match", "confidence": 1.0}

        def _run_geo():
            if self.location:
                return {"stadium": self.location, "source": "manual"}
            return _detect_geo(client, self.video_path) or {}

        def _run_sport():
            return _detect_sport(client, self.video_path)

        t_detect = time.time()
        logger.info("detection phase starting (classify+geo+sport in parallel)")
        with TPE(max_workers=3) as pool:
            f_cls = pool.submit(_run_classify)
            f_geo = pool.submit(_run_geo)
            f_spt = pool.submit(_run_sport)
            video_type = f_cls.result()
            geo = f_geo.result()
            sport_info = f_spt.result()

        vt = video_type["video_type"]
        sport_id = sport_info.get("sport", "generic")
        logger.info("detection phase complete in %.1fs", time.time() - t_detect)
        logger.info("  classify=%s (%.0f%%), geo=%s, sport=%s (%.0f%%)",
                     vt, video_type["confidence"] * 100,
                     geo.get("stadium", geo.get("city", "") or "unknown"),
                     sport_id, sport_info.get("confidence", 0) * 100)
        print(f" → {vt} ({video_type['confidence']:.0%}) | "
              f"{geo.get('stadium', geo.get('city', '') or 'unknown')} | "
              f"{sport_id} ({sport_info.get('confidence', 0):.0%})")

        self.emitter.on_detection(
            sport=sport_id,
            video_type=vt,
            location=geo.get("stadium", geo.get("city", "")),
            league=geo.get("league", ""),
            teams=geo.get("teams", []),
        )

        # ── build context ──
        self.ctx = MatchContext(
            sport=sport_id,
            video_type=video_type["video_type"],
            teams=geo.get("teams", []),
            location=geo.get("stadium", geo.get("city", "")),
            league=geo.get("league", ""),
        )
        router = AgentRouter(self.ctx)

        # ── load prompts ──
        type_prompts = _load_type_prompts().get(video_type["video_type"], {})
        sport_events_prompt = _load_sport_events_prompt(sport_id)

        scene_prompt = agents.get("scene_detector", "")
        event_prompt = sport_events_prompt or agents.get("event_detector", "")
        commentary_prompt = agents.get("commentary_agent", "")
        reasoning_prompt = agents.get("reasoning_agent", "")
        summary_prompt = type_prompts.get("summary_prompt") or agents.get("summary_agent", "")
        highlight_prompt = type_prompts.get("highlight_prompt") or agents.get("highlight_agent", "")

        do_event = self.depth in ("fast", "full")
        do_analysis = self.depth == "full"

        if self.live:
            sampler = sample_live
        else:
            sampler = sample_frames

        processed = 0

        # emit detection complete BEFORE frame loop starts
        self.emitter.on_detection_complete(
            sport=sport_id, video_type=vt,
            location=geo.get("stadium", geo.get("city", "")),
            league=geo.get("league", ""),
            teams=geo.get("teams", []),
            total_frames=total_frames,
        )

        # live mode: progress is wall-clock, not frame count
        video_duration = None
        if self.live:
            video_duration = total_frames * self.sample_interval

        # ── live reel builder (generates clips as events happen) ──
        live_reel = None
        if self.generate_reel:
            live_reel = LiveReelBuilder(self.video_path, video_stem,
                                        clip_before=self.clip_before,
                                        clip_after=self.clip_after)

        # ── frame loop ──
        scoreboard_prompt = None
        scoreboard_path = agents_dir() / "scoreboard_agent.md"
        if scoreboard_path.exists():
            scoreboard_prompt = scoreboard_path.read_text()

        # scoreboard history buffer: only accept new score after 2 consistent readings
        sb_history = []

        # goal confirmation buffer: require 2 frames within 8s to confirm a GOAL
        pending_goals = {}  # {goal_hash: {"event": ev, "first_ts": ts, "count": n}}

        for timestamp, frame in sampler(self.video_path, self.sample_interval):
            t_start = time.time()
            processed += 1
            sig_events = []
            key_events = []
            reasoning_str = None
            commentary_str = None
            image_b64 = encode_frame(frame)
            if image_b64 is None:
                continue

            # step 0: YOLO analysis (fast, every 3rd frame to avoid overhead)
            t_yolo = 0.0
            yolo = None
            if processed % 3 == 1:
                t_yolo = time.time()
                yolo = _analyze_frame_yolo(frame)
                t_yolo = time.time() - t_yolo
                if yolo:
                    self.emitter.on_yolo_frame(yolo["ball_zone"], yolo["player_count"], yolo["phase_hint"])

            # step 1: scene + event + scoreboard ALL in parallel (3 independent vLLM calls)
            yolo_hint = ""
            if yolo:
                yolo_hint = (f"\n\n[YOLO: ball={yolo['ball_zone']} "
                             f"players={yolo['player_count']} "
                             f"phase={yolo['phase_hint']}]")

            parallel_tasks = {"scene": (scene_prompt + yolo_hint, image_b64)}

            # event detection: runs AFTER scene (needs scene_desc for accuracy)
            # scoreboard: every 5th frame — rotating region + full frame fallback
            sb_applied = False
            sb_b64 = None
            if scoreboard_prompt and processed % 5 == 0:
                fh, fw = frame.shape[:2]
                crops = [
                    frame[0:int(fh * 0.15), :],                  # top bar
                    frame[int(fh * 0.85):fh, :],                  # bottom bar
                ]
                crop_idx = (processed // 5) % len(crops)
                crop = crops[crop_idx]
                if crop.size > 0:
                    sb_b64 = encode_frame(crop)
                # every 15th frame: try full frame as fallback for tricky scoreboards
                if processed % 15 == 0:
                    sb_b64 = image_b64
            if sb_b64:
                parallel_tasks["scoreboard"] = (scoreboard_prompt, sb_b64)

            t_parallel = time.time()
            parallel_results = self._run_parallel(client, parallel_tasks)
            t_parallel = time.time() - t_parallel

            scene_desc = parallel_results.get("scene")
            t_scene = t_parallel
            self._emit_agent("scene")
            if scene_desc is None:
                continue

            # process scoreboard result
            sb_applied = False
            sb_result = parallel_results.get("scoreboard")
            if sb_result:
                self._emit_agent("scoreboard")
                sb_parsed = _parse_json_safe(sb_result)
                sb_score = sb_parsed.get("score", "")
                sb_conf = sb_parsed.get("confidence", 0)
                if sb_score and sb_score != "NO_SCOREBOARD" and sb_conf >= 0.5:
                    try:
                        parts = sb_score.strip().split("-")
                        if len(parts) == 2:
                            sh, sa = int(parts[0].strip()), int(parts[1].strip())
                            jump = (sh - self.ctx.home_score) + (sa - self.ctx.away_score)
                            sb_history.append((sh, sa, sb_conf))
                            if len(sb_history) > 6:
                                sb_history.pop(0)
                            min_conf = 0.50 if jump <= 1 else 0.85
                            required_consensus = 1 if jump <= 1 else 2
                            if sh == 0 and sa == 0:
                                pass
                            else:
                                if self.ctx.home_score == 0 and self.ctx.away_score == 0:
                                    required_consensus = 1
                                    min_conf = 0.50
                                consistent = sum(1 for hs, a_, _ in sb_history[-required_consensus:]
                                                if (hs, a_) == (sh, sa))
                                if sb_conf >= min_conf and consistent >= required_consensus:
                                    if sh >= self.ctx.home_score and sa >= self.ctx.away_score:
                                        if (sh, sa) != (self.ctx.home_score, self.ctx.away_score):
                                            self.ctx.home_score = sh
                                            self.ctx.away_score = sa
                                            self.ctx.last_score_change = "scoreboard"
                                            self.emitter.on_score_change(sh, sa)
                                            sb_applied = True
                                            logger.debug("  scoreboard %d-%d", sh, sa)
                    except (ValueError, IndexError):
                        pass

            # extract score + phase from scene description
            old_score = self.ctx.score_string()
            old_phase = self.ctx.phase
            _update_context_from_scene(self.ctx, scene_desc, skip_score=sb_applied)
            if self.ctx.score_string() != old_score:
                self.emitter.on_score_change(self.ctx.home_score, self.ctx.away_score)

            # step 2: event detection (sequential — needs scene_desc for accuracy)
            key_events = []
            t_event = 0.0
            if do_event and event_prompt:
                route = router.route(scene_desc, processed)
                skip_event = (route["reason"].startswith("skip:") and not (
                    yolo and yolo.get("has_goal_activity")))
                if not skip_event:
                    t0 = time.time()
                    yolo_ctx = f"\n[YOLO: {yolo['ball_zone']}]" if yolo else ""
                    event_str = _ask_with_retry(
                        client, f"{event_prompt}\n\nFrame: {scene_desc}{yolo_ctx}",
                        label="event")
                    t_event = time.time() - t0
                    self._emit_agent("event")
                    logger.info("event vllm returned %d chars", len(event_str) if event_str else 0)
                    if event_str:
                        parsed = _parse_json_safe(event_str)
                        events = parsed.get("events", [])
                        if not events:
                            logger.debug("event detector returned empty events list")
                        for e in events:
                            e["timestamp"] = f"{timestamp:.1f}s"
                        seen_types = set()
                        deduped = []
                        for e in events:
                            et = e.get("type", "")
                            if et not in seen_types:
                                seen_types.add(et)
                                deduped.append(e)
                        if deduped:
                            key_events = router.process_event(deduped)
                        else:
                            key_events = []
                        # dedup within ctx history
                        ctx_evs = self.ctx.key_events
                        if len(ctx_evs) >= 2:
                            prev = ctx_evs[-1]
                            curr = key_events[0] if key_events else None
                            if curr and prev.get("type") == "GOAL_ATTEMPT" and curr.get("type") == "GOAL":
                                try:
                                    pt = float(prev.get("timestamp", "0").replace("s", ""))
                                    ct = float(curr.get("timestamp", "0").replace("s", ""))
                                    if ct - pt < 4.0:
                                        ctx_evs[-1] = curr
                                        ctx_evs[-1]["timestamp"] = prev["timestamp"]
                                        key_events = []
                                except (ValueError, TypeError):
                                    pass
                        # dedup same-type events within 8s
                        recent_window = 8.0
                        try:
                            current_ts = float(timestamp)
                        except (ValueError, TypeError):
                            current_ts = 0.0
                        for ev in list(key_events):
                            et = ev.get("type", "")
                            if et == "GOAL_ATTEMPT":
                                continue
                            for past in reversed(ctx_evs):
                                if past.get("type") == et:
                                    try:
                                        pt = float(past.get("timestamp", "0").replace("s", ""))
                                        if abs(current_ts - pt) < recent_window:
                                            key_events.remove(ev)
                                            break
                                    except (ValueError, TypeError):
                                        pass
                        if "possession_home" in parsed:
                            self.ctx.update_momentum(int(parsed["possession_home"]))
                        if "phase" in parsed:
                            self.ctx.update_phase(parsed["phase"])
                        # GOAL validation + cross-frame confirmation
                        sig_events = [e for e in key_events if e.get("type") != "GOAL_ATTEMPT"]
                        goal_confirm = ["celebration", "celebrating", "arms raised",
                                        "sliding on knees", "fist pump", "hugging",
                                        "players running", "crowd", "jumping"]
                        goal_miss = ["missed", "wide", "over the bar", "saved",
                                     "blocked", "cleared", "deflected", "goalkeeper saves",
                                     "hands on head", "disappointed", "nearly", "almost",
                                     "outside the box", "side netting"]
                        has_goal_context = (sum(1 for kw in goal_confirm if kw in scene_desc.lower()) >= 2
                                            and sum(1 for kw in goal_miss if kw in scene_desc.lower()) == 0)
                        validated = []
                        for ev in sig_events:
                            if ev.get("type") == "GOAL":
                                yolo_ok = (yolo is None) or yolo.get("has_goal_activity", True)
                                if has_goal_context and yolo_ok:
                                        gk = f"{ev.get('team','?')}_{ev.get('player','?')}"
                                        if gk in pending_goals:
                                            pending_goals[gk]["count"] += 1
                                            validated.append(ev)
                                            logger.info("  GOAL confirmed: %s (frame %d)", gk, pending_goals[gk]["count"])
                                            if pending_goals[gk]["count"] >= 1:
                                                del pending_goals[gk]
                                        else:
                                            pending_goals[gk] = {"event": ev, "first_ts": timestamp, "count": 1}
                                            validated.append(ev)  # accept first frame as valid
                                            logger.info("  GOAL: %s (first detection)", gk)
                                else:
                                    ev = dict(ev, type="GOAL_ATTEMPT")
                                    validated.append(ev)
                            else:
                                validated.append(ev)
                        sig_events = validated
                        for gk in list(pending_goals.keys()):
                            try:
                                if timestamp - pending_goals[gk]["first_ts"] > 10.0:
                                    del pending_goals[gk]
                            except (TypeError, KeyError):
                                pass
                        if key_events:
                            logger.info("frame events: %s", [(e.get("type"), e.get("team", "?")) for e in key_events])

            if self.verbose and route:
                self._vprint(f"frame={processed} phase={self.ctx.phase} "
                             f"({route.get('reason','?')})")

            t_analysis = 0.0
            if do_analysis and sig_events:
                parallel_tasks = {}
                if reasoning_prompt:
                    parallel_tasks["reasoning"] = (
                        f"{reasoning_prompt}\n\n"
                        f"Event: {json.dumps(sig_events)}",
                        None
                    )
                if commentary_prompt:
                    parallel_tasks["commentary"] = (
                        f"{commentary_prompt}\n\n"
                        f"Event: {json.dumps(sig_events)}",
                        None
                    )
                if parallel_tasks:
                    t0 = time.time()
                    r = self._run_parallel(client, parallel_tasks)
                    reasoning_str = r.get("reasoning")
                    commentary_str = r.get("commentary")
                    t_analysis = time.time() - t0
                    self._emit_agent("reasoning")
                    self._emit_agent("commentary")

            # ── immediate clip generation for live reel (significant events only) ──
            if live_reel and sig_events:
                for ev in sig_events:
                    clip = live_reel.add_event(ev)
                    if clip:
                        self.emitter.on_clip_generated(
                            clip["event_type"], f"{clip['timestamp']}s",
                            clip["path"], len(live_reel.clips))

            # emit phase change if it happened this frame
            if self.ctx.phase != old_phase:
                self.emitter.on_phase_change(self.ctx.phase)

            scene_parsed = _parse_json_safe(scene_desc)

            event_dict = {
                "timestamp": f"{timestamp:.1f}s",
                "scene": scene_desc,
                "scene_type": scene_parsed.get("scene_type", "unknown"),
                "phase": self.ctx.phase,
                "score": self.ctx.score_string(),
                "key_events": json.dumps(key_events) if key_events else "",
                "event": event_str or "",
                "reasoning": reasoning_str or "",
                "commentary": commentary_str or "",
                "result": scene_desc,
            }

            timeline.add(event_dict)

            # ── emitter events ──
            stype = scene_parsed.get("scene_type", "unknown")
            self.emitter.on_scene(f"{timestamp:.1f}s", stype,
                                  scene_parsed.get("activity", ""),
                                  scene_desc)

            ball_pos = scene_parsed.get("ball_position", "")
            if ball_pos and ball_pos != "not_visible":
                self.emitter.on_ball_position(ball_pos, f"{timestamp:.1f}s")

            if key_events:
                for ev in key_events:
                    self.emitter.on_key_event(ev)

            if self.stream_mode and self.verbose:
                prefix = ""
                if key_events:
                    prefix = " ".join(f"[{e['type']}]" for e in key_events)
                short = _parse_json_safe(scene_desc)
                stype = short.get("scene_type", "")
                activity = short.get("activity", "")
                summary = f"{stype}: {activity}" if stype else scene_desc[:200]
                print(f"\n[{timestamp:.1f}s] {prefix} {summary}")
                if event_str:
                    print(f"  * {event_str[:250]}")
                if commentary_str:
                    print(f"  > {commentary_str[:250]}")
                print()
            elif not self.report_only and not self.stream_mode:
                bar_len = 30
                done = int(bar_len * processed / max(total_frames, 1))
                bar = f"[{'#' * done}{'-' * (bar_len - done)}]"
                pct = min(timestamp / video_duration * 100, 100) if video_duration else processed / max(total_frames, 1) * 100
                events_str = ""
                if key_events:
                    events_str = " | " + ",".join(e["type"] for e in key_events[:3])
                yolo_tag = f" [{yolo['ball_zone']}]" if yolo else ""
                print(f"\r  {bar} {pct:.0f}% ({processed}/{total_frames}) "
                      f"{self.ctx.phase} {self.ctx.score_string()}{yolo_tag}{events_str}",
                      end="", flush=True)

            pct = int(min(timestamp / video_duration * 100, 100)) if video_duration else int(processed / max(total_frames, 1) * 100)
            self.emitter.on_progress(processed, total_frames, pct)

            elapsed = time.time() - t_start
            parts = [f"vllm={elapsed:.1f}s", f"yolo={t_yolo:.1f}s", f"scene+t={t_scene:.1f}s"]
            if t_event:
                parts.append(f"event={t_event:.1f}s")
            if t_analysis:
                parts.append(f"reason+comm={t_analysis:.1f}s")
            if key_events:
                types = ",".join(e["type"] for e in key_events[:5])
                parts.append(f"events=[{types}]")
            logger.info("frame %3d/%d | %s", processed, total_frames, " ".join(parts))

        if not self.stream_mode and not self.report_only:
            print()

        video_stem = Path(self.video_path).stem
        csv_path = save_csv(timeline.events, video_stem, self.ctx)

        # ── emit analysis complete BEFORE reel generation so UI doesn't look stuck ──
        self.emitter.on_analysis_complete(
            len(self.ctx.key_events), self.ctx.score_string())

        # ── summary generation ──
        highlights = ""
        if highlight_prompt and self.ctx.key_events:
            highlights = _ask_with_retry(
                client,
                f"{highlight_prompt}\n\n"
                f"Key Events:\n{json.dumps(self.ctx.key_events, indent=2)}",
                label="highlights"
            ) or ""

        summary_payload = (
            f"{summary_prompt}\n\n"
            f"Match Context: {json.dumps(self.ctx.summary(), indent=2)}\n\n"
            f"Timeline:\n{_format_events_for_summary(timeline)}"
        )
        if highlights:
            summary_payload += f"\n\nHighlights:\n{highlights}"

        final_summary = _ask_with_retry(client, summary_payload, label="summary")
        if final_summary is None:
            print("Summary generation failed", file=sys.stderr)
            final_summary = "Summary unavailable"

        header = _build_report_header(self.ctx, video_stem)
        if highlights:
            header += f"\n## Highlights\n\n{highlights}\n\n---\n\n"

        report_path = save_report(header + final_summary, video_stem)

        reel_paths = {}
        if self.generate_reel:
            if live_reel:
                live_path = live_reel.finalize()
                if live_path:
                    reel_paths["live"] = str(live_path)
                    print(f"\nLive reel: {live_path} ({len(live_reel.clips)} clips generated during analysis)")

            if self.ctx.key_events:
                flavors = ["all", "goals", "drama", "social_goals"]
                if self.player_filter:
                    flavors = []
                if self.team_filter:
                    flavors = []

                print("Generating pro reels:")
                reel_flavors = flavors or ["custom"]
                for idx, flavor in enumerate(reel_flavors, 1):
                    self.emitter.on_reel_progress(flavor, idx, len(reel_flavors))
                more_paths = generate_all_reels(
                    self.video_path, self.ctx.key_events, video_stem,
                    flavors=flavors or None,
                    clip_before=self.clip_before,
                    clip_after=self.clip_after,
                    player=self.player_filter,
                    team=self.team_filter,
                )
                reel_paths.update(more_paths)
                for idx, flavor in enumerate(reel_flavors, 1):
                    self.emitter.on_reel_progress(flavor, idx, len(reel_flavors))

        if not self.report_only:
            print(f"\nType:   {self.ctx.video_type}  |  Sport: {self.ctx.sport}")
            print(f"Score:  {self.ctx.score_string()}  |  Events: {len(self.ctx.key_events)}")
            print(f"CSV:    {csv_path}")
            print(f"Report: {report_path}")
            if reel_paths:
                for flavor, path in reel_paths.items():
                    print(f"Reel:   {path} ({flavor})")
        else:
            print(report_path)

        self.emitter.on_complete(str(report_path), str(csv_path),
                                 reel_paths, len(self.ctx.key_events))

        return report_path
