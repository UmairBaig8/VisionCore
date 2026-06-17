import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from core.agent_loader import AgentLoader
from core.config import load_config
from core.llm_client import VLLMClient
from core.paths import agents_dir

from skills.frame_sampler import sample_frames, count_frames
from skills.live_sampler import sample_live, count_live_frames
from skills.frame_encoder import encode_frame
from skills.timeline import Timeline
from skills.report_generator import save_report
from skills.csv_writer import save_csv

MAX_RETRIES = 3
RETRY_BACKOFF = 2
CLASSIFY_FRAMES = 3


def _ask_with_retry(client, prompt, image_b64=None):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.ask(prompt, image_b64)
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
    for ts, frame in sample_frames(video_path, 5.0):
        b64 = encode_frame(frame)
        if not b64:
            continue
        result = _ask_with_retry(client, geo_prompt, b64)
        if result:
            try:
                text = result.strip()
                if "```" in text:
                    for part in text.split("```"):
                        part = part.strip()
                        if part.startswith("json"):
                            part = part[4:]
                        if part.startswith("{") and part.endswith("}"):
                            text = part
                            break
                return json.loads(text)
            except (json.JSONDecodeError, AttributeError):
                return None
        break
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
    for ts, frame in sample_frames(video_path, 3.0):
        b64 = encode_frame(frame)
        if not b64:
            continue
        result = _ask_with_retry(client, sport_prompt, b64)
        if result:
            parsed = _parse_classification(result)  # reuses same JSON parser
            if "sport" in parsed:
                return parsed
        break
    return {"sport": "generic", "confidence": 0.0}


def _load_sport_events_prompt(sport):
    path = agents_dir() / "sports" / f"{sport}_events.md"
    if path.exists():
        return path.read_text()
    return None


def _parse_classification(raw_text):
    try:
        text = raw_text.strip()
        if "```" in text:
            for part in text.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:]
                if part.startswith("{") and part.endswith("}"):
                    text = part
                    break
        return json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        return {"video_type": "full_match", "confidence": 0.5,
                "evidence": "classification parse failed"}


def _classify_video(client, video_path, interval):
    classifier_prompt = (agents_dir() / "video_classifier.md").read_text()
    samples = []
    for ts, frame in sample_frames(video_path, interval):
        b64 = encode_frame(frame)
        if b64:
            result = _ask_with_retry(client, classifier_prompt, b64)
            if result:
                samples.append(_parse_classification(result))
        if len(samples) >= CLASSIFY_FRAMES:
            break

    if not samples:
        return {"video_type": "full_match", "confidence": 0.0,
                "evidence": "no frames classified"}

    tally = {}
    for s in samples:
        t = s.get("video_type", "full_match")
        tally[t] = tally.get(t, 0) + 1
    best_type = max(tally, key=tally.get)
    return {"video_type": best_type,
            "confidence": tally[best_type] / len(samples),
            "evidence": samples[0].get("evidence", ""),
            "tally": tally}


class VideoOrchestrator:

    def __init__(self, video_path, sample_interval=0.5,
                 depth="full", stream_mode=False, report_only=False,
                 live=False, classify=True, location=None):
        self.video_path = video_path
        self.sample_interval = sample_interval
        self.depth = depth
        self.stream_mode = stream_mode
        self.report_only = report_only
        self.live = live
        self.classify = classify
        self.location = location
        self.video_type = None
        self.sport = None
        self.geo = None
        self.key_events = []

    def _run_parallel(self, client, tasks):
        results = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(_ask_with_retry, client, p, i): k
                       for k, (p, i) in tasks.items()}
            for f in as_completed(futures):
                key = futures[f]
                try:
                    results[key] = f.result()
                except Exception as e:
                    results[key] = None
                    print(f"  [{key}] failed: {e}", file=sys.stderr)
        return results

    def run(self):
        cfg = load_config()
        agents = AgentLoader().load()
        client = VLLMClient(cfg["vllm_endpoint"], cfg["model"])
        timeline = Timeline()

        # ── classify ──
        if self.classify:
            print("Classifying video type", end="", flush=True)
            self.video_type = _classify_video(client, self.video_path,
                                              max(self.sample_interval * 4, 2.0))
            vt = self.video_type["video_type"]
            print(f" → {vt} (confidence: {self.video_type['confidence']:.0%})")
        else:
            self.video_type = {"video_type": "full_match", "confidence": 1.0,
                               "evidence": "classification skipped"}

        # ── geo detection ──
        if self.location:
            self.geo = {"stadium": self.location, "source": "manual"}
            print(f"Location: {self.location} (manual)")
        else:
            print("Detecting location", end="", flush=True)
            self.geo = _detect_geo(client, self.video_path)
            if self.geo:
                print(f" → {self.geo.get('stadium', 'unknown')}, {self.geo.get('country', '')}")
            else:
                print(" → unknown")

        # ── sport detection ──
        print("Detecting sport", end="", flush=True)
        self.sport = _detect_sport(client, self.video_path)
        sport_id = self.sport.get("sport", "generic")
        print(f" → {sport_id} (confidence: {self.sport.get('confidence', 0):.0%})")

        vt = self.video_type["video_type"]

        # ── type-specific prompts ──
        type_prompts = _load_type_prompts().get(vt, {})

        scene_prompt = agents.get("scene_detector", "")
        sport_events_prompt = _load_sport_events_prompt(sport_id)
        event_prompt = sport_events_prompt or agents.get("event_detector", "")
        commentary_prompt = agents.get("commentary_agent", "")
        reasoning_prompt = agents.get("reasoning_agent", "")
        summary_prompt = type_prompts.get("summary_prompt") or agents.get("summary_agent", "")
        highlight_prompt = type_prompts.get("highlight_prompt") or agents.get("highlight_agent", "")

        do_event = self.depth in ("fast", "full")
        do_analysis = self.depth == "full"

        if self.live:
            sampler = sample_live
            counter = count_live_frames
        else:
            sampler = sample_frames
            counter = count_frames

        total_frames = counter(self.video_path, self.sample_interval)
        processed = 0

        for timestamp, frame in sampler(self.video_path, self.sample_interval):
            processed += 1
            image_b64 = encode_frame(frame)
            if image_b64 is None:
                print(f"[{timestamp:.1f}s] encode failed", file=sys.stderr)
                continue

            scene_desc = _ask_with_retry(client, scene_prompt, image_b64)
            if scene_desc is None:
                print(f"[{timestamp:.1f}s] LLM failed", file=sys.stderr)
                continue

            event_str = None
            reasoning_str = None
            commentary_str = None

            if do_event and event_prompt:
                event_str = _ask_with_retry(
                    client, f"{event_prompt}\n\nFrame: {scene_desc}"
                )
                if event_str and sport_events_prompt:
                    parsed_events = _parse_classification(event_str)
                    for ev in parsed_events.get("events", []):
                        ev["timestamp"] = f"{timestamp:.1f}s"
                        self.key_events.append(ev)

            if do_analysis and event_str:
                parallel_tasks = {}
                if reasoning_prompt:
                    parallel_tasks["reasoning"] = (
                        f"{reasoning_prompt}\n\nObservation: {event_str}", None
                    )
                if commentary_prompt:
                    parallel_tasks["commentary"] = (
                        f"{commentary_prompt}\n\nEvent: {event_str}", None
                    )
                if parallel_tasks:
                    r = self._run_parallel(client, parallel_tasks)
                    reasoning_str = r.get("reasoning")
                    commentary_str = r.get("commentary")

            event_dict = {
                "timestamp": f"{timestamp:.1f}s",
                "result": scene_desc,
                "scene": scene_desc,
                "event": event_str or "",
                "reasoning": reasoning_str or "",
                "commentary": commentary_str or "",
            }

            timeline.add(event_dict)

            if self.stream_mode:
                print(f"\n[{timestamp:.1f}s] {scene_desc}")
                if event_str:
                    print(f"  * {event_str}")
                if commentary_str:
                    print(f"  > {commentary_str}")
                print()
            elif not self.report_only:
                bar_len = 30
                done = int(bar_len * processed / max(total_frames, 1))
                bar = f"[{'#' * done}{'-' * (bar_len - done)}]"
                pct = processed / max(total_frames, 1) * 100
                label = f"depth={self.depth} type={vt} sport={sport_id}"
                print(f"\r  {bar} {pct:.0f}% ({processed}/{total_frames}) {label}",
                      end="", flush=True)

        if not self.stream_mode and not self.report_only:
            print()

        video_stem = Path(self.video_path).stem
        csv_path = save_csv(timeline.events, video_stem)

        highlights = ""
        if highlight_prompt and timeline.events:
            highlights = _ask_with_retry(
                client,
                f"{highlight_prompt}\n\nTimeline:\n{_format_events_for_summary(timeline)}"
            ) or ""

        summary_payload = (
            f"{summary_prompt}\n\n"
            f"Timeline:\n{_format_events_for_summary(timeline)}"
        )
        if highlights:
            summary_payload += f"\n\nHighlights:\n{highlights}"

        final_summary = _ask_with_retry(client, summary_payload)
        if final_summary is None:
            print("Summary generation failed", file=sys.stderr)
            final_summary = "Summary unavailable"

        header = (
            f"# {video_stem}\n\n"
            f"**Video Type:** {vt} (confidence: {self.video_type['confidence']:.0%})\n"
        )
        if self.geo:
            geo = self.geo
            header += (
                f"**Location:** {geo.get('stadium', 'unknown')}"
                f" — {geo.get('city', '')}, {geo.get('country', '')}\n"
            )
            if geo.get("league"):
                header += f"**League:** {geo['league']}\n"
            if geo.get("teams"):
                header += f"**Teams:** {', '.join(geo['teams'])}\n"

        if self.key_events:
            header += f"\n## Key Events ({len(self.key_events)})\n\n"
            for ev in self.key_events:
                ts = ev.get("timestamp", "?")
                et = ev.get("type", "event")
                team = ev.get("team", ev.get("batsman", ev.get("player", "")))
                runs = ev.get("runs", ev.get("points", ""))
                detail = f" — {team}" if team else ""
                detail += f" ({runs})" if runs else ""
                header += f"- **[{ts}]** {et}{detail}\n"

        header += f"**Evidence:** {self.video_type.get('evidence', '')}\n\n"
        if highlights:
            header += f"## Highlights\n\n{highlights}\n\n---\n\n"

        report_path = save_report(header + final_summary, video_stem)

        if not self.report_only:
            print(f"\nType:   {vt}")
            print(f"CSV:    {csv_path}")
            print(f"Report: {report_path}")
        else:
            print(report_path)

        return report_path
