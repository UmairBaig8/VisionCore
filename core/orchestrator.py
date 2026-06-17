import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from core.agent_loader import AgentLoader
from core.config import load_config
from core.llm_client import VLLMClient

from skills.frame_sampler import sample_frames, count_frames
from skills.live_sampler import sample_live, count_live_frames
from skills.frame_encoder import encode_frame
from skills.timeline import Timeline
from skills.report_generator import save_report
from skills.csv_writer import save_csv

MAX_RETRIES = 3
RETRY_BACKOFF = 2


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


class VideoOrchestrator:

    def __init__(self, video_path, sample_interval=0.5,
                 depth="full", stream_mode=False, report_only=False,
                 live=False):
        self.video_path = video_path
        self.sample_interval = sample_interval
        self.depth = depth
        self.stream_mode = stream_mode
        self.report_only = report_only
        self.live = live

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

        scene_prompt = agents.get("scene_detector", "")
        event_prompt = agents.get("event_detector", "")
        commentary_prompt = agents.get("commentary_agent", "")
        reasoning_prompt = agents.get("reasoning_agent", "")
        summary_prompt = agents.get("summary_agent", "")
        highlight_prompt = agents.get("highlight_agent", "")

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
                label = f"depth={self.depth}"
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

        if highlights:
            final_summary = (
                f"## Highlights\n\n{highlights}\n\n"
                f"---\n\n"
                f"## Full Analysis\n\n{final_summary}"
            )

        report_path = save_report(final_summary, video_stem)

        if not self.report_only:
            print(f"\nCSV:    {csv_path}")
            print(f"Report: {report_path}")
        else:
            print(report_path)

        return report_path
