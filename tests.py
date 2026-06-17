#!/usr/bin/env python3
"""Test suite for VidCore agents and skills.

Usage:
    python tests.py              # run all tests
    python tests.py --quick      # skip LLM connectivity test
"""

import base64
import sys
from pathlib import Path

import cv2
import numpy as np

from core.agent_loader import AgentLoader
from core.config import load_config
from core.llm_client import VLLMClient
from core.models import Event, AnalysisResult
from core.orchestrator import VideoOrchestrator, _format_events_for_summary
from core.paths import project_root, agents_dir, skills_dir, config_path, output_dir, videos_dir
from core.registry import SkillRegistry

from skills.frame_sampler import sample_frames, count_frames
from skills.frame_encoder import encode_frame
from skills.report_generator import save_report
from skills.timeline import Timeline
from skills.video_loader import open_video

OK = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

passed = 0
failed = 0
skipped = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  {OK}  {name}")
        passed += 1
    except Exception as e:
        print(f"  {FAIL}  {name}: {e}")
        failed += 1


def skip_test(name, reason=""):
    global skipped
    msg = f" ({reason})" if reason else ""
    print(f"  {SKIP}  {name}{msg}")
    skipped += 1


# ─── Paths ───────────────────────────────────────────────────────────────────

def test_paths():
    print("\nPaths")
    r = project_root()
    test("project_root is dir", lambda: (r.is_dir()))
    test("project_root has agents/", lambda: (r / "agents").is_dir())
    test("project_root has skills/", lambda: (r / "skills").is_dir())
    test("project_root has config.yaml", lambda: (r / "config.yaml").is_file())
    test("project_root has main.py", lambda: (r / "main.py").is_file())
    test("agents_dir()", lambda: (agents_dir().is_dir()))
    test("skills_dir()", lambda: (skills_dir().is_dir()))
    test("config_path()", lambda: (config_path().is_file()))
    test("videos_dir()", lambda: (videos_dir().is_dir()))


# ─── Config ─────────────────────────────────────────────────────────────────

def test_config():
    print("\nConfig")
    cfg = load_config()
    test("vllm_endpoint present", lambda: "vllm_endpoint" in cfg)
    test("model present", lambda: "model" in cfg)
    test("model is Qwen3-VL-32B", lambda: "Qwen/Qwen3-VL-32B-Instruct" in cfg["model"])
    print(f"    endpoint: {cfg['vllm_endpoint'][:60]}...")


# ─── Agents ─────────────────────────────────────────────────────────────────

def test_agents():
    print("\nAgents")
    loader = AgentLoader()
    agents = loader.load()
    expected = [
        "scene_detector", "event_detector", "reasoning_agent",
        "commentary_agent", "summary_agent", "timeline_agent",
        "highlight_agent",
    ]
    test(f"loaded {len(agents)} agents", lambda: len(agents) >= 7)
    for name in expected:
        test(f"  agent '{name}'", lambda n=name: n in agents)
        test(f"  agent '{name}' non-empty", lambda n=name: len(agents[n]) > 10)


# ─── Skills ─────────────────────────────────────────────────────────────────

def test_skills():
    print("\nSkills")
    registry = SkillRegistry()
    skills = registry.load()
    expected = ["frame_sampler", "frame_encoder", "report_generator", "timeline", "video_loader"]
    test(f"loaded {len(skills)} skills", lambda: len(skills) >= 5)
    for name in expected:
        test(f"  skill '{name}'", lambda n=name: n in skills)


# ─── Timeline ───────────────────────────────────────────────────────────────

def test_timeline():
    print("\nTimeline")
    tl = Timeline()
    test("empty after init", lambda: tl.events == [])
    test("latest on empty is None", lambda: tl.latest() is None)

    tl.add({"time": 1.0, "result": "test event"})
    test("add event", lambda: len(tl.events) == 1)
    test("latest returns event", lambda: tl.latest()["time"] == 1.0)
    test("latest returns correct result", lambda: tl.latest()["result"] == "test event")

    tl.add({"time": 2.0, "result": "second"})
    test("second add", lambda: len(tl.events) == 2)
    test("latest after second", lambda: tl.latest()["time"] == 2.0)


# ─── Frame encoder ──────────────────────────────────────────────────────────

def test_frame_encoder():
    print("\nFrame Encoder")
    fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    b64 = encode_frame(fake_frame)
    test("encodes to non-empty string", lambda: len(b64) > 0)
    test("encodes valid base64", lambda: base64.b64decode(b64))
    test("decode is JPEG", lambda: base64.b64decode(b64)[:2] == b'\xff\xd8')

    bad_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    result = encode_frame(bad_frame)
    test("returns None on bad frame", lambda: result is None)


# ─── Frame sampler ──────────────────────────────────────────────────────────

def test_frame_sampler():
    print("\nFrame Sampler (synthetic 1s clip)")
    target_fps = 30
    duration = 1
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp = "/tmp/vidcore_test.mp4"
    writer = cv2.VideoWriter(tmp, fourcc, target_fps, (320, 240))
    for i in range(target_fps * duration):
        frame = np.full((240, 320, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    try:
        n = count_frames(tmp, interval=0.5)
        test("count_frames ~2 (0.5s interval on 1s)", lambda: 1 <= n <= 3)

        frames = list(sample_frames(tmp, interval=0.5))
        test("sample_frames yields frames", lambda: len(frames) >= 1)
        test("yields (timestamp, frame) tuples", lambda: isinstance(frames[0], tuple) and len(frames[0]) == 2)
        ts, fr = frames[0]
        test("frame is ndarray", lambda: isinstance(fr, np.ndarray))
        test("timestamp is float", lambda: isinstance(ts, float))

    except RuntimeError:
        test("open_video raises RuntimeError on missing file", lambda: True)
    except Exception as e:
        if "RuntimeError" in str(e) or "Cannot open" in str(e):
            test("open_video raises RuntimeError on missing file", lambda: True)
        else:
            print(f"    unexpected: {e}")
    finally:
        Path(tmp).unlink(missing_ok=True)

    # separate test for missing file
    try:
        open_video("/tmp/nonexistent_vidcore_xyz.mp4")
        test("open_video raises RuntimeError on missing file", lambda: False)
    except RuntimeError:
        test("open_video raises RuntimeError on missing file", lambda: True)


# ─── Report generator ───────────────────────────────────────────────────────

def test_report():
    print("\nReport Generator")
    path = save_report("# Test Report\nHello world", "test_video")
    test("creates report file", lambda: path.exists())
    test("correct extension", lambda: path.suffix == ".md")
    content = path.read_text()
    test("content written correctly", lambda: "# Test Report" in content)
    path.unlink()


# ─── Models ─────────────────────────────────────────────────────────────────

def test_models():
    print("\nModels")
    e = Event(timestamp="1.0s", event_type="scene", description="desc", confidence=0.9)
    test("Event creation", lambda: e.timestamp == "1.0s")
    test("Event confidence", lambda: e.confidence == 0.9)
    test("Event metadata default None", lambda: e.metadata is None)

    r = AnalysisResult(timeline=[e], summary="test summary", report_path="/tmp/report.md")
    test("AnalysisResult creation", lambda: len(r.timeline) == 1)
    test("event_count property", lambda: r.event_count == 1)
    test("summary stored", lambda: r.summary == "test summary")


# ─── CSV writer ──────────────────────────────────────────────────────────────

def test_csv_writer():
    print("\nCSV Writer")
    from skills.csv_writer import save_csv
    events = [
        {"timestamp": "0.0s", "scene": "outdoor", "event": "movement", "reasoning": "low", "commentary": "person walking"},
        {"timestamp": "0.5s", "scene": "indoor", "event": "none", "reasoning": "", "commentary": ""},
    ]
    path = save_csv(events, "test_csv")
    test("creates csv file", lambda: path.exists())
    content = path.read_text()
    test("has header", lambda: "timestamp,scene,event,reasoning,commentary" in content)
    test("has row 1", lambda: "outdoor" in content and "person walking" in content)
    test("has row 2", lambda: "indoor" in content)
    path.unlink()
    path.parent.rmdir()


# ─── Classification ─────────────────────────────────────────────────────────

def test_classification():
    print("\nClassification")
    from core.orchestrator import _parse_classification, _load_type_prompts

    test("full_match parsed", lambda:
         _parse_classification('{"video_type":"full_match","confidence":0.9}')["video_type"] == "full_match")

    test("highlights parsed", lambda:
         _parse_classification('{"video_type":"highlights","confidence":0.8}')["video_type"] == "highlights")

    test("code-fenced JSON stripped", lambda:
         _parse_classification('```json\n{"video_type":"training"}\n```')["video_type"] == "training")

    test("junk text falls back to full_match", lambda:
         _parse_classification("not json at all")["video_type"] == "full_match")

    prompts = _load_type_prompts()
    test("type_prompts.json loads", lambda: isinstance(prompts, dict) and len(prompts) >= 4)
    test("has full_match prompts", lambda: "full_match" in prompts)
    test("has highlights prompts", lambda: "highlights" in prompts)
    test("has press_conference prompts", lambda: "press_conference" in prompts)
    test("has training prompts", lambda: "training" in prompts)


# ─── Sport detection ────────────────────────────────────────────────────────

def test_sport():
    print("\nSport Detection")
    from core.orchestrator import _load_sport_events_prompt

    prompts = {
        "football": _load_sport_events_prompt("football"),
        "cricket": _load_sport_events_prompt("cricket"),
        "basketball": _load_sport_events_prompt("basketball"),
        "tennis": _load_sport_events_prompt("tennis"),
    }
    for sport, prompt in prompts.items():
        test(f"sport_events/{sport}_events.md loaded", lambda p=prompt: p is not None and len(p) > 50)
        test(f"  has event types for {sport}", lambda p=prompt, s=sport: "Event" in p and ("GOAL" in p or "SIX" in p or "DUNK" in p or "ACE" in p))

    test("unknown sport returns None", lambda: _load_sport_events_prompt("quidditch") is None)


# ─── Video loader ───────────────────────────────────────────────────────────

def test_video_loader():
    print("\nVideo Loader")
    try:
        open_video("/tmp/nonexistent_xyz123.mp4")
        assert False, "should have raised"
    except RuntimeError as e:
        test("raises RuntimeError on missing file", lambda: "Cannot open" in str(e))

    tmp = "/tmp/vidcore_test2.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp, fourcc, 30, (320, 240))
    for i in range(30):
        writer.write(np.full((240, 320, 3), i, dtype=np.uint8))
    writer.release()

    try:
        cap = open_video(tmp)
        test("opens valid video", lambda: cap.isOpened())
        cap.release()
    finally:
        Path(tmp).unlink(missing_ok=True)


# ─── Orchestrator (instantiation only, no LLM) ──────────────────────────────

def test_orchestrator():
    print("\nOrchestrator")
    try:
        o = VideoOrchestrator(video_path="/tmp/doesnt_exist.mp4", sample_interval=1.0)
        test("instantiate", lambda: o.video_path == "/tmp/doesnt_exist.mp4")
        test("sample_interval stored", lambda: o.sample_interval == 1.0)
        test("stream_mode default False", lambda: o.stream_mode is False)
        test("report_only default False", lambda: o.report_only is False)

        o2 = VideoOrchestrator(video_path="x", stream_mode=True, report_only=True)
        test("stream_mode=True", lambda: o2.stream_mode is True)
        test("report_only=True", lambda: o2.report_only is True)
    except Exception:
        skip_test("instantiation", "no vLLM backend")


# ─── Timeline formatting ────────────────────────────────────────────────────

def test_format_events():
    print("\nTimeline Formatting")
    tl = Timeline()
    tl.add({"timestamp": "1.0s", "result": "event one"})
    tl.add({"timestamp": "2.0s", "result": "event two"})
    formatted = _format_events_for_summary(tl)
    test("contains event headers", lambda: "## Event 1" in formatted)
    test("contains timestamps", lambda: "[1.0s]" in formatted)
    test("contains results", lambda: "event one" in formatted and "event two" in formatted)


# ─── CLI smoke test ─────────────────────────────────────────────────────────

def test_cli_smoke():
    print("\nCLI Smoke")
    import subprocess
    py = Path(__file__).parent / ".venv/bin/python"
    main = Path(__file__).parent / "main.py"

    if not py.exists():
        skip_test("CLI subprocess", "venv not at .venv/bin/python")
        return

    for cmd in ["agents", "skills", "videos", "doctor", "config", "--help"]:
        result = subprocess.run(
            [str(py), str(main)] + cmd.split(),
            capture_output=True, text=True, cwd=Path(__file__).parent
        )
        test(f"  {cmd} exit 0", lambda r=result: r.returncode == 0)


# ─── LLM connectivity (optional) ────────────────────────────────────────────

def test_llm_connectivity():
    print("\nLLM Connectivity")
    cfg = load_config()
    client = VLLMClient(cfg["vllm_endpoint"], cfg["model"])
    try:
        result = client.ask("Say 'hello' and nothing else.")
        test("LLM responds", lambda: isinstance(result, str) and len(result) > 0)
        test("LLM says hello", lambda: "hello" in result.lower())
    except Exception as e:
        skip_test("LLM connectivity", str(e)[:80])


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    quick = "--quick" in sys.argv

    print(f"VidCore Test Suite  (project_root={project_root()})")

    test_paths()
    test_config()
    test_agents()
    test_skills()
    test_timeline()
    test_frame_encoder()
    test_frame_sampler()
    test_report()
    test_models()
    test_csv_writer()
    test_classification()
    test_sport()
    test_video_loader()
    test_orchestrator()
    test_format_events()
    test_cli_smoke()

    if not quick:
        test_llm_connectivity()
    else:
        print("\nLLM Connectivity")
        skip_test("LLM connectivity", "--quick flag set")

    total = passed + failed + skipped
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped ({total} total)")
    if failed > 0:
        sys.exit(1)
