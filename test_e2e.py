#!/usr/bin/env python3
"""
End-to-end test — runs full pipeline on a real video with live vLLM.

Usage:
    python test_e2e.py                          # auto-pick video, stream + live sync
    python test_e2e.py videos/11.mp4            # specific video
"""

import json
import sys
import time
from pathlib import Path

from core.orchestrator import VideoOrchestrator
from core.paths import output_dir, videos_dir


def find_video(path=None):
    if path:
        p = Path(path)
        if p.exists():
            return p
        print(f"Video not found: {path}")
        sys.exit(1)

    vdir = videos_dir()
    mp4s = sorted(vdir.glob("*.mp4"))
    if not mp4s:
        print("No .mp4 files in videos/.")
        sys.exit(1)
    print(f"Using: {mp4s[0].name}")
    return mp4s[0]


def test_e2e(video_path):
    video_stem = video_path.stem

    # stream mode with live sync — matches video playback speed
    depth = "fast"
    interval = 1.0
    live = True
    stream = True
    reel = True

    print(f"\n{'='*60}")
    print(f"E2E: {video_path.name}  [stream + live sync, {depth}, {interval}s interval]")
    print(f"{'='*60}\n")

    t0 = time.time()

    print("1. Starting stream analysis...")
    orch = VideoOrchestrator(
        video_path=str(video_path),
        sample_interval=interval,
        depth=depth,
        live=live,
        stream_mode=stream,
        generate_reel_flag=reel,
        clip_before=4.0,
        clip_after=2.0,
    )
    report_path = orch.run()
    elapsed = time.time() - t0
    print(f"\nDuration: {elapsed:.0f}s")

    results = {"passed": 0, "failed": 0}

    def check(name, path):
        if path and Path(path).exists():
            print(f"  ✓ {name}: {path}")
            results["passed"] += 1
            return True
        else:
            print(f"  ✗ {name}: missing ({path})")
            results["failed"] += 1
            return False

    # ── Outputs ──
    print("\n2. Output files:")
    check("Report", report_path)
    csv_path = output_dir() / "csv" / f"{video_stem}.csv"
    check("CSV", csv_path)
    summary_csv = output_dir() / "csv" / f"{video_stem}_summary.csv"
    check("Summary CSV", summary_csv)

    # ── CSV check ──
    if csv_path.exists():
        lines = csv_path.read_text().strip().split("\n")
        print(f"\n3. CSV: {len(lines)-1} rows, {len(lines[0].split(','))} columns")
        if len(lines) > 1:
            results["passed"] += 1
        else:
            results["failed"] += 1

    # ── Context ──
    if orch.ctx:
        ctx = orch.ctx
        print(f"\n4. Context: sport={ctx.sport} type={ctx.video_type} "
              f"score={ctx.score_string()} phase={ctx.phase} "
              f"events={len(ctx.key_events)}")
        results["passed"] += 1
        if ctx.sport != "generic":
            results["passed"] += 1
        else:
            print("  ✗ Sport not detected (generic)")
            results["failed"] += 1

        for ev in ctx.key_events[:5]:
            print(f"    [{ev.get('timestamp','?')}] {ev.get('type','?')}"
                  f"{' (' + ev.get('team','') + ')' if ev.get('team') else ''}")
        if ctx.key_events:
            results["passed"] += 1
        else:
            print("  ✗ No key events detected")
            results["failed"] += 1
    else:
        print("\n4. Context: MISSING")
        results["failed"] += 3

    # ── Reels ──
    print("\n5. Reels:")
    live_dir = output_dir() / "reels" / "live"
    reel_path = live_dir / f"{video_stem}_reel.mp4"
    manifest = live_dir / f"{video_stem}_manifest.json"

    check("Live reel", reel_path)
    if reel_path.exists():
        print(f"    Size: {reel_path.stat().st_size / 1e6:.1f} MB")

    if manifest.exists():
        data = json.loads(manifest.read_text())
        n = data.get("count", 0)
        print(f"  Manifest: {n} clips")
        if n > 0:
            results["passed"] += 1
        else:
            results["failed"] += 1
    else:
        results["failed"] += 1

    pro_dir = output_dir() / "reels"
    for flavor in ["all", "goals", "drama"]:
        p = pro_dir / f"{video_stem}_{flavor}.mp4"
        if p.exists():
            print(f"  ✓ {flavor}: {p.stat().st_size/1e6:.1f} MB")
            results["passed"] += 1

    # ── Report has content ──
    if report_path and Path(report_path).exists():
        content = Path(report_path).read_text()
        if len(content) > 100:
            results["passed"] += 1
        else:
            results["failed"] += 1

    print(f"\n{'='*60}")
    print(f"E2E: {results['passed']} passed, {results['failed']} failed ({elapsed:.0f}s)")
    print(f"{'='*60}")

    return results["failed"] == 0


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else None
    ok = test_e2e(find_video(video))
    sys.exit(0 if ok else 1)
