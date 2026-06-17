#!/usr/bin/env python3
"""
End-to-end test — runs full pipeline on a real video with live vLLM.
Requires: vLLM server running, video in videos/ directory.

Usage:
    python test_e2e.py videos/11.mp4           # specific video
    python test_e2e.py                         # auto-pick first .mp4 in videos/
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
        print("No .mp4 files in videos/. Place a video there first.")
        sys.exit(1)

    print(f"Found {len(mp4s)} videos. Using: {mp4s[0].name}")
    return mp4s[0]


def test_e2e(video_path):
    video_stem = video_path.stem
    print(f"\n{'='*60}")
    print(f"E2E Test: {video_path.name}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # ── 1. Run analysis ──
    print("1. Running analysis (depth=fast, clip=5s/3s)...")
    orch = VideoOrchestrator(
        video_path=str(video_path),
        sample_interval=0.5,
        depth="fast",
        generate_reel_flag=True,
        live=True,
        clip_before=4.0,
        clip_after=2.0,
    )
    report_path = orch.run()
    elapsed = time.time() - t0

    print(f"\nDuration: {elapsed:.0f}s")

    # ── 2. Check outputs exist ──
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

    print("\n2. Output files:")
    check("Report", report_path)

    csv_path = output_dir() / "csv" / f"{video_stem}.csv"
    check("CSV", csv_path)

    summary_csv = output_dir() / "csv" / f"{video_stem}_summary.csv"
    check("Summary CSV", summary_csv)

    # ── 3. Check CSV content ──
    if csv_path.exists():
        print("\n3. CSV content:")
        lines = csv_path.read_text().strip().split("\n")
        print(f"  Rows: {len(lines) - 1} data rows (+ header)")
        header = lines[0].split(",")
        print(f"  Columns: {len(header)} ({', '.join(header[:5])}...)")
        if len(lines) > 1:
            print(f"  Sample: {lines[1][:120]}...")
            results["passed"] += 1
        else:
            print("  ✗ No data rows")
            results["failed"] += 1

    # ── 4. Check context ──
    if orch.ctx:
        print("\n4. Match context:")
        ctx = orch.ctx
        print(f"  Sport: {ctx.sport}")
        print(f"  Type:  {ctx.video_type}")
        print(f"  Score: {ctx.score_string()}")
        print(f"  Phase: {ctx.phase}")
        print(f"  Key Events: {len(ctx.key_events)}")
        if ctx.sport != "generic":
            results["passed"] += 1
        else:
            print("  ✗ Sport not detected")
            results["failed"] += 1

        for ev in ctx.key_events[:5]:
            print(f"    [{ev.get('timestamp', '?')}] {ev.get('type', '?')}"
                  f"{' (' + ev.get('team', '') + ')' if ev.get('team') else ''}")
        if ctx.key_events:
            results["passed"] += 1
        else:
            print("  ✗ No key events detected")
            results["failed"] += 1
    else:
        print("\n4. Context: MISSING")
        results["failed"] += 2

    # ── 5. Check reels ──
    print("\n5. Reels:")
    live_dir = output_dir() / "reels" / "live"
    reel = live_dir / f"{video_stem}_reel.mp4"
    manifest = live_dir / f"{video_stem}_manifest.json"

    check("Live reel", reel)
    if reel.exists():
        size_mb = reel.stat().st_size / 1e6
        print(f"    Size: {size_mb:.1f} MB")

    if manifest.exists():
        data = json.loads(manifest.read_text())
        print(f"  Manifest: {data.get('count', 0)} clips")
        for c in data.get("clips", []):
            print(f"    [{c['timestamp']}s] {c['event_type']} → {Path(c['path']).name}")
        if data.get("count", 0) > 0:
            results["passed"] += 1
        else:
            results["failed"] += 1
    else:
        check("Manifest", manifest)
        results["failed"] += 1

    # ── 6. Check pro reels ──
    pro_dir = output_dir() / "reels"
    for flavor in ["all", "goals", "drama", "social_goals"]:
        pro_reel = pro_dir / f"{video_stem}_{flavor}.mp4"
        if pro_reel.exists():
            print(f"  ✓ {flavor}: {pro_reel.name} ({pro_reel.stat().st_size / 1e6:.1f} MB)")
            results["passed"] += 1

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"E2E Results: {results['passed']} passed, {results['failed']} failed")
    print(f"Duration: {elapsed:.0f}s")
    print(f"{'='*60}")

    return results["failed"] == 0


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else None
    ok = test_e2e(find_video(video))
    sys.exit(0 if ok else 1)
