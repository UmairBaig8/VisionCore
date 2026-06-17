#!/usr/bin/env python3
"""
Manual E2E API test — calls the API + WebSocket with a real video.
Requires: api_server.py running, vLLM running.

Usage:
  python api_server.py          # Terminal 1
  python manual_api_e2e.py      # Terminal 2 (requires websocket-client)
"""

import json
import sys
import time
import requests

BASE = "http://localhost:9000"


def manual_e2e(video_path="videos/10.mp4"):
    print(f"Video: {video_path}")
    print(f"API:   {BASE}\n")

    # 1. health check
    r = requests.get(f"{BASE}/health", timeout=5)
    print(f"1. Health: {r.json()}")

    # 2. list videos
    r = requests.get(f"{BASE}/videos")
    videos = r.json()
    print(f"2. Videos: {len(videos)} files")

    # 3. start analysis
    r = requests.post(f"{BASE}/analyze?video={video_path}&depth=fast&interval=1.0&live=true")
    if r.status_code != 200:
        print(f"ERROR: {r.json()}")
        return False
    job = r.json()
    job_id = job["job_id"]
    print(f"3. Job created: {job_id}")

    # 4. check jobs list
    r = requests.get(f"{BASE}/jobs")
    jobs = r.json()
    found = any(j["id"] == job_id for j in jobs)
    print(f"4. Job in list: {found}")

    # 5. wait for analysis to start
    print("5. Waiting for analysis...")
    time.sleep(5)

    # 6. check status
    for _ in range(60):  # poll for 60s
        r = requests.get(f"{BASE}/status/{job_id}")
        status = r.json()
        s = status["status"]
        ctx = status.get("context", {})
        if ctx:
            print(f"6. Status: {s} | sport={ctx.get('sport','?')} "
                  f"score={ctx.get('score','?')} events={ctx.get('key_events_count',0)}")
        else:
            print(f"6. Status: {s}")
        if s in ("complete", "error"):
            break
        time.sleep(3)

    # 7. context
    r = requests.get(f"{BASE}/context/{job_id}")
    ctx = r.json()
    print(f"7. Context: sport={ctx.get('sport','?')} score={ctx.get('score','?')}")

    # 8. key events
    r = requests.get(f"{BASE}/key_events/{job_id}")
    events = r.json()
    print(f"8. Key Events: {len(events)}")
    for ev in events[:5]:
        print(f"    [{ev.get('timestamp','?')}] {ev.get('type','?')}"
              f"{' (' + ev.get('team','') + ')' if ev.get('team') else ''}")

    # 9. reels manifest
    r = requests.get(f"{BASE}/reels/{job_id}")
    reels = r.json()
    print(f"9. Reels: {reels.get('count', 0)} clips")

    # 10. report
    r = requests.get(f"{BASE}/report/{job_id}")
    report_len = len(r.text) if r.status_code == 200 else 0
    print(f"10. Report: {report_len} chars")

    # 11. CSV
    r = requests.get(f"{BASE}/csv/{job_id}")
    csv_rows = len(r.json()) if r.status_code == 200 else 0
    print(f"11. CSV: {csv_rows} rows")

    # 12. cleanup
    r = requests.delete(f"{BASE}/jobs/{job_id}")
    print(f"12. Cleanup: {r.status_code}")

    print(f"\n✓ API E2E complete")
    return True


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "videos/10.mp4"
    ok = manual_e2e(video)
    sys.exit(0 if ok else 1)
