"""
VidCore API Server — FastAPI + WebSocket for real-time video analysis.
"""

import asyncio
import json
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.emitter import EventEmitter
from core.orchestrator import VideoOrchestrator
from core.paths import output_dir

app = FastAPI(title="VidCore API", version="1.0")

# mount output dir for serving reels + CSV + reports
static_dir = output_dir()
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(static_dir)), name="output")

# job tracking
jobs = {}
job_lock = threading.Lock()


class WebSocketEmitter(EventEmitter):
    """Forwards orchestrator events to a WebSocket."""

    def __init__(self, ws: WebSocket, loop):
        self.ws = ws
        self.loop = loop

    def _send(self, data):
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.ws.send_json(data), self.loop
            )
            future.result(timeout=5)
        except Exception:
            pass

    def on_scene(self, timestamp, scene_type, activity, scene_raw):
        self._send({"type": "scene", "timestamp": timestamp,
                     "scene_type": scene_type, "activity": activity})

    def on_key_event(self, event):
        self._send({"type": "key_event", **event})

    def on_clip_generated(self, event_type, timestamp, path, total_clips):
        self._send({"type": "clip", "event_type": event_type,
                     "timestamp": timestamp, "path": path,
                     "total_clips": total_clips})

    def on_score_change(self, home, away):
        self._send({"type": "score", "home": home, "away": away})

    def on_phase_change(self, phase):
        self._send({"type": "phase", "phase": phase})

    def on_progress(self, frame, total, pct):
        self._send({"type": "progress", "frame": frame,
                     "total": total, "pct": pct})

    def on_complete(self, report_path, csv_path, reel_paths, key_events_count):
        self._send({"type": "complete", "report": report_path,
                     "csv": csv_path, "reels": reel_paths,
                     "key_events_count": key_events_count})

    def on_error(self, message):
        self._send({"type": "error", "message": message})


def _run_analysis(job_id, video_path, **kwargs):
    """Run orchestrator in background thread, emitting to WebSocket."""
    try:
        emitter = jobs[job_id].get("emitter")
        orchestrator = VideoOrchestrator(
            video_path=str(video_path),
            emitter=emitter,
            generate_reel_flag=True,
            live=True,
            **kwargs,
        )
        result = orchestrator.run()
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["result"] = str(result) if result else None
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        if jobs[job_id].get("emitter"):
            jobs[job_id]["emitter"].on_error(str(e))


# ─── REST Endpoints ──────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"app": "VidCore API", "endpoints": [
        "POST /analyze", "WS /ws/{job_id}",
        "GET /jobs", "GET /status/{job_id}",
        "GET /reels/{job_id}", "GET /output/{path}",
    ]}


@app.post("/analyze")
def start_analysis(video: str, depth: str = "fast", interval: float = 0.5):
    video_path = Path(video)
    if not video_path.exists():
        return JSONResponse({"error": f"Video not found: {video}"}, status_code=404)

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "id": job_id,
        "video": str(video_path),
        "status": "starting",
        "emitter": None,
    }

    return {"job_id": job_id, "video": str(video_path)}


@app.get("/status/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return {"id": job_id, "status": job["status"],
            "result": job.get("result"), "error": job.get("error")}


@app.get("/jobs")
def list_jobs():
    return [{"id": j["id"], "video": j["video"], "status": j["status"]}
            for j in jobs.values()]


@app.get("/reels/{job_id}")
def list_reels(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    video_stem = Path(job["video"]).stem
    live_dir = output_dir() / "reels" / "live"
    manifest = live_dir / f"{video_stem}_manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    return {"clips": [], "count": 0}


@app.get("/manifest/{job_id}")
def get_manifest(job_id: str):
    return list_reels(job_id)


# ─── WebSocket ──────────────────────────────────────────────────────────────

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(ws: WebSocket, job_id: str):
    await ws.accept()

    # wait for job to be ready (POST /analyze creates it)
    for _ in range(50):  # 5s timeout
        if job_id in jobs:
            break
        await asyncio.sleep(0.1)

    job = jobs.get(job_id)
    if not job:
        await ws.send_json({"type": "error", "message": "Job not found"})
        await ws.close()
        return

    video_path = job["video"]
    emitter = WebSocketEmitter(ws, asyncio.get_event_loop())
    job["emitter"] = emitter
    job["status"] = "running"

    await ws.send_json({"type": "started", "job_id": job_id,
                         "video": video_path})

    thread = threading.Thread(
        target=_run_analysis,
        args=(job_id, video_path),
        kwargs={"depth": "fast", "sample_interval": 0.5},
        daemon=True,
    )
    thread.start()

    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
