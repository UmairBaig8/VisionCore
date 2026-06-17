"""
VidCore API Server — FastAPI + WebSocket for real-time video analysis dashboard.
"""

import asyncio
import csv
import io
import json
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from core.emitter import EventEmitter
from core.orchestrator import VideoOrchestrator
from core.paths import output_dir, videos_dir, project_root

app = FastAPI(title="VidCore API", version="1.0")

# static files for dashboard
from fastapi.staticfiles import StaticFiles as _SM
static_ui = Path(__file__).parent / "static"
if static_ui.exists():
    app.mount("/static", _SM(directory=str(static_ui)), name="static_ui")

# CORS — allow browser dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# concurrency limit — prevent vLLM overload
MAX_CONCURRENT_JOBS = 2
job_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)

static_dir = output_dir()
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(static_dir)), name="output")

# mount videos dir so browser can play source videos
vdir = videos_dir()
if vdir.exists():
    app.mount("/videos", StaticFiles(directory=str(vdir)), name="videos_serve")

# job tracking
jobs = {}
job_lock = threading.Lock()


class WebSocketEmitter(EventEmitter):
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
        self._send({"type": "key_event",
                     "event_type": event.get("type", ""),
                     "timestamp": event.get("timestamp", ""),
                     "team": event.get("team", ""),
                     "player": event.get("player", ""),
                     "description": event.get("global_time", "")})

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
                     "key_events_count": key_events_count,
                     "reel_urls": {k: f"/output/reels/live/{Path(v).name}"
                                   for k, v in (reel_paths or {}).items()}})

    def on_error(self, message):
        self._send({"type": "error", "message": message})


def _run_analysis(job_id, video_path, **kwargs):
    job_semaphore.acquire()
    try:
        emitter = jobs[job_id].get("emitter")
        kwargs.setdefault("stream_mode", True)
        kwargs.setdefault("live", True)
        orchestrator = VideoOrchestrator(
            video_path=str(video_path),
            emitter=emitter,
            generate_reel_flag=True,
            **kwargs,
        )
        jobs[job_id]["orchestrator"] = orchestrator
        result = orchestrator.run()
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["result"] = str(result) if result else None
        if orchestrator.ctx:
            jobs[job_id]["context"] = orchestrator.ctx.summary()
            jobs[job_id]["key_events"] = orchestrator.ctx.key_events
            jobs[job_id]["sport"] = orchestrator.ctx.sport
            jobs[job_id]["score"] = orchestrator.ctx.score_string()
    except Exception as e:
        if job_id in jobs:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)
            if jobs[job_id].get("emitter"):
                jobs[job_id]["emitter"].on_error(str(e))
    finally:
        job_semaphore.release()


# ─── REST Endpoints ──────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"app": "VidCore API", "dashboard": "/dashboard",
            "docs": "/docs", "endpoints": [
        "POST /analyze", "WS /ws/{job_id}",
        "GET /jobs", "GET /status/{job_id}", "DELETE /jobs/{job_id}",
        "GET /report/{job_id}", "GET /csv/{job_id}",
        "GET /key_events/{job_id}", "GET /context/{job_id}",
        "GET /reels/{job_id}", "GET /videos", "POST /upload",
        "GET /health", "GET /output/{path}",
    ]}


@app.get("/dashboard")
def dashboard():
    from fastapi.responses import FileResponse
    return FileResponse(static_ui / "index.html")


@app.get("/demo")
def demo_page():
    from fastapi.responses import FileResponse
    return FileResponse(static_ui / "demo.html")


@app.get("/health")
def health_check():
    vllm_ok = False
    try:
        from core.config import load_config
        cfg = load_config()
        import requests
        r = requests.get(cfg["vllm_endpoint"].replace("/v1/chat/completions", "/health"),
                         timeout=5)
        vllm_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok",
        "vllm": "connected" if vllm_ok else "unreachable",
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }


@app.get("/videos")
def list_videos():
    vdir = videos_dir()
    if not vdir.exists():
        return []
    return sorted([
        {"name": f.name, "path": str(f), "size_mb": round(f.stat().st_size / 1e6, 1)}
        for f in vdir.iterdir() if f.is_file() and f.suffix in (".mp4", ".avi", ".mov")
    ], key=lambda x: x["name"])


@app.post("/upload")
async def upload_video(file: UploadFile):
    vdir = videos_dir()
    vdir.mkdir(parents=True, exist_ok=True)
    path = vdir / file.filename
    with open(path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"name": file.filename, "path": str(path),
            "size_mb": round(len(content) / 1e6, 1)}


@app.post("/analyze")
def start_analysis(video: str, depth: str = "fast", interval: float = 1.0):
    video_path = Path(video)
    if not video_path.exists():
        return JSONResponse({"error": f"Video not found: {video}"}, status_code=404)

    job_id = str(uuid.uuid4())[:8]
    emitter = EventEmitter()
    jobs[job_id] = {
        "id": job_id,
        "video": str(video_path),
        "status": "running",
        "emitter": emitter,
        "context": None,
        "key_events": [],
        "orchestrator": None,
    }

    thread = threading.Thread(
        target=_run_analysis,
        args=(job_id, video_path),
        kwargs={"depth": depth, "sample_interval": interval},
        daemon=True,
    )
    thread.start()

    # check if job is queued (semaphore blocked)
    time.sleep(0.1)
    if jobs[job_id]["status"] == "running":
        pass  # started immediately
    else:
        jobs[job_id]["status"] = "queued"

    return {"job_id": job_id, "video": str(video_path),
            "ws_url": f"/ws/{job_id}"}


@app.get("/jobs")
def list_jobs():
    return [{"id": j["id"], "video": j["video"], "status": j["status"],
             "sport": j.get("sport", ""), "score": j.get("score", ""),
             "events": len(j.get("key_events", []))}
            for j in jobs.values()]


@app.get("/status/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    orch = job.get("orchestrator")
    ctx = None
    sport = job.get("sport", "unknown")
    score = job.get("score", "0-0")
    events_count = len(job.get("key_events", []))

    if orch and orch.ctx:
        ctx = orch.ctx.summary()
        sport = orch.ctx.sport
        score = orch.ctx.score_string()
        events_count = len(orch.ctx.key_events)

    return {"id": job_id, "status": job["status"],
            "result": job.get("result"), "error": job.get("error"),
            "sport": sport, "score": score,
            "context": ctx, "key_events_count": events_count}


@app.get("/sse/{job_id}")
async def sse_stream(job_id: str):
    """Server-Sent Events fallback — works through proxies that block WebSocket."""
    from fastapi.responses import StreamingResponse
    import asyncio as aio

    async def event_stream():
        job = jobs.get(job_id)
        if not job:
            yield f"data: {json.dumps({'type':'error','message':'Job not found'})}\n\n"
            return

        last_events = 0
        while True:
            job = jobs.get(job_id)
            if not job:
                break

            orch = job.get("orchestrator")
            if orch and orch.ctx:
                ctx = orch.ctx
                current = len(ctx.key_events)
                if current > last_events:
                    for ev in ctx.key_events[last_events:]:
                        yield f"data: {json.dumps({'type':'key_event','event_type':ev.get('type',''),'timestamp':ev.get('timestamp',''),'team':ev.get('team',''),'player':ev.get('player',''),'description':ev.get('global_time','')})}\n\n"
                    last_events = current

                yield f"data: {json.dumps({'type':'status','score':ctx.score_string(),'sport':ctx.sport,'phase':ctx.phase,'events':last_events})}\n\n"

            if job.get("status") in ("complete", "error"):
                yield f"data: {json.dumps({'type':'complete','status':job['status']})}\n\n"
                break

            await aio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    job = jobs.pop(job_id, None)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return {"deleted": job_id}


@app.get("/context/{job_id}")
def get_context(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return job.get("context") or {"sport": "unknown", "score": "0-0"}


@app.get("/key_events/{job_id}")
def get_key_events(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return job.get("key_events", [])


@app.get("/reels/{job_id}")
def list_reels(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    video_stem = Path(job["video"]).stem
    live_dir = output_dir() / "reels" / "live"
    manifest = live_dir / f"{video_stem}_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        # add relative URLs
        for c in data.get("clips", []):
            c["url"] = f"/output/reels/live/{Path(c['path']).name}"
        data["reel_url"] = f"/output/reels/live/{video_stem}_reel.mp4"
        return data
    return {"clips": [], "count": 0}


@app.get("/report/{job_id}")
def get_report(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if job.get("result"):
        report_path = Path(job["result"])
        if report_path.exists():
            return PlainTextResponse(report_path.read_text(), media_type="text/markdown")
    return JSONResponse({"error": "Report not available yet"}, status_code=404)


@app.get("/csv/{job_id}")
def get_csv_json(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    video_stem = Path(job["video"]).stem
    csv_path = output_dir() / "csv" / f"{video_stem}.csv"
    if not csv_path.exists():
        return JSONResponse({"error": "CSV not available yet"}, status_code=404)

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ─── WebSocket ──────────────────────────────────────────────────────────────

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(ws: WebSocket, job_id: str):
    await ws.accept()

    job = jobs.get(job_id)
    if not job:
        await ws.send_json({"type": "error", "message": "Job not found"})
        await ws.close()
        return

    # swap no-op emitter with live WebSocket emitter
    old_emitter = job.get("emitter")
    emitter = WebSocketEmitter(ws, asyncio.get_event_loop())
    job["emitter"] = emitter

    orch = job.get("orchestrator")
    if orch:
        orch.emitter = emitter

    await ws.send_json({"type": "connected", "job_id": job_id,
                         "video": job["video"],
                         "status": job.get("status"),
                         "sport": job.get("sport"),
                         "score": job.get("score")})

    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        job["emitter"] = old_emitter or EventEmitter()
        if orch:
            orch.emitter = job["emitter"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
