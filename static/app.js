// VidCore Dashboard — WebSocket-powered live UI
const API = '';
let ws = null;
let jobId = null;
let selectedVideo = null;

// ── Init ────────────────────────────────────────────────────────────────
async function init() {
  await loadVideos();
}

async function loadVideos() {
  const r = await fetch(`${API}/videos`);
  const videos = await r.json();
  const list = document.getElementById('video-list');
  if (!videos.length) {
    list.innerHTML = '<div style="color:#8b949e;font-size:12px">No videos found. Upload one above.</div>';
    return;
  }
  list.innerHTML = videos.map(v => {
    const mb = v.size_mb ? ` (${v.size_mb}MB)` : '';
    return `<div class="video-item" onclick="selectVideo('${v.path}','${v.name}')" data-path="${v.path}">
      🎬 ${v.name}${mb}
    </div>`;
  }).join('');
}

function selectVideo(path, name) {
  selectedVideo = path;
  document.querySelectorAll('.video-item').forEach(el => el.classList.remove('active'));
  document.querySelector(`[data-path="${path}"]`)?.classList.add('active');
  document.getElementById('status-text').textContent = `● Selected: ${name}`;
}

// ── Upload ──────────────────────────────────────────────────────────────
async function uploadVideo() {
  const file = document.getElementById('upload-input').files[0];
  if (!file) return;
  setStatus('Uploading...', 'uploading');
  const form = new FormData();
  form.append('file', file);
  const r = await fetch(`${API}/upload`, { method: 'POST', body: form });
  const data = await r.json();
  setStatus(`Uploaded: ${data.name}`, 'ready');
  selectedVideo = data.path;
  await loadVideos();
}

// ── Analysis + WebSocket ────────────────────────────────────────────────
async function startAnalysis() {
  if (!selectedVideo) { alert('Select a video first'); return; }
  if (jobId) { stopAnalysis(); }

  const depth = document.getElementById('depth-select').value;
  setStatus('Starting...', 'running');
  document.getElementById('analyze-btn').disabled = true;

  const r = await fetch(`${API}/analyze?video=${encodeURIComponent(selectedVideo)}&depth=${depth}&interval=1.0`);
  const job = await r.json();
  jobId = job.job_id;

  // connect WebSocket
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws/${jobId}`);

  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    handleEvent(data);
  };

  ws.onclose = () => {
    document.getElementById('analyze-btn').disabled = false;
    if (document.getElementById('status-text').textContent.includes('Running')) {
      setStatus('Analysis complete', 'complete');
    }
  };

  ws.onerror = () => setStatus('WebSocket error', 'error');

  // start polling status for context updates
  pollStatus();
}

function stopAnalysis() {
  if (ws) { ws.close(); ws = null; }
  if (jobId) { fetch(`${API}/jobs/${jobId}`, { method: 'DELETE' }); jobId = null; }
  setStatus('Stopped', 'ready');
  document.getElementById('analyze-btn').disabled = false;
  document.getElementById('progress-fill').style.width = '0%';
}

async function pollStatus() {
  if (!jobId) return;
  try {
    const r = await fetch(`${API}/status/${jobId}`);
    const s = await r.json();
    if (s.sport) document.getElementById('sport-val').textContent = s.sport;
    if (s.score) document.getElementById('score-val').textContent = s.score;
    if (s.key_events_count !== undefined) document.getElementById('events-val').textContent = s.key_events_count;
    if (s.status === 'complete' || s.status === 'error') {
      setStatus(s.status === 'error' ? 'Error: ' + s.error : 'Complete', s.status);
      document.getElementById('analyze-btn').disabled = false;
      loadReport();
      loadReels();
      return;
    }
    setTimeout(pollStatus, 3000);
  } catch(e) {
    setTimeout(pollStatus, 3000);
  }
}

// ── Event Handlers ──────────────────────────────────────────────────────
function handleEvent(data) {
  const eventsDiv = document.getElementById('events');

  switch(data.type) {
    case 'connected':
      setStatus('Connected — analyzing...', 'running');
      clearEvents();
      break;

    case 'scene':
      // scene events are high-frequency — only update phase/scoreboard
      break;

    case 'key_event':
      if (data.event_type) {
        addEvent(data);
        updateReelPlayer();
      }
      break;

    case 'score':
      document.getElementById('score-val').textContent = `${data.home}-${data.away}`;
      break;

    case 'phase':
      document.getElementById('phase-val').textContent = data.phase || '--';
      break;

    case 'clip':
      updateReelPlayer();
      break;

    case 'progress':
      document.getElementById('progress-fill').style.width = `${data.pct}%`;
      document.getElementById('events-val').textContent = data.key_events_count || 0;
      break;

    case 'complete':
      setStatus('Complete', 'complete');
      document.getElementById('analyze-btn').disabled = false;
      document.getElementById('progress-fill').style.width = '100%';
      loadReport();
      loadReels();
      break;

    case 'error':
      setStatus('Error: ' + data.message, 'error');
      break;
  }
}

function addEvent(ev) {
  const eventsDiv = document.getElementById('events');
  if (eventsDiv.querySelector('.waiting')) eventsDiv.innerHTML = '';

  const cssClass = ev.event_type?.includes('GOAL') ? 'goal' :
                   ev.event_type?.includes('CARD') || ev.event_type?.includes('FOUL') ? 'card' : '';

  const team = ev.team ? ` (${ev.team})` : '';
  const div = document.createElement('div');
  div.className = `event ${cssClass}`;
  div.innerHTML = `<span class="ts">${ev.timestamp || '?'}</span>
                   <span class="type">${ev.event_type}</span>${team}`;
  eventsDiv.prepend(div);

  document.getElementById('events-val').textContent =
    parseInt(document.getElementById('events-val').textContent || 0) + 1;
}

function clearEvents() {
  document.getElementById('events').innerHTML = '<div class="waiting" style="color:#8b949e;font-size:12px">Waiting for events...</div>';
  document.getElementById('events-val').textContent = '0';
}

// ── Reel Player ─────────────────────────────────────────────────────────
function updateReelPlayer() {
  if (!selectedVideo || !jobId) return;
  const name = selectedVideo.split('/').pop().replace('.mp4','');
  const player = document.getElementById('reel-player');
  const label = document.getElementById('reel-label');
  player.querySelector('source').src = `/output/reels/live/${name}_reel.mp4?t=${Date.now()}`;
  player.load();
  player.play().catch(() => {});
  label.textContent = 'Live reel — auto-updating as events are detected';
}

async function loadReels() {
  if (!jobId) return;
  try {
    const r = await fetch(`${API}/reels/${jobId}`);
    const data = await r.json();
    if (data.reel_url) {
      const player = document.getElementById('reel-player');
      player.querySelector('source').src = data.reel_url;
      player.load();
      document.getElementById('reel-label').textContent =
        `Final reel — ${data.count} clips`;
    }
  } catch(e) {}
}

// ── Report ──────────────────────────────────────────────────────────────
async function loadReport() {
  if (!jobId) return;
  try {
    const r = await fetch(`${API}/report/${jobId}`);
    if (r.ok) {
      const md = await r.text();
      const panel = document.getElementById('report-panel');
      panel.style.display = 'block';
      panel.innerHTML = `<h3>📋 Match Report</h3>` + md;
    }
  } catch(e) {}
}

// ── Helpers ─────────────────────────────────────────────────────────────
function setStatus(text, state) {
  const el = document.getElementById('status-text');
  const icons = { ready: '●', running: '🟢', complete: '✅', error: '❌', uploading: '⬆️' };
  el.textContent = `${icons[state] || '●'} ${text}`;
}

// ── Start ───────────────────────────────────────────────────────────────
init();
