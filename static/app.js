// VidCore Live Dashboard — all APIs, all states, production-ready
const API = '';
let ws = null, jobId = null, selectedVideo = null, filter = 'all';

// ── Init ────────────────────────────────────────────────────────────────
(async function init() {
  await checkHealth();
  await loadVideos();
})();

async function checkHealth() {
  const dot = document.getElementById('health-dot'), txt = document.getElementById('health-text');
  try {
    const r = await fetch(`${API}/health`);
    const h = await r.json();
    if (h.vllm === 'connected') { dot.className = 'dot on'; txt.textContent = 'vLLM Online'; }
    else { dot.className = 'dot off'; txt.textContent = 'vLLM Offline'; }
  } catch(e) { dot.className = 'dot off'; txt.textContent = 'API Offline'; }
}

async function loadVideos() {
  try {
    const r = await fetch(`${API}/videos`);
    const videos = await r.json();
    const list = document.getElementById('video-list');
    if (!videos.length) { list.innerHTML = '<div class="empty-state">No videos<br>Upload one above</div>'; return; }
    list.innerHTML = videos.map(v =>
      `<div class="vid-item" onclick="selectVideo('${v.path}','${v.name}')" data-path="${v.path}">
        <span>🎬</span><span class="name">${v.name}</span><span class="size">${v.size_mb}MB</span>
      </div>`
    ).join('');
  } catch(e) { document.getElementById('video-list').innerHTML = '<div class="empty-state">Failed to load</div>'; }
}

function selectVideo(path, name) {
  selectedVideo = path;
  document.querySelectorAll('.vid-item').forEach(el => el.classList.remove('active'));
  document.querySelector(`[data-path="${path}"]`)?.classList.add('active');
  setStat('type-val', name.replace(/_/g,' ').replace('.mp4',''));
}

async function uploadVideo() {
  const file = document.getElementById('upload-input').files[0];
  if (!file) return;
  const form = new FormData(); form.append('file', file);
  await fetch(`${API}/upload`, { method: 'POST', body: form });
  await loadVideos();
}

// ── Analysis ────────────────────────────────────────────────────────────
async function startAnalysis() {
  if (!selectedVideo) { alert('Select a video first'); return; }
  stopAnalysis();

  document.getElementById('analyze-btn').disabled = true;
  document.getElementById('analyze-btn').textContent = '⏳ Starting...';
  document.getElementById('stop-btn').style.display = 'block';
  document.getElementById('progress-area').style.display = 'flex';
  resetUI();

  const depth = document.getElementById('depth-select').value;
  const r = await fetch(`${API}/analyze?video=${encodeURIComponent(selectedVideo)}&depth=${depth}&interval=1.0`);
  const job = await r.json();
  jobId = job.job_id;

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws/${jobId}`);
  ws.onmessage = e => handleEvent(JSON.parse(e.data));
  ws.onclose = () => { document.getElementById('analyze-btn').disabled = false; document.getElementById('analyze-btn').textContent = '▶ Start Analysis'; };
  ws.onerror = () => document.getElementById('analyze-btn').textContent = '⚠ Retry';

  pollStatus();
}

function stopAnalysis() {
  if (ws) { ws.close(); ws = null; }
  if (jobId) { fetch(`${API}/jobs/${jobId}`, { method: 'DELETE' }).catch(()=>{}); jobId = null; }
  document.getElementById('analyze-btn').disabled = false;
  document.getElementById('analyze-btn').textContent = '▶ Start Analysis';
  document.getElementById('stop-btn').style.display = 'none';
}

function resetUI() {
  document.getElementById('events').innerHTML = '<div class="empty-state">Connecting to analysis engine...</div>';
  document.getElementById('timeline-bar').innerHTML = '';
  document.getElementById('clips-row').innerHTML = '';
  document.getElementById('reel-wrap').innerHTML = '<div class="placeholder"><div class="spinner"></div><div>Analysis starting...</div></div>';
  setStat('score-val', '0-0');
  setStat('phase-val', 'running');
  setStat('events-val', '0');
  setStat('sport-val', '--');
  document.getElementById('report-content').innerHTML = '<div class="empty-state"><div class="spinner"></div><div>Analyzing...</div></div>';
  document.getElementById('stats-content').innerHTML = '<div class="empty-state">--</div>';
  document.getElementById('info-content').innerHTML = '<div class="empty-state">--</div>';
}

async function pollStatus() {
  if (!jobId) return;
  try {
    const r = await fetch(`${API}/status/${jobId}`);
    const s = await r.json();
    if (s.sport && s.sport !== 'unknown') setStat('sport-val', s.sport);
    if (s.score) setStat('score-val', s.score);
    if (s.phase) setStat('phase-val', s.phase);
    if (s.key_events_count) setStat('events-val', s.key_events_count);
    if (s.status === 'complete') { onComplete(); return; }
    if (s.status === 'error') { onError(s.error); return; }
    setTimeout(pollStatus, 2000);
  } catch(e) { setTimeout(pollStatus, 3000); }
}

// ── WebSocket ───────────────────────────────────────────────────────────
function handleEvent(data) {
  switch(data.type) {
    case 'connected':
      document.getElementById('events').innerHTML = '';
      document.getElementById('analyze-btn').textContent = '🟢 Analyzing...';
      break;
    case 'key_event':
      addEvent(data);
      addTimelineMarker(data);
      addClipThumb(data.event_type);
      updateReelPlayer();
      break;
    case 'clip':
      addClipThumb(data.event_type);
      updateReelPlayer();
      break;
    case 'score':
      setStat('score-val', `${data.home}-${data.away}`);
      break;
    case 'phase':
      setStat('phase-val', data.phase || '--');
      break;
    case 'progress':
      setProgress(data.pct);
      break;
    case 'complete':
      onComplete();
      break;
    case 'error':
      onError(data.message);
      break;
  }
}

function addEvent(ev) {
  const el = document.getElementById('events');
  const et = ev.event_type||'';
  let css = et.includes('GOAL')||et==='SIX'||et==='FOUR'?'goal':
            et.includes('FOUL')||et==='WICKET'?'foul':
            et.includes('CARD')?'card':et.includes('VAR')||et.includes('DRS')?'var':'';

  if (filter !== 'all') {
    const match = filter==='CARD'?et.includes('CARD'):et.includes(filter);
    if (!match) return;
  }

  const team = ev.team ? ` <span style="color:var(--muted)">(${ev.team})</span>` : '';
  const div = document.createElement('div');
  div.className = `event ${css}`;
  div.setAttribute('data-filter', et);
  div.innerHTML = `<div class="e-time">${ev.timestamp||ev.global_time||'?'}</div>
                   <div class="e-type">${et}${team}</div>`;
  el.prepend(div);

  setStat('events-val', parseInt(document.getElementById('events-val').textContent||0) + 1);
}

function addTimelineMarker(ev) {
  const bar = document.getElementById('timeline-bar');
  const et = ev.event_type||'';
  const cls = et.includes('GOAL')?'t-g':et.includes('FOUL')?'t-r':et.includes('CARD')?'t-y':et.includes('VAR')?'t-p':'t-b';
  const m = document.createElement('div');
  m.className = `dot ${cls}`;
  m.style.left = (Math.random()*85+8) + '%';
  m.title = `[${ev.timestamp}] ${et}`;
  bar.appendChild(m);
}

function addClipThumb(type) {
  const row = document.getElementById('clips-row');
  const colors = {GOAL:'#22c55e',GOAL_ATTEMPT:'#3b82f6',FOUL:'#ef4444',YELLOW_CARD:'#f59e0b',RED_CARD:'#ef4444',VAR_CHECK:'#8b5cf6'};
  const d = document.createElement('div');
  d.className = 'clip-thumb';
  d.innerHTML = `<span class="tag" style="background:${colors[type]||'#64748b'}">${type}</span>`;
  row.appendChild(d);
}

function updateReelPlayer() {
  if (!selectedVideo || !jobId) return;
  const name = selectedVideo.split('/').pop().replace('.mp4','');
  const wrap = document.getElementById('reel-wrap');
  wrap.innerHTML = `<video controls autoplay muted loop playsinline style="width:100%;height:100%;object-fit:contain">
    <source src="/output/reels/live/${name}_reel.mp4?t=${Date.now()}" type="video/mp4">
  </video>`;
}

async function loadFlavorReel(flavor) {
  if (!selectedVideo) { alert('Select a video first'); return; }
  const name = selectedVideo.split('/').pop().replace('.mp4','');
  const wrap = document.getElementById('reel-wrap');
  wrap.innerHTML = `<div class="placeholder"><div class="spinner"></div><div>Loading ${flavor} reel...</div></div>`;
  wrap.innerHTML = `<video controls autoplay muted loop playsinline style="width:100%;height:100%;object-fit:contain">
    <source src="/output/reels/${name}_${flavor}.mp4?t=${Date.now()}" type="video/mp4">
  </video>`;
}

function setFilter(f, el) {
  filter = f;
  document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  document.querySelectorAll('.event').forEach(e => {
    const et = e.getAttribute('data-filter')||'';
    e.style.display = (f==='all' || et.includes(f)) ? '' : 'none';
  });
}

// ── Complete / Error ────────────────────────────────────────────────────
async function onComplete() {
  document.getElementById('analyze-btn').disabled = false;
  document.getElementById('analyze-btn').textContent = '✓ Complete';
  document.getElementById('stop-btn').style.display = 'none';
  setStat('phase-val', 'complete');
  setProgress(100);

  try {
    // context
    const ctx = await fetch(`${API}/context/${jobId}`).then(r=>r.json());
    setStat('sport-val', ctx.sport||'--');
    setStat('type-val', ctx.type||'--');
    if (ctx.teams) setStat('location-val', ctx.teams.join(' vs '));
    document.getElementById('info-content').innerHTML = [
      `<div style="font-size:12px;line-height:2">`,
      `<div>Sport: <b>${ctx.sport||'?'}</b></div>`,
      `<div>Type: <b>${ctx.type||'?'}</b></div>`,
      `<div>Teams: <b>${(ctx.teams||[]).join(' vs ')||'?'}</b></div>`,
      `<div>Score: <b>${ctx.score||'?'}</b></div>`,
      `<div>Momentum: <b>${ctx.momentum||0}</b></div>`,
      `</div>`
    ].join('');
    document.getElementById('stats-content').innerHTML = [
      `<div style="font-size:12px;line-height:2">`,
      `<div>Key Events: <b>${ctx.key_events_count||0}</b></div>`,
      `<div>Phase: <b>${ctx.phase||'?'}</b></div>`,
      `<div>Sport: <b>${ctx.sport||'?'}</b></div>`,
      `</div>`
    ].join('');
  } catch(e) {}

  // report
  try {
    const md = await fetch(`${API}/report/${jobId}`).then(r=>r.ok?r.text():null);
    if (md) document.getElementById('report-content').innerHTML = md;
    else document.getElementById('report-content').innerHTML = '<div class="empty-state">Report not available</div>';
  } catch(e) {}

  // key events summary
  try {
    const events = await fetch(`${API}/key_events/${jobId}`).then(r=>r.json());
    if (events.length) {
      const tl = document.getElementById('timeline-bar');
      tl.innerHTML = '';
      events.forEach(ev => addTimelineMarker(ev));
    }
  } catch(e) {}
}

function onError(msg) {
  document.getElementById('analyze-btn').disabled = false;
  document.getElementById('analyze-btn').textContent = '⚠ Failed';
  document.getElementById('stop-btn').style.display = 'none';
  setStat('phase-val', 'error');
  document.getElementById('report-content').innerHTML = `<div class="empty-state" style="color:var(--red)">Error: ${msg||'Unknown'}</div>`;
}

// ── Download ────────────────────────────────────────────────────────────
function downloadCSV() {
  if (!jobId) { alert('No analysis running'); return; }
  window.open(`${API}/csv/${jobId}`, '_blank');
  // also download as file
  fetch(`${API}/csv/${jobId}`).then(r=>r.json()).then(rows=>{
    if (!rows.length) return;
    const csv = [Object.keys(rows[0]).join(',')].concat(rows.map(r=>Object.values(r).map(v=>`"${String(v).replace(/"/g,'""')}"`).join(','))).join('\n');
    const blob = new Blob([csv],{type:'text/csv'});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = `${(selectedVideo||'analysis').split('/').pop().replace('.mp4','')}.csv`;
    a.click();
  }).catch(()=>{});
}

function downloadReport() {
  if (!jobId) { alert('No analysis running'); return; }
  fetch(`${API}/report/${jobId}`).then(r=>r.text()).then(md=>{
    const blob = new Blob([md],{type:'text/markdown'});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = `${(selectedVideo||'analysis').split('/').pop().replace('.mp4','')}_report.md`;
    a.click();
  }).catch(()=>alert('Report not available'));
}

// ── Helpers ─────────────────────────────────────────────────────────────
function setStat(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function setProgress(pct) {
  document.getElementById('progress-fill').style.width = (pct||0) + '%';
  document.getElementById('progress-label').textContent = (pct||0) + '%';
}
