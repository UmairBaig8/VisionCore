// VidCore Live Dashboard — WebSocket-powered
const API = '';
let ws = null, jobId = null, selectedVideo = null, filter = 'all';

async function init() { await loadVideos(); }

async function loadVideos() {
  const r = await fetch(`${API}/videos`);
  const videos = await r.json();
  const list = document.getElementById('video-list');
  if (!videos.length) { list.innerHTML = '<div style="color:var(--muted);font-size:12px">No videos. Upload above.</div>'; return; }
  list.innerHTML = videos.map(v =>
    `<div class="video-item" onclick="selectVideo('${v.path}','${v.name}')" data-path="${v.path}">🎬 ${v.name}${v.size_mb ? ' ('+v.size_mb+'MB)' : ''}</div>`
  ).join('');
}

function selectVideo(path, name) {
  selectedVideo = path;
  document.querySelectorAll('.video-item').forEach(el => el.classList.remove('active'));
  document.querySelector(`[data-path="${path}"]`)?.classList.add('active');
  document.getElementById('match-title').textContent = name.replace(/_/g,' ').replace('.mp4','').toUpperCase();
  document.getElementById('match-sub').textContent = 'Ready — press Start Analysis';
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

  const depth = document.getElementById('depth-select').value;
  document.getElementById('analyze-btn').disabled = true;
  resetUI();

  const r = await fetch(`${API}/analyze?video=${encodeURIComponent(selectedVideo)}&depth=${depth}&interval=1.0`);
  const job = await r.json();
  jobId = job.job_id;

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws/${jobId}`);

  ws.onmessage = e => handleEvent(JSON.parse(e.data));
  ws.onclose = () => { document.getElementById('analyze-btn').disabled = false; };
  ws.onerror = () => setProgress('WebSocket error');

  pollStatus();
}

function stopAnalysis() {
  if (ws) { ws.close(); ws = null; }
  if (jobId) { fetch(`${API}/jobs/${jobId}`, { method: 'DELETE' }); jobId = null; }
  document.getElementById('analyze-btn').disabled = false;
  setProgress('0%', 0);
}

function resetUI() {
  document.getElementById('events').innerHTML = '<div style="color:var(--muted);font-size:12px">Connecting...</div>';
  document.getElementById('timeline-bar').innerHTML = '';
  document.getElementById('clips-row').innerHTML = '';
  document.getElementById('score-val').textContent = '0-0';
  document.getElementById('phase-val').textContent = '--';
  document.getElementById('events-val').textContent = '0';
  document.getElementById('report-panel').innerHTML = '<span style="color:var(--muted)">Analysis in progress...</span>';
}

async function pollStatus() {
  if (!jobId) return;
  try {
    const r = await fetch(`${API}/status/${jobId}`);
    const s = await r.json();
    if (s.sport && s.sport !== 'unknown') document.getElementById('sport-val').textContent = s.sport;
    if (s.score) document.getElementById('score-val').textContent = s.score;
    if (s.key_events_count) document.getElementById('events-val').textContent = s.key_events_count;
    if (s.status === 'complete') { onComplete(); return; }
    if (s.status === 'error') { document.getElementById('analyze-btn').disabled = false; return; }
    setTimeout(pollStatus, 3000);
  } catch(e) { setTimeout(pollStatus, 3000); }
}

// ── WebSocket Events ────────────────────────────────────────────────────
function handleEvent(data) {
  switch(data.type) {
    case 'connected':
      document.getElementById('events').innerHTML = '';
      break;
    case 'key_event':
      addEvent(data);
      addTimelineMarker(data);
      updateReelPlayer();
      break;
    case 'clip':
      updateReelPlayer();
      break;
    case 'score':
      document.getElementById('score-val').textContent = `${data.home}-${data.away}`;
      break;
    case 'phase':
      document.getElementById('phase-val').textContent = data.phase || '--';
      break;
    case 'progress':
      setProgress(`${data.pct}%`, data.pct);
      break;
    case 'complete':
      onComplete();
      break;
  }
}

function addEvent(ev) {
  const el = document.getElementById('events');
  let css = '';
  if ((ev.event_type||'').includes('GOAL') || ev.event_type==='SIX'||ev.event_type==='FOUR') css='goal';
  else if ((ev.event_type||'').includes('FOUL')||ev.event_type==='WICKET') css='foul';
  else if ((ev.event_type||'').includes('CARD')) css='card-y';
  else if ((ev.event_type||'').includes('VAR')||(ev.event_type||'').includes('DRS')) css='var';

  if (filter !== 'all') {
    const et = ev.event_type||'';
    const match = filter==='CARD' ? et.includes('CARD') : et.includes(filter);
    if (!match) return;
  }

  const team = ev.team ? ` <span style="color:var(--muted)">(${ev.team})</span>` : '';
  const div = document.createElement('div');
  div.className = `event ${css}`;
  div.setAttribute('data-filter', ev.event_type);
  div.innerHTML = `<div class="ts">${ev.timestamp||'?'}</div>
                   <div class="et">${ev.event_type}${team}</div>`;
  el.prepend(div);

  document.getElementById('events-val').textContent = parseInt(document.getElementById('events-val').textContent||0) + 1;
}

function addTimelineMarker(ev) {
  const bar = document.getElementById('timeline-bar');
  const et = ev.event_type||'';
  const colors = {'GOAL':'goal','GOAL_ATTEMPT':'goal','FOUL':'foul','YELLOW_CARD':'card-y','RED_CARD':'foul','VAR_CHECK':'var'};
  const m = document.createElement('div');
  m.className = `marker ${colors[et]||''}`;
  m.style.left = (Math.random()*80+10) + '%';
  m.title = `[${ev.timestamp}] ${et}`;
  bar.appendChild(m);
}

function updateReelPlayer() {
  if (!selectedVideo || !jobId) return;
  const name = selectedVideo.split('/').pop().replace('.mp4','');
  const player = document.getElementById('reel-player');
  player.querySelector('source')?.remove();
  const src = document.createElement('source');
  src.src = `/output/reels/live/${name}_reel.mp4?t=${Date.now()}`;
  src.type = 'video/mp4';
  player.appendChild(src);
  player.load(); player.play().catch(()=>{});
}

function setFilter(f, el) {
  filter = f;
  document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('events').querySelectorAll('.event').forEach(e => {
    const et = e.getAttribute('data-filter')||'';
    e.style.display = (f==='all' || et.includes(f)) ? '' : 'none';
  });
}

async function onComplete() {
  document.getElementById('analyze-btn').disabled = false;
  setProgress('100%', 100);
  document.getElementById('phase-val').textContent = 'complete';

  try {
    const ctx = await fetch(`${API}/context/${jobId}`).then(r=>r.json());
    if (ctx.sport) document.getElementById('sport-val').textContent = ctx.sport;
    if (ctx.score) document.getElementById('score-val').textContent = ctx.score;
    document.getElementById('info-list').innerHTML = [
      `<li>Sport: <b>${ctx.sport||'?'}</b></li>`,
      `<li>Type: <b>${ctx.type||'?'}</b></li>`,
      `<li>Teams: <b>${(ctx.teams||[]).join(' vs ')||'?'}</b></li>`,
    ].join('');
    document.getElementById('stats-list').innerHTML = [
      `<li>Key Events: <b>${ctx.key_events_count||0}</b></li>`,
      `<li>Phase: <b>${ctx.phase||'?'}</b></li>`,
      `<li>Momentum: <b>${ctx.momentum||0}</b></li>`,
    ].join('');
  } catch(e) {}

  try {
    const md = await fetch(`${API}/report/${jobId}`).then(r=>r.ok?r.text():null);
    if (md) document.getElementById('report-panel').innerHTML = md;
  } catch(e) {}
}

function setProgress(label, pct) {
  document.getElementById('progress-label').textContent = label;
  document.getElementById('progress-fill').style.width = (pct||0) + '%';
}

init();
