// VidCore Premium Dashboard Logic — WebSocket integration, Chart.js momentum, Canvas field visualizer
// Auto-detect base path for proxy (AMD Jupyter) or direct access
const API = (() => {
  const p = window.location.pathname;
  const idx = p.lastIndexOf('/dashboard');
  return idx >= 0 ? p.substring(0, idx) : '/';
})();
let ws = null;
let jobId = null;
let selectedVideo = null;
let selectedVideoName = '';
let filter = 'all';

// Telemetry & UI States
let currentSport = 'football';
let currentPhase = 'idle';
let activeView = 'live-console';
let isMobileCropActive = false;

// Chart.js & Canvas instances
let momentumChart = null;
let pitchCanvas = null;
let pitchCtx = null;
let chartTimepoints = [0];
let homeMomentumData = [50];
let awayMomentumData = [50];

// Reel compilation tracking
let studioReelsManifest = null;
let activeStudioFlavor = 'all';

// ── INIT ────────────────────────────────────────────────────────────────
(async function init() {
  switchTab('live-console');
  initChart();
  initPitch();
  await checkHealth();
  await loadVideos();
  
  // Re-draw pitch on window resize
  window.addEventListener('resize', () => {
    resizePitchCanvas();
    drawPitch();
  });
})();

// Health telemetry
async function checkHealth() {
  const dot = document.getElementById('health-dot');
  const txt = document.getElementById('health-text');
  try {
    const r = await fetch(`${API}/health`);
    const h = await r.json();
    if (h.vllm === 'connected') {
      dot.className = 'dot on';
      txt.textContent = 'vLLM Core Online';
    } else {
      dot.className = 'dot loading';
      txt.textContent = 'vLLM Standby (vLLM Offline)';
    }
  } catch(e) {
    dot.className = 'dot off';
    txt.textContent = 'Orchestration Engine Offline';
  }
}

// ── TAB SWITCHING ───────────────────────────────────────────────────────
function switchTab(tabId) {
  activeView = tabId;
  
  // Update button classes
  document.querySelectorAll('.nav-tabs .tab-btn').forEach(btn => {
    if (btn.getAttribute('data-tab') === tabId) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  // Show/hide view sections
  document.querySelectorAll('.view-section').forEach(section => {
    if (section.id === tabId) {
      section.classList.add('active');
    } else {
      section.classList.remove('active');
    }
  });

  // Special draw updates
  if (tabId === 'live-console') {
    setTimeout(() => {
      resizePitchCanvas();
      drawPitch();
      if (momentumChart) momentumChart.update();
    }, 50);
  }
}

// ── VIDEO LIBRARY & UPLOADS ─────────────────────────────────────────────
async function loadVideos() {
  const videoList = document.getElementById('video-list');
  const libraryList = document.getElementById('library-full-list');
  
  try {
    const r = await fetch(`${API}/videos`);
    const videos = await r.json();
    
    if (!videos.length) {
      const emptyHTML = '<div class="empty-state">No videos available<br>Upload sports footage to analyze</div>';
      videoList.innerHTML = emptyHTML;
      if (libraryList) libraryList.innerHTML = emptyHTML;
      return;
    }

    const html = videos.map(v => `
      <div class="vid-item" onclick="selectVideo('${v.path}','${v.name}')" data-path="${v.path}">
        <div class="vid-name-container">
          <span>🎬</span>
          <span class="vid-name" title="${v.name}">${v.name}</span>
        </div>
        <span class="vid-size">${v.size_mb} MB</span>
      </div>
    `).join('');

    videoList.innerHTML = html;
    if (libraryList) {
      libraryList.innerHTML = videos.map(v => `
        <div class="vid-item" style="padding: 14px;" onclick="selectVideo('${v.path}','${v.name}'); switchTab('live-console');" data-path="${v.path}">
          <div class="vid-name-container">
            <span style="font-size: 18px;">⚽</span>
            <div style="display:flex; flex-direction:column; gap: 2px;">
              <span class="vid-name" style="font-size: 13px; font-weight:600;">${v.name}</span>
              <span style="font-size: 10px; color: var(--text-muted);">Local Sports Source File</span>
            </div>
          </div>
          <span style="font-size: 12px; font-weight:700; color: var(--accent-blue);">${v.size_mb} MB</span>
        </div>
      `).join('');
    }
  } catch(e) {
    const errorHTML = '<div class="empty-state" style="color:var(--accent-crimson);">Failed to load library items</div>';
    videoList.innerHTML = errorHTML;
    if (libraryList) libraryList.innerHTML = errorHTML;
  }
}

function selectVideo(path, name) {
  selectedVideo = path;
  selectedVideoName = name;
  document.querySelectorAll('.vid-item').forEach(el => {
    if (el.getAttribute('data-path') === path) el.classList.add('active');
    else el.classList.remove('active');
  });
  setStat('type-val', 'File: ' + name);
}

async function uploadVideo() {
  const fileInput = document.getElementById('upload-input');
  const file = fileInput.files[0];
  if (!file) return;

  const zone = document.querySelector('.upload-zone');
  const originalText = zone.innerHTML;
  zone.innerHTML = '<div class="player-spinner" style="margin: 0 auto 8px auto;"></div><span class="upload-text">Transferring video...</span>';

  try {
    const form = new FormData();
    form.append('file', file);
    await fetch(`${API}/upload`, { method: 'POST', body: form });
    await loadVideos();
  } catch(e) {
    alert('Upload failed: server connection issue.');
  } finally {
    zone.innerHTML = originalText;
  }
}

// ── CHART.JS MATCH MOMENTUM GRAPH ───────────────────────────────────────
function initChart() {
  const ctx = document.getElementById('momentum-chart');
  if (!ctx) return;

  momentumChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: chartTimepoints,
      datasets: [
        {
          label: 'Home Momentum',
          data: homeMomentumData,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.15)',
          fill: true,
          tension: 0.4,
          borderWidth: 2.5,
          pointRadius: 0,
          pointHoverRadius: 4
        },
        {
          label: 'Away Momentum',
          data: awayMomentumData,
          borderColor: '#f97316',
          backgroundColor: 'rgba(249, 115, 22, 0.15)',
          fill: true,
          tension: 0.4,
          borderWidth: 2.5,
          pointRadius: 0,
          pointHoverRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#64748b', font: { size: 9 } }
        },
        y: {
          min: 0,
          max: 100,
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#64748b', font: { size: 9 }, stepSize: 25 }
        }
      }
    }
  });
}

function updateMomentumChart(timeStr, homeVal) {
  if (!momentumChart) return;
  const home = Math.min(Math.max(homeVal, 0), 100);
  const away = 100 - home;
  
  chartTimepoints.push(timeStr);
  homeMomentumData.push(home);
  awayMomentumData.push(away);

  // Keep last 40 data points
  if (chartTimepoints.length > 40) {
    chartTimepoints.shift();
    homeMomentumData.shift();
    awayMomentumData.shift();
  }

  momentumChart.update();
}

// ── TACTICAL PITCH CANVAS DRAWING ────────────────────────────────────────
function initPitch() {
  pitchCanvas = document.getElementById('pitch-canvas');
  if (!pitchCanvas) return;
  pitchCtx = pitchCanvas.getContext('2d');
  resizePitchCanvas();
  drawPitch();
}

function resizePitchCanvas() {
  if (!pitchCanvas) return;
  const rect = pitchCanvas.parentElement.getBoundingClientRect();
  let w = rect.width || pitchCanvas.clientWidth || 400;
  let h = rect.height || pitchCanvas.clientHeight || 260;
  pitchCanvas.width = w * window.devicePixelRatio;
  pitchCanvas.height = h * window.devicePixelRatio;
  pitchCtx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
}

function drawPitch() {
  if (!pitchCtx || !pitchCanvas) return;
  
  const w = pitchCanvas.width / window.devicePixelRatio;
  const h = pitchCanvas.height / window.devicePixelRatio;
  
  if (w < 10 || h < 10) { setTimeout(drawPitch, 500); return; }
  const h = pitchCanvas.height / window.devicePixelRatio;
  pitchCtx.clearRect(0, 0, w, h);

  // Render Green Background
  pitchCtx.fillStyle = '#0f2416';
  pitchCtx.fillRect(0, 0, w, h);

  if (currentSport.toLowerCase().includes('cricket')) {
    drawCricketOval(w, h);
  } else {
    drawSoccerField(w, h);
  }
  
  drawDynamicPhaseHighlights(w, h);
}

function drawSoccerField(w, h) {
  const pad = 12;
  const fw = Math.max(w - pad * 2, 10);
  const fh = Math.max(h - pad * 2, 10);

  pitchCtx.strokeStyle = 'rgba(255,255,255,0.18)';
  pitchCtx.lineWidth = 1.5;

  // Outer Bound
  pitchCtx.strokeRect(pad, pad, fw, fh);

  // Center Line
  pitchCtx.beginPath();
  pitchCtx.moveTo(w / 2, pad);
  pitchCtx.lineTo(w / 2, h - pad);
  pitchCtx.stroke();

  // Center Circle
  pitchCtx.beginPath();
    pitchCtx.arc(w / 2, h / 2, Math.max(Math.min(fw, fh) * 0.15, 2), 0, Math.PI * 2);
  pitchCtx.stroke();

  // Penalty Areas (Left / Right)
  const boxW = fw * 0.16;
  const boxH = fh * 0.55;
  pitchCtx.strokeRect(pad, (h - boxH) / 2, boxW, boxH);
  pitchCtx.strokeRect(w - pad - boxW, (h - boxH) / 2, boxW, boxH);

  // Wickets/Goals Indicator
  pitchCtx.fillStyle = 'rgba(255,255,255,0.1)';
  pitchCtx.fillRect(pad - 2, (h - 24)/2, 2, 24);
  pitchCtx.fillRect(w - pad, (h - 24)/2, 2, 24);
}

function drawCricketOval(w, h) {
  const cx = w / 2;
  const cy = h / 2;
  
  pitchCtx.strokeStyle = 'rgba(255,255,255,0.18)';
  pitchCtx.lineWidth = 1.5;

  // Oval Boundary
  pitchCtx.beginPath();
  pitchCtx.ellipse(cx, cy, w * 0.42, h * 0.42, 0, 0, Math.PI * 2);
  pitchCtx.stroke();

  // 30-Yard Circle
  pitchCtx.beginPath();
  pitchCtx.ellipse(cx, cy, w * 0.28, h * 0.28, 0, 0, Math.PI * 2);
  pitchCtx.stroke();

  // Center Pitch
  const pw = 12;
  const ph = 34;
  pitchCtx.fillStyle = '#c8a261'; // Clay pitch color
  pitchCtx.fillRect(cx - pw/2, cy - ph/2, pw, ph);

  // Wickets
  pitchCtx.fillStyle = '#fff';
  pitchCtx.fillRect(cx - 3, cy - ph/2, 6, 2);
  pitchCtx.fillRect(cx - 3, cy + ph/2 - 2, 6, 2);
}

function drawDynamicPhaseHighlights(w, h) {
  const pad = 12;
  const fw = w - pad * 2;
  const fh = h - pad * 2;

  // Highlights based on Match Phase
  if (currentPhase === 'attack_final_third') {
    // Glow attack zones (Left penalty box + Right penalty box)
    const gradient = pitchCtx.createRadialGradient(w - pad - 30, h / 2, 10, w - pad, h / 2, 60);
    gradient.addColorStop(0, 'rgba(239, 68, 68, 0.35)');
    gradient.addColorStop(1, 'rgba(239, 68, 68, 0)');
    pitchCtx.fillStyle = gradient;
    pitchCtx.fillRect(w / 2, pad, w / 2 - pad, fh);
    
    // Pulse animation
    pitchCtx.shadowColor = '#ef4444';
    pitchCtx.shadowBlur = 10;
    pitchCtx.fillStyle = 'rgba(239, 68, 68, 0.4)';
    pitchCtx.beginPath();
    pitchCtx.arc(w - pad - 20, h / 2, 6, 0, Math.PI * 2);
    pitchCtx.fill();
    pitchCtx.shadowBlur = 0;
  } else if (currentPhase === 'open_play') {
    // Highlight Center Circle
    pitchCtx.fillStyle = 'rgba(37, 99, 235, 0.08)';
    pitchCtx.beginPath();
  pitchCtx.arc(w / 2, h / 2, Math.max(Math.min(fw, fh) * 0.15, 2), 0, Math.PI * 2);
    pitchCtx.fill();
  } else if (currentPhase === 'commercial' || currentPhase === 'half_time') {
    // Dim the pitch
    pitchCtx.fillStyle = 'rgba(0,0,0,0.5)';
    pitchCtx.fillRect(0, 0, w, h);
  }
}

// ── PLAY DETECT OVERLAYS & 9:16 CROP ──
function toggleMobileCrop() {
  isMobileCropActive = !isMobileCropActive;
  const overlay = document.getElementById('crop-overlay');
  if (overlay) overlay.style.display = isMobileCropActive ? 'grid' : 'none';
}

// ── ORCHESTRATION ENGINE RUNNER (WS / REST) ─────────────────────────────
async function startAnalysis() {
  if (!selectedVideo) {
    alert('Please select a video file from the list first.');
    return;
  }
  
  stopAnalysis();
  
  // Set UI state to active
  const startBtn = document.getElementById('analyze-btn');
  startBtn.disabled = true;
  startBtn.textContent = '⚡ Spinning Up Agents...';
  document.getElementById('stop-btn').style.display = 'block';
  document.getElementById('progress-area').style.display = 'flex';
  
  resetDashboardUI();
  
  const depth = document.getElementById('depth-select').value;
  const r = await fetch(`${API}/analyze?video=${encodeURIComponent(selectedVideo)}&depth=${depth}&interval=1.0`, { method: 'POST' });
  const job = await r.json();
  jobId = job.job_id;

  // Load source video immediately
  const pw = document.getElementById('player-window');
  if (pw) pw.innerHTML = `
    <video class="video-element" id="live-player" controls autoplay muted playsinline>
      <source src="${API}/videos/${selectedVideoName}" type="video/mp4">
    </video>
  `;

  // Initialize Socket connection
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}${API}/ws/${jobId}`);
  ws.onmessage = e => handleWebSocketEvent(JSON.parse(e.data));
  
  ws.onclose = () => {
    startBtn.disabled = false;
    startBtn.textContent = '▶ Run Core Orchestration';
    document.getElementById('stop-btn').style.display = 'none';
  };
  
  ws.onopen = () => {
    if (window._sse) { window._sse.close(); window._sse = null; }
  };

  ws.onerror = () => {
    startBtn.textContent = '⚠️ Connection Failed';
  };

  window._sse = new EventSource(`${API}/sse/${jobId}`);
  window._sse.onmessage = e => {
    if (ws && ws.readyState === WebSocket.OPEN) return;
    try { handleWebSocketEvent(JSON.parse(e.data)); } catch(_){}
  };

  pollJobStatus();
}

function stopAnalysis() {
  if (ws) {
    ws.close();
    ws = null;
  }
  if (jobId) {
    fetch(`${API}/jobs/${jobId}`, { method: 'DELETE' }).catch(()=>{});
    jobId = null;
  }
  document.getElementById('analyze-btn').disabled = false;
  document.getElementById('analyze-btn').textContent = '▶ Run Core Orchestration';
  document.getElementById('stop-btn').style.display = 'none';
}

function resetDashboardUI() {
  document.getElementById('events').innerHTML = '<div class="empty-state"><div class="player-spinner"></div>Connecting to telemetry logs...</div>';
  document.getElementById('timeline-bar').innerHTML = '<div class="timeline-progress-fill" id="timeline-progress"></div>';
  document.getElementById('clips-row').innerHTML = '<div class="empty-state">Waiting for highlights...</div>';
  
  const wrap = document.getElementById('player-window');
  wrap.innerHTML = `
    <div class="player-placeholder" id="player-placeholder">
      <div class="player-spinner" id="player-spinner"></div>
      <span id="player-placeholder-msg">Spinning up neural engine...</span>
    </div>
    <div class="crop-overlay-916" id="crop-overlay">
      <div class="crop-pillar crop-pillar-left"><span class="crop-label-tag">Cut Pillar</span></div>
      <div></div>
      <div class="crop-pillar crop-pillar-right"><span class="crop-label-tag">Cut Pillar</span></div>
    </div>
  `;
  document.getElementById('crop-overlay').style.display = isMobileCropActive ? 'grid' : 'none';

  setStat('score-val', '0-0');
  setStat('phase-val', 'Running');
  setStat('events-val', '0');
  setStat('sport-val', 'Sport: Classification...');
  
  document.getElementById('report-content').innerHTML = '<div class="empty-state"><div class="player-spinner"></div>Analyzing video frames...</div>';
  document.getElementById('stats-content').innerHTML = '<div class="empty-state">No stats yet</div>';
  document.getElementById('info-content').innerHTML = '<div class="empty-state">No info yet</div>';
  
  // Reset chart
  chartTimepoints.length = 0;
  homeMomentumData.length = 0;
  awayMomentumData.length = 0;
  chartTimepoints.push(0);
  homeMomentumData.push(50);
  awayMomentumData.push(50);
  if (momentumChart) momentumChart.update();
}

async function pollJobStatus() {
  if (!jobId) return;
  try {
    const r = await fetch(`${API}/status/${jobId}`);
    const s = await r.json();
    
    if (s.sport && s.sport !== 'unknown') {
      currentSport = s.sport;
      setStat('sport-val', 'Sport: ' + s.sport);
      document.getElementById('sport-field-label').textContent = s.sport.toUpperCase() + ' GRAPH';
      drawPitch();
    }
    
    if (s.score) setStat('score-val', s.score);
    if (s.phase) {
      currentPhase = s.phase;
      setStat('phase-val', s.phase);
      drawPitch();
    }
    if (s.key_events_count !== undefined) setStat('events-val', s.key_events_count);
    
    if (s.status === 'complete') {
      await onProcessingComplete();
      return;
    }
    if (s.status === 'error') {
      onProcessingError(s.error);
      return;
    }
    
    setTimeout(pollJobStatus, 2000);
  } catch(e) {
    setTimeout(pollJobStatus, 3000);
  }
}

// ── WEBSOCKET EVENT PARSING ─────────────────────────────────────────────
function handleWebSocketEvent(data) {
  switch(data.type) {
    case 'connected':
      document.getElementById('events').innerHTML = '';
      document.getElementById('analyze-btn').textContent = '⚡ Analyzing frames...';
      break;
      
    case 'scene':
      flashAgent('scene');
      if (data.activity) {
        // Feed rolling momentum chart dynamically based on keywords
        let randMomentum = 50;
        const act = data.activity.toLowerCase();
        if (act.includes('attack') || act.includes('shot') || act.includes('goal')) {
          randMomentum = 75 + Math.random() * 15;
        } else if (act.includes('save') || act.includes('foul') || act.includes('defend')) {
          randMomentum = 25 - Math.random() * 15;
        } else {
          randMomentum = 40 + Math.random() * 20;
        }
        updateMomentumChart(data.timestamp, randMomentum);
      }
      break;

    case 'key_event':
      flashAgent('event');
      flashAgent('reasoning');
      addTickerEvent(data);
      addTimelineMarker(data);
      
      // Update chart momentum based on event
      let eventMom = 50;
      if (data.event_type.includes('GOAL') || data.event_type === 'SIX' || data.event_type === 'FOUR') {
        eventMom = (data.team && data.team === 'Away') ? 10 : 90;
      } else if (data.event_type.includes('FOUL') || data.event_type.includes('WICKET')) {
        eventMom = (data.team && data.team === 'Away') ? 80 : 20;
      }
      updateMomentumChart(data.timestamp || data.global_time, eventMom);
      break;
      
    case 'clip':
      addCarouselClip(data.event_type, data.timestamp, data.path);
      break;
      
    case 'score':
      setStat('score-val', `${data.home}-${data.away}`);
      break;

    case 'status':
      if (data.phase) { currentPhase = data.phase; setStat('phase-val', currentPhase); }
      if (data.score) setStat('score-val', data.score);
      if (data.sport) setStat('sport-val', data.sport);
      drawPitch();
      break;

    case 'phase':
      currentPhase = data.phase || 'open_play';
      setStat('phase-val', currentPhase);
      drawPitch();
      break;
      
    case 'progress':
      setProgressBar(data.pct);
      break;
      
    case 'complete':
      onProcessingComplete();
      break;
      
    case 'error':
      onProcessingError(data.message);
      break;
  }
}

// Flash Agent Indicator tags
function flashAgent(agentName) {
  const el = document.getElementById(`agent-${agentName}-status`);
  if (!el) return;
  el.className = 'agent-status-tag active';
  el.textContent = 'ACTIVE';
  
  if (el.timeoutId) clearTimeout(el.timeoutId);
  el.timeoutId = setTimeout(() => {
    el.className = 'agent-status-tag';
    el.textContent = 'IDLE';
  }, 1000);
}

// Telemetry timeline markers
function addTimelineMarker(ev) {
  const bar = document.getElementById('timeline-bar');
  if (!bar) return;
  const et = ev.event_type || '';
  const ts = ev.timestamp || '0s';
  const sec = parseFloat(ts);
  
  let cls = 'timeline-marker';
  if (et.includes('GOAL') || et === 'SIX' || et === 'FOUR') cls += ' goal';
  else if (et.includes('FOUL') || et.includes('WICKET')) cls += ' foul';
  else if (et.includes('CARD')) cls += ' card-y';
  else if (et.includes('VAR')) cls += ' var';

  // calculate position based on arbitrary length or random spreads
  const marker = document.createElement('div');
  marker.className = cls;
  marker.style.left = (Math.random() * 85 + 8) + '%';
  marker.title = `[${ts}] ${et} — ${ev.description || ''}`;
  
  // Seek trigger
  marker.onclick = () => {
    const video = document.getElementById('live-player');
    if (video) {
      video.currentTime = sec;
      video.play();
    }
  };
  
  bar.appendChild(marker);
}

// Dynamic ticker logger
function addTickerEvent(ev) {
  const ticker = document.getElementById('events');
  if (ticker.querySelector('.empty-state')) ticker.innerHTML = '';

  const et = ev.event_type || 'EVENT';
  const ts = ev.timestamp || '0s';
  const desc = ev.description || ev.desc || 'Match event recorded';
  const team = ev.team ? ` (${ev.team})` : '';

  let css = 'ticker-item';
  if (et.includes('GOAL') || et === 'SIX' || et === 'FOUR') css += ' goal';
  else if (et.includes('FOUL') || et.includes('WICKET')) css += ' foul';
  else if (et.includes('CARD')) css += ' card-y';
  else if (et.includes('VAR')) css += ' var';

  const div = document.createElement('div');
  div.className = css;
  div.setAttribute('data-filter', et);
  div.innerHTML = `
    <div class="ticker-item-header">
      <span class="ticker-item-title">${et}${team}</span>
      <span class="ticker-item-time">${ts}</span>
    </div>
    <div class="ticker-item-desc">${desc}</div>
  `;
  
  // Seek trigger
  div.onclick = () => {
    const video = document.getElementById('live-player');
    if (video) {
      video.currentTime = parseFloat(ts);
      video.play();
    }
  };

  ticker.prepend(div);
  setStat('events-val', parseInt(document.getElementById('events-val').textContent || 0) + 1);
}

// Carousel of cut clip cards
function addCarouselClip(type, timestamp, path) {
  const row = document.getElementById('clips-row');
  if (row.querySelector('.empty-state')) row.innerHTML = '';

  const colors = {
    GOAL: '#10b981', GOAL_ATTEMPT: '#2563eb', FOUL: '#ef4444', 
    YELLOW_CARD: '#f59e0b', RED_CARD: '#ef4444', VAR_CHECK: '#8b5cf6',
    SIX: '#10b981', WICKET: '#ef4444', FOUR: '#2563eb'
  };

  const card = document.createElement('div');
  card.className = 'clip-card';
  card.innerHTML = `
    <div class="clip-card-overlay">
      <span class="clip-badge" style="background:${colors[type] || '#64748b'}">${type}</span>
      <span class="clip-time">${timestamp}</span>
    </div>
  `;

  // Seek trigger
  card.onclick = () => {
    const video = document.getElementById('live-player');
    if (video) {
      video.currentTime = parseFloat(timestamp);
      video.play();
    }
  };

  row.appendChild(card);
}

function setFilter(f, el) {
  filter = f;
  document.querySelectorAll('#ticker-filters .ticker-chip').forEach(c => c.classList.remove('active'));
  el.classList.add('active');

  document.querySelectorAll('.ticker-item').forEach(e => {
    const et = e.getAttribute('data-filter') || '';
    e.style.display = (f === 'all' || et.includes(f)) ? 'flex' : 'none';
  });
}

// ── COMPLETED & ERROR ACTIONS ───────────────────────────────────────────
async function onProcessingComplete() {
  document.getElementById('analyze-btn').disabled = false;
  document.getElementById('analyze-btn').textContent = '✓ Orchestration Complete';
  document.getElementById('stop-btn').style.display = 'none';
  setStat('phase-val', 'Complete');
  setProgressBar(100);
  flashAgent('commentary');

  // Load Main Player Video
  const playerWindow = document.getElementById('player-window');
  const cleanName = selectedVideoName.replace('.mp4','');
  playerWindow.innerHTML = `
    <video class="video-element" id="live-player" controls autoplay muted>
      <source src="${API}/output/reels/live/${cleanName}_reel.mp4?t=${Date.now()}" type="video/mp4">
      <source src="${API}/videos/${selectedVideoName}" type="video/mp4">
    </video>
    <div class="crop-overlay-916" id="crop-overlay">
      <div class="crop-pillar crop-pillar-left"><span class="crop-label-tag">Cut Pillar</span></div>
      <div></div>
      <div class="crop-pillar crop-pillar-right"><span class="crop-label-tag">Cut Pillar</span></div>
    </div>
  `;
  document.getElementById('crop-overlay').style.display = isMobileCropActive ? 'grid' : 'none';

  // Load Context
  try {
    const ctx = await fetch(`${API}/context/${jobId}`).then(r => r.json());
    setStat('sport-val', 'Sport: ' + (ctx.sport || 'unknown'));
    setStat('type-val', 'Match Type: ' + (ctx.type || 'unknown'));

    // Populate Report Details Column
    document.getElementById('info-content').innerHTML = `
      <div class="info-row"><span class="info-label">League</span><span class="info-value">${ctx.league || 'Unknown'}</span></div>
      <div class="info-row"><span class="info-label">Teams</span><span class="info-value">${(ctx.teams || []).join(' vs ') || 'Unknown'}</span></div>
      <div class="info-row"><span class="info-label">Location</span><span class="info-value">${ctx.location || 'Unknown'}</span></div>
      <div class="info-row"><span class="info-label">Final Score</span><span class="info-value">${ctx.score || '0-0'}</span></div>
    `;

    document.getElementById('stats-content').innerHTML = `
      <div class="info-row"><span class="info-label">Events Processed</span><span class="info-value">${ctx.key_events_count || 0}</span></div>
      <div class="info-row"><span class="info-label">Match Phase</span><span class="info-value" style="color:var(--accent-purple); text-transform:uppercase;">${ctx.phase || 'Unknown'}</span></div>
      <div class="info-row"><span class="info-label">Momentum Bias</span><span class="info-value" style="color:var(--accent-blue);">${ctx.momentum || 50}% Home</span></div>
    `;
  } catch(e) {}

  // Load Report Markdown
  try {
    const reportText = await fetch(`${API}/report/${jobId}`).then(r => r.text());
    // Convert basic markdown tags to styled HTML
    let formattedText = reportText
      .replace(/# (.*)/g, '<h1>$1</h1>')
      .replace(/## (.*)/g, '<h2>$1</h2>')
      .replace(/### (.*)/g, '<h3>$1</h3>')
      .replace(/- \*\*(.*?)\*\*/g, '<li><strong>$1</strong>')
      .replace(/\n\n/g, '<p>')
      .replace(/---\n/g, '<hr>');
    
    document.getElementById('report-content').innerHTML = formattedText;
  } catch(e) {
    document.getElementById('report-content').innerHTML = '<div class="empty-state">Match summary report unavailable</div>';
  }

  // Load Highlights Studio reels manifest
  try {
    studioReelsManifest = await fetch(`${API}/reels/${jobId}`).then(r => r.json());
    loadStudioReel('all');
    
    // Populate segment lists in Studio
    const clipsList = document.getElementById('studio-clips-list');
    if (studioReelsManifest.clips && studioReelsManifest.clips.length) {
      clipsList.innerHTML = studioReelsManifest.clips.map((clip, i) => `
        <div class="vid-item" style="padding: 10px;" onclick="seekStudioPlayer('${clip.timestamp}')">
          <div class="vid-name-container">
            <span style="color: var(--accent-emerald)">●</span>
            <div style="display:flex; flex-direction:column; gap:2px;">
              <span class="vid-name" style="font-size:11px; font-weight:600;">Highlight Segment #${i+1}</span>
              <span style="font-size:9px; color:var(--text-muted);">${clip.event_type} at ${clip.timestamp}</span>
            </div>
          </div>
          <span style="font-size:10px; color:var(--accent-blue)">Seek 🔍</span>
        </div>
      `).join('');
    } else {
      clipsList.innerHTML = '<div class="empty-state">No compiled segments found</div>';
    }
  } catch(e) {}
}

function onProcessingError(msg) {
  document.getElementById('analyze-btn').disabled = false;
  document.getElementById('analyze-btn').textContent = '⚠️ Orchestration Failed';
  document.getElementById('stop-btn').style.display = 'none';
  setStat('phase-val', 'Error');
  document.getElementById('report-content').innerHTML = `<div class="empty-state" style="color:var(--accent-crimson);">Engine Execution Error:<br>${msg || 'Unknown failure'}</div>`;
}

// ── HIGHLIGHTS STUDIO TABS ──────────────────────────────────────────────
function loadStudioReel(flavor, btnElement) {
  activeStudioFlavor = flavor;
  
  if (btnElement) {
    // Style tabs
    btnElement.parentElement.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    btnElement.classList.add('active');
  }

  const studioPlayer = document.getElementById('studio-player-window');
  const cleanName = selectedVideoName.replace('.mp4','');
  document.getElementById('studio-playback-status').textContent = 'Active compilation flavor: ' + flavor.toUpperCase();

  // Map flavors to dynamic backend static paths
  let reelSource = `${API}/output/reels/${cleanName}_${flavor}.mp4?t=${Date.now()}`;
  if (flavor === 'all' && studioReelsManifest && studioReelsManifest.reel_url) {
    reelSource = studioReelsManifest.reel_url + `?t=${Date.now()}`;
  }

  studioPlayer.innerHTML = `
    <video class="video-element" id="studio-player" controls autoplay muted>
      <source src="${reelSource}" type="video/mp4">
    </video>
    <div class="crop-overlay-916" id="studio-crop-overlay">
      <div class="crop-pillar crop-pillar-left"><span class="crop-label-tag">Cut Pillar</span></div>
      <div></div>
      <div class="crop-pillar crop-pillar-right"><span class="crop-label-tag">Cut Pillar</span></div>
    </div>
  `;
  
  // Show vertical crop overlay specifically for social vertical compilations
  const cropOverlay = document.getElementById('studio-crop-overlay');
  if (cropOverlay) {
    cropOverlay.style.display = (flavor === 'social_goals') ? 'grid' : 'none';
  }
}

function seekStudioPlayer(timeStr) {
  const studioVid = document.getElementById('studio-player');
  if (studioVid) {
    studioVid.currentTime = parseFloat(timeStr);
    studioVid.play();
  }
}

function downloadActiveReel() {
  if (!selectedVideo) return;
  const cleanName = selectedVideoName.replace('.mp4','');
  
  let downloadUrl = `${API}/output/reels/${cleanName}_${activeStudioFlavor}.mp4`;
  if (activeStudioFlavor === 'all' && studioReelsManifest && studioReelsManifest.reel_url) {
    downloadUrl = studioReelsManifest.reel_url;
  }
  
  window.open(downloadUrl, '_blank');
}

// ── DOWNLOAD EXPORTS ──────────────────────────────────────────────────
function downloadCSV() {
  if (!jobId) {
    alert('No telemetry session found. Please run core orchestration first.');
    return;
  }
  window.open(`${API}/csv/${jobId}`, '_blank');
}

function downloadReport() {
  if (!jobId) {
    alert('No report found. Please run core orchestration first.');
    return;
  }
  
  fetch(`${API}/report/${jobId}`)
    .then(r => r.text())
    .then(md => {
      const blob = new Blob([md], { type: 'text/markdown' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${selectedVideoName.replace('.mp4','')}_analytics_report.md`;
      a.click();
    })
    .catch(() => alert('Report could not be downloaded.'));
}

function copyReportText() {
  const el = document.getElementById('report-content');
  if (!el || el.innerText.includes('No report generated')) {
    alert('No report content available to copy.');
    return;
  }
  navigator.clipboard.writeText(el.innerText)
    .then(() => alert('Summary report copied to clipboard.'))
    .catch(() => alert('Failed to copy to clipboard.'));
}

// ── HELPERS ─────────────────────────────────────────────────────────────
function setStat(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function setProgressBar(pct) {
  const fill = document.getElementById('progress-fill');
  const label = document.getElementById('progress-label');
  const bar = document.getElementById('timeline-progress');
  
  if (fill) fill.style.width = (pct || 0) + '%';
  if (label) label.textContent = (pct || 0) + '%';
  if (bar) bar.style.width = (pct || 0) + '%';
}
