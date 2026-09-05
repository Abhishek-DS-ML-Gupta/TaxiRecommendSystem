/**
 * app.js — Main application controller.
 * Orchestrates map, dashboard, API calls, and UI state.
 */

// ─── App State ────────────────────────────────────────────────────────────────
const AppState = {
  clickedLat: null,
  clickedLon: null,
  currentRecommendations: [],
  isLoading: false,
  selectedTaxiIdx: 0,
  dataReady: false,

  // Controls
  timeLimit: 10,
  numTaxis: 1,
  scenario: 'normal',
  showHeatmap: true,
  showNetwork: true,

  activeTab: 'recommend',
};

// ─── DOM References ────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ─── Initialize App ────────────────────────────────────────────────────────────
async function initApp() {
  // Fetch configuration from backend
  await fetchConfig();

  // Init Leaflet map
  MapModule.init(onMapClick, AppState.cartoApiKey);

  // UI control bindings
  bindControls();

  // Tab switching
  bindTabs();

  // Check server health and load map data
  await waitForServer();
}

async function fetchConfig() {
  try {
    const resp = await fetch('/api/config');
    const data = await resp.json();
    AppState.cartoApiKey = data.carto_api_key || '';
  } catch (err) {
    console.error("Failed to load frontend config:", err);
    AppState.cartoApiKey = '';
  }
}

async function waitForServer() {
  setStatus('loading', 'Loading road network...');
  $('map-loading').classList.remove('hidden');
  $('map-loading').querySelector('.loading-text').textContent = 'Loading 38K road segments...';
  $('map-loading').querySelector('.loading-sub').textContent = 'Beijing 3rd Ring Road network';

  let attempts = 0;
  const maxAttempts = 30;

  async function checkHealth() {
    try {
      const resp = await fetch('/health');
      const data = await resp.json();
      if (data.data_loaded) {
        AppState.dataReady = true;
        setStatus('ready', `Ready · ${(data.road_segments || 0).toLocaleString()} segments`);
        onDataReady();
        return true;
      }
    } catch (e) {
      // Server might still be starting
    }
    return false;
  }

  while (attempts < maxAttempts) {
    const ready = await checkHealth();
    if (ready) break;
    await delay(2000);
    attempts++;
    $('map-loading').querySelector('.loading-text').textContent =
      `Initializing... (${attempts * 2}s)`;
  }

  if (!AppState.dataReady) {
    setStatus('error', 'Server not responding');
    $('map-loading').querySelector('.loading-text').textContent = 'Server connection failed';
    $('map-loading').querySelector('.loading-sub').textContent = 'Is uvicorn running? Check terminal.';
  }
}

async function onDataReady() {
  // Load map layers in parallel
  const [edgeCount, hotspotCount] = await Promise.all([
    MapModule.loadRoadNetwork(),
    MapModule.loadHeatmap(),
  ]);

  // Load network summary for overview tab
  Dashboard.loadNetworkSummary();

  // Hide loading overlay
  $('map-loading').classList.add('hidden');

  // Show instruction
  const inst = document.querySelector('.map-instruction');
  if (inst) inst.classList.remove('hidden');

  console.log(`✅ Loaded ${edgeCount} road edges, ${hotspotCount} hotspot points`);
}

// ─── Map Click Handler ────────────────────────────────────────────────────────
function onMapClick(lat, lon) {
  if (!AppState.dataReady) return;

  AppState.clickedLat = lat;
  AppState.clickedLon = lon;

  MapModule.setStartMarker(lat, lon);

  // Hide instruction
  const inst = document.querySelector('.map-instruction');
  if (inst) inst.classList.add('hidden');

  // Auto-trigger recommendation
  requestRecommendation();
}

// ─── Request Recommendation ────────────────────────────────────────────────────
async function requestRecommendation() {
  if (AppState.clickedLat === null || AppState.isLoading) return;

  AppState.isLoading = true;
  $('btn-recommend').disabled = true;
  $('btn-recommend').innerHTML = '<i data-lucide="loader" class="lucide-spin" style="margin-right:6px"></i> Computing Route...';
  if (window.lucide) lucide.createIcons();

  setStatus('loading', 'Running RRS algorithm...');

  try {
    const payload = {
      lat: AppState.clickedLat,
      lon: AppState.clickedLon,
      time_limit: AppState.timeLimit,
      num_taxis: AppState.numTaxis,
      scenario: AppState.scenario,
    };

    const resp = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || 'Recommendation failed');
    }

    const data = await resp.json();
    AppState.currentRecommendations = data.recommendations;

    // Update map
    MapModule.drawRoutes(data.recommendations);

    // Update dashboard
    Dashboard.updateMetrics(data.recommendations);
    Dashboard.renderTaxiList(data.recommendations, onTaxiSelected);

    // Update algo badge
    const algoEl = $('algo-badge');
    if (algoEl) {
      algoEl.innerHTML = data.recommendations.length > 1 ? '<i data-lucide="zap" style="margin-right:4px"></i> SRRS Algorithm' : '<i data-lucide="search" style="margin-right:4px"></i> RRS Algorithm';
      algoEl.title = data.algorithm || '';
      if (window.lucide) lucide.createIcons();
    }

    setStatus('ready', `✓ Route ready · ${data.recommendations.length} taxi${data.recommendations.length > 1 ? 's' : ''}`);
  } catch (err) {
    console.error('Recommendation error:', err);
    setStatus('error', err.message || 'Error — try clicking closer to a road');
    showToast('⚠️ ' + (err.message || 'Recommendation failed. Try a different location.'));
  } finally {
    AppState.isLoading = false;
    $('btn-recommend').disabled = false;
    $('btn-recommend').innerHTML = '<i data-lucide="car" style="margin-right:6px"></i> Get Route';
    if (window.lucide) lucide.createIcons();
  }
}

// ─── Taxi Selection from List ─────────────────────────────────────────────────
function onTaxiSelected(idx) {
  AppState.selectedTaxiIdx = idx;
  // Re-draw with selected taxi highlighted (handled by CSS active class in dashboard)
}

// ─── Control Bindings ─────────────────────────────────────────────────────────
function bindControls() {
  // Time limit slider
  const timeLimitSlider = $('slider-time-limit');
  const timeLimitVal = $('val-time-limit');
  if (timeLimitSlider) {
    timeLimitSlider.addEventListener('input', e => {
      AppState.timeLimit = parseFloat(e.target.value);
      timeLimitVal.textContent = AppState.timeLimit + ' min';
    });
  }

  // Num taxis slider
  const taxiSlider = $('slider-num-taxis');
  const taxiVal = $('val-num-taxis');
  if (taxiSlider) {
    taxiSlider.addEventListener('input', e => {
      AppState.numTaxis = parseInt(e.target.value);
      taxiVal.textContent = AppState.numTaxis + (AppState.numTaxis > 1 ? ' (SRRS)' : ' (RRS)');
    });
  }

  // Scenario buttons
  document.querySelectorAll('.scenario-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      AppState.scenario = btn.dataset.scenario;
    });
  });

  // Recommend button
  const recBtn = $('btn-recommend');
  if (recBtn) {
    recBtn.addEventListener('click', requestRecommendation);
  }

  // Clear button
  const clearBtn = $('btn-clear');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      MapModule.clearRoutes();
      AppState.currentRecommendations = [];
      AppState.clickedLat = null;
      AppState.clickedLon = null;
      Dashboard.renderTaxiList([], null);
      // Reset metrics
      ['metric-earning', 'metric-segments', 'metric-time', 'metric-taxis'].forEach(id => {
        const el = $(id);
        if (el) { el.textContent = '—'; el.dataset.current = '0'; }
      });
      document.querySelector('.map-instruction')?.classList.remove('hidden');
    });
  }

  // Heatmap toggle
  const heatToggle = $('toggle-heatmap');
  if (heatToggle) {
    heatToggle.addEventListener('change', e => {
      AppState.showHeatmap = e.target.checked;
      MapModule.toggleHeatmap(e.target.checked);
    });
  }

  // Network toggle
  const netToggle = $('toggle-network');
  if (netToggle) {
    netToggle.addEventListener('change', e => {
      AppState.showNetwork = e.target.checked;
      MapModule.toggleNetwork(e.target.checked);
    });
  }
}

// ─── Tab Switching ────────────────────────────────────────────────────────────
function bindTabs() {
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      switchTab(target);
    });
  });
}

function switchTab(tabName) {
  AppState.activeTab = tabName;

  // Update tab buttons
  document.querySelectorAll('.nav-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === tabName);
  });

  // Show/hide panels
  document.querySelectorAll('.tab-panel').forEach(p => {
    p.style.display = p.dataset.panel === tabName ? 'flex' : 'none';
    p.style.flexDirection = 'column';
  });

  // Lazy-load stats tab
  if (tabName === 'stats') {
    Dashboard.loadModelStats();
  }
}

// ─── Status Updates ────────────────────────────────────────────────────────────
function setStatus(state, text) {
  const dot = document.querySelector('.status-dot');
  const label = document.querySelector('.status-label');
  if (dot) {
    dot.className = 'status-dot ' + state;
  }
  if (label) label.textContent = text;
}

// ─── Toast Notification ────────────────────────────────────────────────────────
function showToast(message) {
  let toast = document.getElementById('toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'toast';
    toast.style.cssText = `
      position:fixed; bottom:24px; left:50%; transform:translateX(-50%);
      background:rgba(255,77,109,0.12); border:1px solid rgba(255,77,109,0.4);
      color:#ff4d6d; padding:10px 20px; border-radius:999px;
      font-size:12px; font-weight:500; z-index:9999;
      backdrop-filter:blur(12px); transition:opacity 0.3s ease;
      font-family:'Inter',sans-serif;
    `;
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.style.opacity = '1';
  clearTimeout(toast._timeout);
  toast._timeout = setTimeout(() => { toast.style.opacity = '0'; }, 4000);
}

// ─── Utility ───────────────────────────────────────────────────────────────────
function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

// ─── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', initApp);
