/**
 * app.js — Main application controller.
 * Orchestrates map, dashboard, API calls, and UI state.
 */

// ─── App State ────────────────────────────────────────────────────────────────
// ─── App State ────────────────────────────────────────────────────────────────
const AppState = {
  clickedLat: null,
  clickedLon: null,
  currentRecommendations: [],
  currentParetoAlternatives: null,
  currentMultiObjectiveSummary: null,
  currentLlmExplanation: null,
  isLoading: false,
  selectedTaxiIdx: 0,
  dataReady: false,

  // Controls
  algorithm: 'mora_llm',
  driverProfileId: 'new_driver_cold_start',
  alpha: 0.20,
  timeLimit: 10,
  numTaxis: 1,
  scenario: 'normal',
  showHeatmap: true,
  showNetwork: true,

  activeTab: 'recommend',
  theme: localStorage.getItem('taxi_theme') || 'dark',
};

const DRIVER_PERSONAS = {
  new_driver_cold_start: {
    tag: 'COLD START · 0 PRIOR TRIPS',
    tagClass: 'cold-start',
    desc: 'Newly onboarded taxi driver with minimal trip history. Unfamiliar with peak passenger clusters. Needs safe, reliable, high-probability pickup corridors along arterial roads rather than high-risk speculative cruising.',
    zones: 'Commercial, Residential'
  },
  experienced_veteran: {
    tag: 'VETERAN · 12 YEARS EXP',
    tagClass: 'veteran',
    desc: 'Highly experienced driver with deep knowledge of Beijing dispatch dynamics. Specializes in rapid CBD drop-offs and high-fare airport corridors.',
    zones: 'CBD, Commercial, Transport Hub'
  },
  night_shift_cruiser: {
    tag: 'NIGHT CRUISE SPECIALIST',
    tagClass: 'veteran',
    desc: 'Specializes in late-night and early-morning shifts. Targets late-night entertainment districts, karaoke strips, and midnight rail interchanges.',
    zones: 'Entertainment, Transport Hub'
  },
  airport_express: {
    tag: 'LONG HAUL TRANSIT',
    tagClass: 'veteran',
    desc: 'Prefers high-speed, long-distance inter-district trips connecting Beijing Capital Airport and high-speed rail hubs.',
    zones: 'Transport Hub, Expressway'
  },
  low_carbon_eco: {
    tag: 'ECO CRUISER · FUEL SAVER',
    tagClass: 'veteran',
    desc: 'Focuses on fuel efficiency and smooth driving profiles. Avoids congested choke points, preferring free-flowing ring-road arterials.',
    zones: 'Arterial, Residential'
  }
};

// ─── DOM References ────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ─── Initialize App ────────────────────────────────────────────────────────────
async function initApp() {
  try {
    // Initialize Duo Mode Theme
    initTheme();

    // Fetch configuration from backend
    await fetchConfig();

    // Init Mapbox GL JS map (with initial theme)
    MapModule.init(onMapClick, AppState.mapboxAccessToken);
    if (AppState.theme === 'light') {
      MapModule.setTheme('light');
    }

    // Mapbox bar controls (search & style toggles)
    bindMapboxControls();

    // UI control bindings
    bindControls();

    // Tab switching
    bindTabs();
  } catch (err) {
    console.error("Error during initApp:", err);
  }

  // Check server health and load map data
  await waitForServer();
}

// ─── Duo Mode Theme Controller ────────────────────────────────────────────────
function initTheme() {
  const savedTheme = localStorage.getItem('taxi_theme') || 'dark';
  applyTheme(savedTheme, false);

  const btnDark = $('theme-btn-dark');
  const btnLight = $('theme-btn-light');

  if (btnDark) {
    btnDark.addEventListener('click', () => applyTheme('dark', true));
  }
  if (btnLight) {
    btnLight.addEventListener('click', () => applyTheme('light', true));
  }
}

function applyTheme(theme, updateMap = true) {
  AppState.theme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('taxi_theme', theme);

  const btnDark = $('theme-btn-dark');
  const btnLight = $('theme-btn-light');
  if (btnDark) {
    btnDark.classList.toggle('active', theme === 'dark');
    btnDark.setAttribute('aria-checked', theme === 'dark');
  }
  if (btnLight) {
    btnLight.classList.toggle('active', theme === 'light');
    btnLight.setAttribute('aria-checked', theme === 'light');
  }

  if (updateMap && typeof MapModule !== 'undefined' && typeof MapModule.setTheme === 'function') {
    MapModule.setTheme(theme);
  }
}

async function fetchConfig() {
  try {
    const resp = await fetch('/api/config');
    const data = await resp.json();
    AppState.mapboxAccessToken = data.mapbox_access_token || '';
  } catch (err) {
    console.error("Failed to load frontend config:", err);
    AppState.mapboxAccessToken = '';
  }
}

function bindMapboxControls() {
  const inputGeocode = $('input-geocode');
  const btnGeocode = $('btn-geocode');
  if (btnGeocode && inputGeocode) {
    const doSearch = async () => {
      const q = inputGeocode.value.trim();
      if (!q) return;
      setStatus('loading', `Searching "${q}"...`);
      const res = await MapModule.searchGeocode(q);
      if (res) {
        setStatus('ready', `Found: ${res.text || res.place_name}`);
      } else {
        setStatus('error', `Location not found`);
      }
    };
    btnGeocode.addEventListener('click', doSearch);
    inputGeocode.addEventListener('keydown', e => {
      if (e.key === 'Enter') doSearch();
    });
  }

  const btnDark = $('btn-style-dark');
  const btnLight = $('btn-style-light');
  const btnStreets = $('btn-style-streets');
  if (btnDark) {
    btnDark.addEventListener('click', () => {
      MapModule.setStyle('dark');
    });
  }
  if (btnLight) {
    btnLight.addEventListener('click', () => {
      MapModule.setStyle('light');
    });
  }
  if (btnStreets) {
    btnStreets.addEventListener('click', () => {
      MapModule.setStyle('streets');
    });
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
  try {
    // Load map layers in parallel
    const [edgeCount, hotspotCount] = await Promise.all([
      MapModule.loadRoadNetwork(),
      MapModule.loadHeatmap(),
    ]);

    // Load network summary for overview tab
    Dashboard.loadNetworkSummary();

    console.log(`✅ Loaded ${edgeCount} road edges, ${hotspotCount} hotspot points`);
  } catch (err) {
    console.warn("Layer loading warning:", err);
  } finally {
    // Always hide loading overlay
    $('map-loading').classList.add('hidden');
    const inst = document.querySelector('.map-instruction');
    if (inst) inst.classList.remove('hidden');
    if (MapModule.resize) MapModule.resize();
  }
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

  const algoLabel = AppState.algorithm.startsWith('mora') ? 'MORA-LLM Multi-Objective Engine' : 'Route Optimizer';
  setStatus('loading', `Running ${algoLabel}...`);

  try {
    const payload = {
      lat: AppState.clickedLat,
      lon: AppState.clickedLon,
      time_limit: AppState.timeLimit,
      num_taxis: AppState.numTaxis,
      scenario: AppState.scenario,
      algorithm: AppState.algorithm,
      driver_profile_id: AppState.driverProfileId,
      alpha: AppState.alpha,
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
    AppState.currentParetoAlternatives = data.pareto_alternatives;
    AppState.currentMultiObjectiveSummary = data.multi_objective_summary;
    AppState.currentLlmExplanation = data.llm_explanation;

    const accuracyBtn = $('btn-accuracy');
    if (accuracyBtn) accuracyBtn.disabled = data.recommendations.length === 0;

    // Update map with primary route
    MapModule.drawRoutes(data.recommendations);

    // Update dashboard metrics and MORA-LLM cards
    Dashboard.updateMetrics(data.recommendations, data.algorithm, data.multi_objective_summary);
    Dashboard.renderMoraObjectives(data.multi_objective_summary, data.pareto_alternatives, onParetoSelected);
    Dashboard.renderLLMExplanation(data.llm_explanation);
    Dashboard.renderTaxiList(data.recommendations, onTaxiSelected);

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

// ─── Pareto Front Alternative Handler ─────────────────────────────────────────
function onParetoSelected(altKey) {
  if (!AppState.currentParetoAlternatives || !AppState.currentParetoAlternatives[altKey]) return;
  const alt = AppState.currentParetoAlternatives[altKey];

  const altRec = {
    taxi_id: alt.label,
    route: alt.route_points,
    total_expected_earning: alt.profit,
    route_length_segments: alt.route_points.length,
    estimated_time_minutes: alt.route_points.length * 0.8,
    novelty_score: alt.novelty,
    topic_diversity_score: alt.diversity,
  };

  MapModule.drawRoutes([altRec]);

  // Animate metrics for this alternative
  const earnEl = $('metric-earning');
  if (earnEl) Dashboard.animateCount(earnEl, alt.profit, '¥', '', 2);
  const segEl = $('metric-segments');
  if (segEl) Dashboard.animateCount(segEl, alt.route_points.length, '', ' segs', 0);

  // Update algo badge
  const algoEl = $('algo-badge');
  if (algoEl) {
    algoEl.innerHTML = `<i data-lucide="compass" style="margin-right:4px"></i> Pareto: ${alt.label}`;
    if (window.lucide) lucide.createIcons();
  }
}

// ─── Taxi Selection from List ─────────────────────────────────────────────────
function onTaxiSelected(idx) {
  AppState.selectedTaxiIdx = idx;
  // Focus on selected taxi's route
  if (AppState.currentRecommendations[idx]) {
    const singleRec = AppState.currentRecommendations[idx];
    const earnEl = $('metric-earning');
    if (earnEl) Dashboard.animateCount(earnEl, singleRec.total_expected_earning, '¥', '', 2);
    const segEl = $('metric-segments');
    if (segEl) Dashboard.animateCount(segEl, singleRec.route_length_segments, '', ' segs', 0);
  }
}

// ─── Control Bindings ─────────────────────────────────────────────────────────
function bindControls() {
  // Algorithm buttons
  document.querySelectorAll('.algo-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.algo-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      AppState.algorithm = btn.dataset.algo;

      const valAlgo = $('val-algorithm');
      const personaGrp = $('group-driver-persona');
      const alphaGrp = $('group-alpha');
      const moraCard = $('mora-objectives-card');
      const xaiCard = $('llm-explanation-card');

      if (AppState.algorithm === 'mora_llm') {
        if (valAlgo) valAlgo.textContent = 'MORA-LLM (2026)';
        if (personaGrp) personaGrp.style.display = 'block';
        if (alphaGrp) alphaGrp.style.display = 'block';
      } else if (AppState.algorithm === 'srrs') {
        if (valAlgo) valAlgo.textContent = 'SRRS (Multi-Taxi)';
        if (personaGrp) personaGrp.style.display = 'none';
        if (alphaGrp) alphaGrp.style.display = 'none';
        if (moraCard) moraCard.style.display = 'none';
        if (xaiCard) xaiCard.style.display = 'none';
        if (AppState.numTaxis === 1) {
          AppState.numTaxis = 3;
          $('slider-num-taxis').value = '3';
          $('val-num-taxis').textContent = '3 (SRRS)';
        }
      } else if (AppState.algorithm === 'rrs') {
        if (valAlgo) valAlgo.textContent = 'RRS (Single Taxi)';
        if (personaGrp) personaGrp.style.display = 'none';
        if (alphaGrp) alphaGrp.style.display = 'none';
        if (moraCard) moraCard.style.display = 'none';
        if (xaiCard) xaiCard.style.display = 'none';
        AppState.numTaxis = 1;
        $('slider-num-taxis').value = '1';
        $('val-num-taxis').textContent = '1 (RRS)';
      }
    });
  });

  // Driver Persona selector
  const personaSelect = $('select-driver-persona');
  if (personaSelect) {
    personaSelect.addEventListener('change', e => {
      AppState.driverProfileId = e.target.value;
      const meta = DRIVER_PERSONAS[AppState.driverProfileId];
      const card = $('driver-persona-card');
      if (card && meta) {
        card.innerHTML = `
          <div class="driver-tag ${meta.tagClass}"><i data-lucide="snowflake" style="width:12px;height:12px"></i> ${meta.tag}</div>
          <div class="driver-desc">${meta.desc}</div>
          <div class="driver-zones"><b>Affinities:</b> ${meta.zones}</div>
        `;
        if (window.lucide) lucide.createIcons();
      }
    });
  }

  // Alpha slider
  const alphaSlider = $('slider-alpha');
  const alphaVal = $('val-alpha');
  if (alphaSlider) {
    alphaSlider.addEventListener('input', e => {
      AppState.alpha = parseFloat(e.target.value);
      if (alphaVal) alphaVal.textContent = AppState.alpha.toFixed(2);
    });
  }

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
      const tag = AppState.algorithm.startsWith('mora') ? ' (MORA)' : (AppState.numTaxis > 1 ? ' (SRRS)' : ' (RRS)');
      taxiVal.textContent = AppState.numTaxis + tag;
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
      AppState.currentParetoAlternatives = null;
      AppState.currentMultiObjectiveSummary = null;
      AppState.currentLlmExplanation = null;
      AppState.clickedLat = null;
      AppState.clickedLon = null;
      Dashboard.renderTaxiList([], null);
      Dashboard.renderMoraObjectives(null);
      Dashboard.renderLLMExplanation(null);
      // Reset metrics
      ['metric-earning', 'metric-segments', 'metric-time', 'metric-taxis'].forEach(id => {
        const el = $(id);
        if (el) { el.textContent = '—'; el.dataset.current = '0'; }
      });
      const accuracyBtn = $('btn-accuracy');
      if (accuracyBtn) accuracyBtn.disabled = true;
      const accuracyPanel = $('accuracy-panel');
      if (accuracyPanel) accuracyPanel.style.display = 'none';
      document.querySelector('.map-instruction')?.classList.remove('hidden');
    });
  }

  const accuracyBtn = $('btn-accuracy');
  if (accuracyBtn) {
    accuracyBtn.addEventListener('click', checkRecommendationAccuracy);
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

async function checkRecommendationAccuracy() {
  if (AppState.currentRecommendations.length === 0) return;

  const panel = $('accuracy-panel');
  const result = $('accuracy-result');
  if (panel) panel.style.display = 'block';
  if (result) result.textContent = 'Checking historical pickup records...';

  try {
    const resp = await fetch('/api/recommend/accuracy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recommendations: AppState.currentRecommendations }),
    });
    const data = await resp.json();
    if (!resp.ok || !data.available) throw new Error(data.message || 'Accuracy data unavailable');
    result.innerHTML = `
      <div>${data.message}</div>
      <div>Segments evaluated: <b>${data.segments_evaluated}</b></div>
      <div>Mean absolute error: <b>${data.mean_absolute_error.toFixed(4)}</b></div>
      <div>Brier score: <b>${data.brier_score.toFixed(4)}</b></div>
      <div>Predicted vs historical pickup rate: <b>${(data.predicted_pickup_rate * 100).toFixed(1)}%</b> vs <b>${(data.historical_pickup_rate * 100).toFixed(1)}%</b></div>
      <div>Estimated route success: <b>${(data.estimated_route_success_rate * 100).toFixed(1)}%</b></div>`;
  } catch (err) {
    if (result) result.textContent = err.message || 'Accuracy check failed.';
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
