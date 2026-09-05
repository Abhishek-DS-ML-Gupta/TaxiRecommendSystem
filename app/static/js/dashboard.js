/**
 * dashboard.js — Stats panel animations, metric card updates,
 *                taxi list rendering, and model stats display.
 */

const Dashboard = (() => {

  // ─── Animated Number Counter ──────────────────────────────────────────────
  function animateCount(el, targetValue, prefix = '', suffix = '', decimals = 0, duration = 700) {
    const start = parseFloat(el.dataset.current || '0');
    const startTime = performance.now();
    const range = targetValue - start;

    function step(now) {
      const elapsed = Math.min((now - startTime) / duration, 1);
      const eased = 1 - Math.pow(1 - elapsed, 3); // ease-out cubic
      const current = start + range * eased;
      el.textContent = prefix + current.toFixed(decimals) + suffix;
      if (elapsed < 1) requestAnimationFrame(step);
      else {
        el.textContent = prefix + targetValue.toFixed(decimals) + suffix;
        el.dataset.current = targetValue;
      }
    }
    requestAnimationFrame(step);
  }

  // ─── Update Metric Cards ───────────────────────────────────────────────────
  function updateMetrics(recommendations) {
    if (!recommendations || recommendations.length === 0) return;

    const primaryRec = recommendations[0];

    // Expected earning
    const earnEl = document.getElementById('metric-earning');
    if (earnEl) animateCount(earnEl, primaryRec.total_expected_earning, '¥', '', 2);

    // Route segments
    const segEl = document.getElementById('metric-segments');
    if (segEl) animateCount(segEl, primaryRec.route_length_segments, '', ' segs', 0);

    // Estimated time
    const timeEl = document.getElementById('metric-time');
    if (timeEl) animateCount(timeEl, primaryRec.estimated_time_minutes, '', ' min', 1);

    // Num taxis
    const taxiEl = document.getElementById('metric-taxis');
    if (taxiEl) animateCount(taxiEl, recommendations.length, '', '', 0);

    // Algorithm badge
    const algoEl = document.getElementById('algo-badge');
    if (algoEl && recommendations.length > 0) {
      // Show RRS vs SRRS
      const algoText = recommendations.length === 1 ? 'RRS Algorithm' : 'SRRS Algorithm';
      algoEl.textContent = algoText;
      algoEl.title = primaryRec.algorithm || '';
    }
  }

  // ─── Render Taxi List ──────────────────────────────────────────────────────
  const ROUTE_COLORS = [
    '#00d4ff', '#ff9f00', '#00ff9d', '#b97aff',
    '#ff4d6d', '#ffdf00', '#47c6ff', '#ff7eb3'
  ];

  function renderTaxiList(recommendations, onSelectCallback) {
    const container = document.getElementById('taxi-list');
    if (!container) return;

    if (!recommendations || recommendations.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon"><i data-lucide="car" style="width:32px;height:32px"></i></div>
          <div class="empty-text">Click anywhere on the<br>Beijing map to get<br>route recommendations.</div>
        </div>`;
      if (window.lucide) lucide.createIcons();
      return;
    }

    container.innerHTML = '';
    recommendations.forEach((rec, idx) => {
      const color = ROUTE_COLORS[idx % ROUTE_COLORS.length];
      const card = document.createElement('div');
      card.className = 'taxi-card animate-in' + (idx === 0 ? ' active' : '');
      card.style.animationDelay = `${idx * 60}ms`;
      card.style.setProperty('--route-color', color);
      card.innerHTML = `
        <div class="taxi-card-header">
          <span class="taxi-id">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
              background:${color};box-shadow:0 0 6px ${color};margin-right:6px;"></span>
            ${rec.taxi_id}
          </span>
          <span class="taxi-earning">¥${rec.total_expected_earning.toFixed(2)}</span>
        </div>
        <div class="taxi-card-meta">
          <div class="taxi-meta-item">Segments: <span>${rec.route_length_segments}</span></div>
          <div class="taxi-meta-item">Time: <span>~${rec.estimated_time_minutes.toFixed(1)} min</span></div>
          <div class="taxi-meta-item">
            Avg Prob: <span>${rec.route.length > 0
              ? (rec.route.reduce((a,b) => a + b.pickup_prob, 0) / rec.route.length * 100).toFixed(1) + '%'
              : 'N/A'
            }</span>
          </div>
        </div>`;

      card.addEventListener('click', () => {
        document.querySelectorAll('.taxi-card').forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        onSelectCallback(idx);
      });

      container.appendChild(card);
    });
  }

  // ─── Load and Render Model Stats ───────────────────────────────────────────
  async function loadModelStats() {
    const container = document.getElementById('stats-content');
    if (!container) return;

    container.innerHTML = `
      <div style="text-align:center;padding:32px 16px;color:var(--text-muted)">
        <div class="spinner" style="width:28px;height:28px;margin:0 auto 10px"></div>
        <div style="font-size:13px">Loading experiment results...</div>
      </div>`;

    try {
      const resp = await fetch('/api/stats');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
      const data = await resp.json();

      if (!data.available) {
        container.innerHTML = `
          <div class="panel-section">
            <div class="empty-state">
              <div class="empty-icon"><i data-lucide="triangle-alert" style="width:32px;height:32px"></i></div>
              <div class="empty-text" style="color:var(--amber)">${data.message || 'Results unavailable'}</div>
            </div>
          </div>`;
        if (window.lucide) lucide.createIcons();
        return;
      }

      const best = data.best_overall;
      const dist = data.mae_distribution;
      const maeMin = dist.min;
      const maeMax = dist.max;
      const maeRange = maeMax - maeMin || 1;

      // Build histogram bars
      const histHTML = (dist.histogram || []).map(b => {
        const maxCount = Math.max(...(dist.histogram || []).map(x => x.count), 1);
        const barH = Math.max(4, Math.round((b.count / maxCount) * 48));
        return `<div title="MAE ≈ ${b.bucket.toFixed(4)}: ${b.count} runs"
                     style="flex:1;display:flex;flex-direction:column;align-items:center;gap:2px;cursor:default">
                  <span style="font-size:8px;color:var(--text-muted)">${b.count}</span>
                  <div style="width:100%;height:${barH}px;
                    background:linear-gradient(180deg,var(--teal),#0073e6);
                    border-radius:2px 2px 0 0;opacity:0.8"></div>
                  <span style="font-size:7px;color:var(--text-muted);transform:rotate(-45deg);
                    transform-origin:top left;margin-top:4px;white-space:nowrap">
                    ${b.bucket.toFixed(3)}
                  </span>
                </div>`;
      }).join('');

      container.innerHTML = `
        <div class="panel-section" style="padding-top:12px">

          <!-- Paper info -->
          <div class="paper-card" style="margin-bottom:14px">
            <div class="paper-title">${data.paper.title}</div>
            <div class="paper-authors">${data.paper.authors} · ${data.paper.year}</div>
            <div class="paper-meta">
              <span class="paper-tag">${data.paper.journal}</span>
              <span class="paper-tag">DOI: ${data.paper.doi.split('/').pop()}</span>
            </div>
          </div>

          <!-- Best config cards -->
          <div class="section-title">Best Model Configuration</div>
          <div class="metrics-grid" style="margin-bottom:14px">
            <div class="metric-card">
              <div class="metric-label">Best MAE</div>
              <div class="metric-value teal" style="font-size:18px">${best.mae.toFixed(5)}</div>
              <div class="metric-sub">Mean Absolute Error</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Experiments</div>
              <div class="metric-value amber">${data.total_experiments}</div>
              <div class="metric-sub">Hyperparameter runs</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Prior Days</div>
              <div class="metric-value purple">${best.prior_days}</div>
              <div class="metric-sub">days of history</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Prior Hours</div>
              <div class="metric-value green">${best.prior_hours}</div>
              <div class="metric-sub">hours of history</div>
            </div>
          </div>

          <!-- MAE distribution -->
          <div class="section-title">MAE Distribution (${data.total_experiments} runs)</div>
          <div style="display:flex;gap:2px;align-items:flex-end;height:80px;
                      padding:0 4px 24px;margin-bottom:8px;overflow:hidden">
            ${histHTML || '<div style="color:var(--text-muted);font-size:11px">No histogram data</div>'}
          </div>
          <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-muted);margin-bottom:14px;padding:0 4px">
            <span>Min: <b style="color:var(--green)">${dist.min.toFixed(5)}</b></span>
            <span>Mean: <b style="color:var(--teal)">${dist.mean.toFixed(5)}</b></span>
            <span>Max: <b style="color:var(--amber)">${dist.max.toFixed(5)}</b></span>
          </div>

          <!-- Per-date table -->
          <div class="section-title">Best Run per Date</div>
          <table class="stats-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>GCN</th>
                <th>LSTM</th>
                <th>MAE</th>
              </tr>
            </thead>
            <tbody>
              ${(data.experiments_by_date || []).map(r => {
                const barW = Math.max(4, Math.round((1 - (r.mae - maeMin) / maeRange) * 56));
                return `<tr>
                  <td style="font-family:inherit;color:var(--text-secondary)">${r.date}</td>
                  <td>${r.gc_dims}</td>
                  <td>${r.lstm_dims}</td>
                  <td>
                    <div class="mae-bar-container">
                      <div class="mae-bar" style="width:${barW}px"></div>
                      <span>${r.mae.toFixed(4)}</span>
                    </div>
                  </td>
                </tr>`;
              }).join('')}
            </tbody>
          </table>
        </div>`;

    } catch (err) {
      container.innerHTML = `
        <div class="panel-section">
          <div class="empty-state">
            <div class="empty-icon"><i data-lucide="x-circle" style="width:32px;height:32px"></i></div>
            <div class="empty-text" style="color:var(--red)">
              Failed to load model stats:<br><code style="font-size:10px">${err.message}</code>
            </div>
          </div>
        </div>`;
      if (window.lucide) lucide.createIcons();
    }
  }


  // ─── Load and Render Network Summary ──────────────────────────────────────
  async function loadNetworkSummary() {
    try {
      const resp = await fetch('/api/network/summary');
      const data = await resp.json();
      if (!data.loaded) return;

      const intersEl = document.getElementById('net-intersections');
      const segmentsEl = document.getElementById('net-segments');
      const meanProbEl = document.getElementById('net-mean-prob');
      const meanEarnEl = document.getElementById('net-mean-earn');

      if (intersEl) animateCount(intersEl, data.total_intersections, '', '', 0);
      if (segmentsEl) animateCount(segmentsEl, data.total_road_segments, '', '', 0);
      if (meanProbEl) animateCount(meanProbEl, data.pickup_prob_stats.mean * 100, '', '%', 1);
      if (meanEarnEl) animateCount(meanEarnEl, data.earning_stats.mean, '¥', '', 1);
    } catch (err) {
      console.warn('Network summary failed:', err);
    }
  }

  return {
    updateMetrics,
    renderTaxiList,
    loadModelStats,
    loadNetworkSummary,
    animateCount,
  };
})();
