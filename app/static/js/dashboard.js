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
  // ─── Update Metric Cards ───────────────────────────────────────────────────
  function updateMetrics(recommendations, algorithmName, moSummary) {
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

    // Num taxis or SDE fitness
    const taxiEl = document.getElementById('metric-taxis');
    if (taxiEl) {
      if (primaryRec.sde_fitness !== undefined && primaryRec.sde_fitness !== null) {
        animateCount(taxiEl, primaryRec.sde_fitness, '', ' fit', 2);
        const taxiSub = taxiEl.parentElement.querySelector('.metric-sub');
        if (taxiSub) taxiSub.textContent = 'Pareto SDE';
      } else {
        animateCount(taxiEl, recommendations.length, '', '', 0);
        const taxiSub = taxiEl.parentElement.querySelector('.metric-sub');
        if (taxiSub) taxiSub.textContent = 'recommended';
      }
    }

    // Algorithm badge
    const algoEl = document.getElementById('algo-badge');
    if (algoEl) {
      const name = algorithmName || primaryRec.algorithm || 'Route Optimizer';
      let icon = 'search';
      if (name.includes('MORA')) icon = 'brain';
      else if (name.includes('SRRS') || recommendations.length > 1) icon = 'zap';
      algoEl.innerHTML = `<i data-lucide="${icon}" style="margin-right:4px"></i> ${name}`;
      algoEl.title = name;
      if (window.lucide) lucide.createIcons();
    }
  }

  // ─── Render MORA Multi-Objective Metrics & Pareto Switcher ───────────────
  function renderMoraObjectives(moSummary, paretoAlts, onParetoCallback) {
    const card = document.getElementById('mora-objectives-card');
    const switcher = document.getElementById('pareto-switcher');
    if (!card) return;

    if (!moSummary) {
      card.style.display = 'none';
      if (switcher) switcher.style.display = 'none';
      return;
    }

    card.style.display = 'block';

    const f1El = document.getElementById('mora-f1-val');
    const f2El = document.getElementById('mora-f2-val');
    const f3El = document.getElementById('mora-f3-val');
    const sdeEl = document.getElementById('mora-sde-val');
    const tagEl = document.getElementById('mora-variant-tag');

    if (f1El) f1El.textContent = `¥${moSummary.expected_profit.toFixed(2)}`;
    if (f2El) f2El.textContent = moSummary.novelty.toFixed(3);
    if (f3El) f3El.textContent = moSummary.diversity.toFixed(3);
    if (sdeEl) sdeEl.textContent = moSummary.sde_fitness.toFixed(3);
    if (tagEl) {
      tagEl.textContent = moSummary.is_cold_start ? 'LEPS COLD START' : 'IEEE TEVC 2026';
      tagEl.style.color = moSummary.is_cold_start ? '#00d4ff' : 'var(--teal)';
    }

    // Render Pareto alternatives if available
    if (paretoAlts && Object.keys(paretoAlts).length > 0 && switcher) {
      switcher.style.display = 'block';
      document.querySelectorAll('.pareto-pill').forEach(btn => {
        btn.onclick = () => {
          document.querySelectorAll('.pareto-pill').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          const altKey = btn.dataset.alt;
          if (onParetoCallback) onParetoCallback(altKey);
        };
      });
    } else if (switcher) {
      switcher.style.display = 'none';
    }
  }

  // ─── Render LLM Explainability Rationale ─────────────────────────────────
  function renderLLMExplanation(llmExp) {
    const card = document.getElementById('llm-explanation-card');
    if (!card) return;

    if (!llmExp) {
      card.style.display = 'none';
      return;
    }

    card.style.display = 'block';

    const quoteEl = document.getElementById('xai-summary-text');
    if (quoteEl) {
      quoteEl.textContent = llmExp.summary_explanation || 'No summary explanation available.';
    }

    const stepsList = document.getElementById('xai-steps-list');
    const toggleBtn = document.getElementById('xai-steps-toggle');

    if (stepsList && llmExp.segment_reasons && llmExp.segment_reasons.length > 0) {
      stepsList.innerHTML = llmExp.segment_reasons.map(s => `
        <div class="xai-step-item">
          <div class="xai-step-title">
            <span>Step ${s.step}: Seg ${s.segment_id}</span>
            <span style="color:#00d4ff">${s.zone} · ${(s.pickup_prob * 100).toFixed(1)}% · ¥${s.expected_earning.toFixed(1)}</span>
          </div>
          <div class="xai-step-reason">${s.reason}</div>
        </div>
      `).join('');

      if (toggleBtn) {
        toggleBtn.onclick = () => {
          const isHidden = stepsList.style.display === 'none';
          stepsList.style.display = isHidden ? 'flex' : 'none';
          toggleBtn.querySelector('span').textContent = isHidden ? 'Hide Step-by-Step AI Rationale' : 'View Step-by-Step AI Rationale';
        };
      }
    } else if (stepsList) {
      stepsList.innerHTML = '';
    }

    if (window.lucide) lucide.createIcons();
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

      if (data.fallback) {
        const summary = data.historical_summary;
        container.innerHTML = `
          <div class="panel-section" style="padding-top:12px">
            <div class="paper-card" style="margin-bottom:14px">
              <div class="paper-title">Historical Pickup Data</div>
              <div class="paper-authors">${data.message}</div>
              <div class="paper-meta"><span class="paper-tag">network_properties.csv</span></div>
            </div>
            <div class="section-title">Available Evaluation Data</div>
            <div class="metrics-grid">
              <div class="metric-card">
                <div class="metric-label">Observed Segments</div>
                <div class="metric-value teal">${summary.segments_with_observations.toLocaleString()}</div>
                <div class="metric-sub">with pickup records</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">Mean Pickup Rate</div>
                <div class="metric-value amber">${(summary.mean_pickup_rate * 100).toFixed(1)}%</div>
                <div class="metric-sub">segment average</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">Overall Pickup Rate</div>
                <div class="metric-value green">${(summary.overall_pickup_rate * 100).toFixed(1)}%</div>
                <div class="metric-sub">weighted by observations</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">Pickup Events</div>
                <div class="metric-value purple">${summary.total_pickups.toLocaleString()}</div>
                <div class="metric-sub">historical records</div>
              </div>
            </div>
            <div style="margin-top:14px;font-size:12px;color:var(--text-secondary);line-height:1.7">
              GCN-LSTM MAE and hyperparameter charts require the optional experiment CSV. The route-level <b>Check Accuracy</b> button evaluates recommendations against these historical pickup rates.
            </div>
          </div>`;
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
    renderMoraObjectives,
    renderLLMExplanation,
    loadModelStats,
    loadNetworkSummary,
    animateCount,
  };
})();
