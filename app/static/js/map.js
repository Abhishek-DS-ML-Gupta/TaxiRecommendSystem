/**
 * map.js — Leaflet map initialization, road network drawing,
 *           heatmap layer, and route animation.
 */

const MapModule = (() => {
  let map = null;
  let networkLayer = null;
  let heatLayer = null;
  let routeLayers = [];
  let markerStart = null;
  let heatmapVisible = true;
  let networkVisible = true;

  // Color palette for multi-taxi routes
  const ROUTE_COLORS = [
    '#00d4ff', '#ff9f00', '#00ff9d', '#b97aff',
    '#ff4d6d', '#ffdf00', '#47c6ff', '#ff7eb3'
  ];

  function init(onClickCallback, cartoApiKey) {
    // Initialize map centered on Beijing 3rd Ring Road
    map = L.map('map', {
      center: [39.9042, 116.4074],
      zoom: 13,
      zoomControl: true,
      attributionControl: true,
    });

    // CartoDB Dark Matter tiles with API key from backend
    L.tileLayer(`https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png?key=${cartoApiKey || ''}`, {
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OSM</a> | © <a href="https://carto.com/attributions">CARTO</a>',
      maxZoom: 18,
      subdomains: 'abcd',
    }).addTo(map);

    // Click handler: set taxi starting position
    map.on('click', (e) => {
      onClickCallback(e.latlng.lat, e.latlng.lng);
    });

    return map;
  }

  function setStartMarker(lat, lon) {
    if (markerStart) map.removeLayer(markerStart);

    const taxiIcon = L.divIcon({
      html: `<div style="
        width:32px; height:32px;
        background: #3b82f6;
        border-radius: 50% 50% 50% 0;
        transform: rotate(-45deg);
        border: 2px solid white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        display:flex; align-items:center; justify-content:center;
      "><div style="transform:rotate(45deg); display:flex;">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9l-1.4 2.9A3.7 3.7 0 0 0 2 12v4c0 .6.4 1 1 1h2"/><circle cx="7" cy="17" r="2"/><circle cx="17" cy="17" r="2"/></svg>
      </div></div>`,
      className: '',
      iconSize: [32, 32],
      iconAnchor: [16, 32],
    });

    markerStart = L.marker([lat, lon], { icon: taxiIcon })
      .addTo(map)
      .bindPopup(`<b>Taxi Start Position</b><br>Lat: ${lat.toFixed(5)}<br>Lon: ${lon.toFixed(5)}`);
  }

  async function loadHeatmap() {
    try {
      const resp = await fetch('/api/hotspots?top_n=150');
      const data = await resp.json();

      if (heatLayer) map.removeLayer(heatLayer);

      const points = data.points.map(p => [p.lat, p.lon, p.intensity]);

      if (typeof L.heatLayer !== 'undefined') {
        heatLayer = L.heatLayer(points, {
          radius: 18,
          blur: 22,
          maxZoom: 16,
          max: 1.0,
          gradient: {
            0.0: '#0073e6',
            0.3: '#00d4ff',
            0.6: '#ff9f00',
            0.9: '#ff4d6d',
            1.0: '#ffffff'
          }
        }).addTo(map);
      } else {
        // Fallback: draw hotspots as circle markers if heatmap plugin unavailable
        points.slice(0, 80).forEach(p => {
          L.circleMarker([p.lat, p.lon], {
            radius: Math.max(4, p.intensity * 12),
            fillColor: '#ff9f00',
            fillOpacity: p.intensity * 0.7,
            color: 'transparent',
            weight: 0,
          }).addTo(map);
        });
        heatLayer = null;
      }

      return data.points.length;
    } catch (err) {
      console.error('Heatmap load failed:', err);
      return 0;
    }
  }

  async function loadRoadNetwork() {
    try {
      const resp = await fetch('/api/network?max_edges=2500');
      const data = await resp.json();

      if (!data.edges || data.edges.length === 0) return 0;

      if (networkLayer) map.removeLayer(networkLayer);

      const lines = data.edges.map(e => ({
        type: 'Feature',
        geometry: {
          type: 'LineString',
          coordinates: [[e.src_lon, e.src_lat], [e.tgt_lon, e.tgt_lat]]
        },
        properties: { edge_id: e.edge_id }
      }));

      networkLayer = L.geoJSON({
        type: 'FeatureCollection',
        features: lines
      }, {
        style: {
          color: 'rgba(0, 212, 255, 0.15)',
          weight: 1.0,
          opacity: 1,
        }
      }).addTo(map);

      return data.displayed_edges;
    } catch (err) {
      console.error('Network load failed:', err);
      return 0;
    }
  }

  function clearRoutes() {
    routeLayers.forEach(layer => {
      if (layer) map.removeLayer(layer);
    });
    routeLayers = [];
  }

  function drawRoutes(recommendations) {
    clearRoutes();

    recommendations.forEach((rec, idx) => {
      if (!rec.route || rec.route.length < 2) return;

      const color = ROUTE_COLORS[idx % ROUTE_COLORS.length];
      const latlngs = rec.route.map(pt => [pt.lat, pt.lon]);

      // Main route line (animated dashed)
      const routeLine = L.polyline(latlngs, {
        color: color,
        weight: 4,
        opacity: 0.9,
        dashArray: idx === 0 ? null : '8, 6',
        lineCap: 'round',
        lineJoin: 'round',
      }).addTo(map);

      routeLayers.push(routeLine);

      // Glow effect (thicker semi-transparent line underneath)
      const glowLine = L.polyline(latlngs, {
        color: color,
        weight: 10,
        opacity: 0.15,
      }).addTo(map);
      routeLayers.push(glowLine);

      // Direction arrows using Leaflet.PolylineDecorator if available
      // Waypoint dots along route
      rec.route.forEach((pt, ptIdx) => {
        if (ptIdx === 0 || ptIdx === rec.route.length - 1 || ptIdx % 3 === 0) {
          const dotColor = ptIdx === rec.route.length - 1 ? '#ff9f00' : color;
          const dotSize = ptIdx === 0 ? 10 : (ptIdx === rec.route.length - 1 ? 10 : 6);

          const dot = L.circleMarker([pt.lat, pt.lon], {
            radius: dotSize / 2,
            fillColor: dotColor,
            fillOpacity: ptIdx === rec.route.length - 1 ? 1.0 : 0.8,
            color: 'rgba(0,0,0,0.3)',
            weight: 1,
          }).addTo(map);

          dot.bindPopup(`
            <b>${rec.taxi_id}</b> — Segment ${pt.segment_id}<br>
            Pickup Prob: <b>${(pt.pickup_prob * 100).toFixed(1)}%</b><br>
            Expected Earning: <b>¥${pt.expected_earning.toFixed(2)}</b>
          `);

          routeLayers.push(dot);
        }
      });

      // Destination marker
      const endPt = rec.route[rec.route.length - 1];
      const endIcon = L.divIcon({
        html: `<div style="
          background:${color};
          width:12px; height:12px;
          border-radius:50%;
          border: 2px solid white;
          box-shadow: 0 0 10px ${color};
        "></div>`,
        className: '',
        iconSize: [12, 12],
        iconAnchor: [6, 6],
      });
      const endMarker = L.marker([endPt.lat, endPt.lon], { icon: endIcon })
        .addTo(map)
        .bindPopup(`<b>${rec.taxi_id}</b><br>Route end<br>Expected: ¥${rec.total_expected_earning.toFixed(2)}`);
      routeLayers.push(endMarker);
    });

    // Fit map to first route
    if (recommendations.length > 0 && recommendations[0].route.length > 0) {
      const allLatLngs = recommendations.flatMap(r => r.route.map(p => [p.lat, p.lon]));
      map.fitBounds(L.latLngBounds(allLatLngs), { padding: [40, 40] });
    }
  }

  function toggleHeatmap(visible) {
    heatmapVisible = visible;
    if (heatLayer) {
      if (visible) map.addLayer(heatLayer);
      else map.removeLayer(heatLayer);
    }
  }

  function toggleNetwork(visible) {
    networkVisible = visible;
    if (networkLayer) {
      if (visible) map.addLayer(networkLayer);
      else map.removeLayer(networkLayer);
    }
  }

  return {
    init,
    setStartMarker,
    loadHeatmap,
    loadRoadNetwork,
    drawRoutes,
    clearRoutes,
    toggleHeatmap,
    toggleNetwork,
  };
})();
