/**
 * map.js — Mapbox GL JS map implementation.
 * Replaces CARTO with Mapbox GL JS for interactive vector tiles,
 * road network rendering, native WebGL heatmaps, route animations,
 * and Geocoding search.
 */

const MapModule = (() => {
  let map = null;
  let mapboxToken = '';
  let onClickCallback = null;
  let markerStart = null;
  let routeMarkers = [];
  let heatmapVisible = true;
  let networkVisible = true;
  let currentStyle = 'dark';
  let currentTheme = 'dark';
  let isMapLoaded = false;
  let pendingNetworkData = null;
  let pendingHeatmapData = null;
  let cachedNetworkEdges = null;
  let cachedHeatmapPoints = null;
  let cachedRecommendations = null;
  let lastStartPos = null;

  const ROUTE_COLORS = [
    '#00d4ff', '#ff9f00', '#00ff9d', '#b97aff',
    '#ff4d6d', '#ffdf00', '#47c6ff', '#ff7eb3'
  ];

  // Fallback dark style (Carto Dark Matter) if no active Mapbox token is configured
  const FALLBACK_DARK_STYLE = {
    version: 8,
    sources: {
      'carto-tiles': {
        type: 'raster',
        tiles: [
          'https://a.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png',
          'https://b.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png'
        ],
        tileSize: 256,
        attribution: '© <a href="https://www.openstreetmap.org/copyright">OSM</a> | © Mapbox GL JS'
      }
    },
    layers: [
      {
        id: 'carto-tiles-layer',
        type: 'raster',
        source: 'carto-tiles',
        minzoom: 0,
        maxzoom: 19
      }
    ]
  };

  // Fallback light style (Carto Positron) for Milky Light Theme
  const FALLBACK_LIGHT_STYLE = {
    version: 8,
    sources: {
      'carto-light-tiles': {
        type: 'raster',
        tiles: [
          'https://a.basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}.png',
          'https://b.basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}.png'
        ],
        tileSize: 256,
        attribution: '© <a href="https://www.openstreetmap.org/copyright">OSM</a> | © Mapbox GL JS'
      }
    },
    layers: [
      {
        id: 'carto-light-layer',
        type: 'raster',
        source: 'carto-light-tiles',
        minzoom: 0,
        maxzoom: 19
      }
    ]
  };

  function hasValidToken(token) {
    return token && token !== 'YOUR_ACCESS_TOKEN' && typeof token === 'string' && token.startsWith('pk.');
  }

  function getMapStyle(styleName, token) {
    if (hasValidToken(token)) {
      if (styleName === 'streets') return 'mapbox://styles/mapbox/streets-v11';
      if (styleName === 'light') return 'mapbox://styles/mapbox/light-v10';
      return 'mapbox://styles/mapbox/dark-v11';
    }
    return styleName === 'light' ? FALLBACK_LIGHT_STYLE : FALLBACK_DARK_STYLE;
  }

  function resize() {
    if (map && typeof map.resize === 'function') {
      try { map.resize(); } catch (e) {}
    }
  }

  function init(clickCallback, token) {
    onClickCallback = clickCallback;
    mapboxToken = token || '';

    if (typeof mapboxgl === 'undefined') {
      console.error('Mapbox GL JS library failed to load.');
      return null;
    }

    // Set accessToken globally before map creation if provided
    if (hasValidToken(mapboxToken)) {
      mapboxgl.accessToken = mapboxToken;
    }

    const initialStyle = getMapStyle(currentStyle, mapboxToken);

    // Initialize Mapbox map centered on Beijing 3rd Ring Road
    try {
      map = new mapboxgl.Map({
        container: 'map',
        style: initialStyle,
        center: [116.4074, 39.9042], // [lon, lat] for Mapbox
        zoom: 12.5,
        attributionControl: true
      });
    } catch (err) {
      console.warn('Primary style creation failed, falling back:', err);
      try {
        map = new mapboxgl.Map({
          container: 'map',
          style: FALLBACK_DARK_STYLE,
          center: [116.4074, 39.9042],
          zoom: 12.5,
          attributionControl: true
        });
      } catch (fatal) {
        console.error('Mapbox fatal init error:', fatal);
        return null;
      }
    }

    // Add navigation controls (zoom, compass)
    try {
      map.addControl(new mapboxgl.NavigationControl({ showCompass: true }), 'top-right');
    } catch (e) {}

    // Ensure map layout sizes properly on initialization
    setTimeout(resize, 150);
    setTimeout(resize, 600);

    map.on('load', () => {
      isMapLoaded = true;
      resize();
      console.log('Mapbox GL JS map successfully loaded.');

      // Update status text in UI
      const statusText = document.getElementById('mapbox-status-text');
      if (statusText) {
        statusText.textContent = hasValidToken(mapboxToken) ? 'Mapbox GL JS (Vector)' : 'Mapbox GL JS (Ready)';
      }

      // If network or heatmap data arrived before style loaded, add them now
      if (pendingNetworkData) applyRoadNetworkData(pendingNetworkData);
      if (pendingHeatmapData) applyHeatmapData(pendingHeatmapData);
    });

    // Fallback on unauthorized style or token error
    map.on('error', (e) => {
      console.warn('Mapbox GL event:', e);
      if (e && e.error && (e.error.status === 401 || e.error.status === 403) && currentStyle !== 'fallback') {
        console.warn('Mapbox token unauthorized or expired. Falling back to dark tile style.');
        currentStyle = 'fallback';
        map.setStyle(FALLBACK_DARK_STYLE);
        const statusText = document.getElementById('mapbox-status-text');
        if (statusText) statusText.textContent = 'Mapbox GL JS (Fallback)';
      }
    });

    // Click handler to select taxi start location
    map.on('click', (e) => {
      if (onClickCallback) {
        onClickCallback(e.lngLat.lat, e.lngLat.lng);
      }
    });

    return map;
  }

  function setStartMarker(lat, lon) {
    if (!map) return;
    lastStartPos = { lat, lon };

    if (markerStart) {
      markerStart.remove();
      markerStart = null;
    }

    const el = document.createElement('div');
    el.className = 'taxi-marker-pulse';
    el.innerHTML = `
      <div style="
        width:32px; height:32px;
        background: #3b82f6;
        border-radius: 50% 50% 50% 0;
        transform: rotate(-45deg);
        border: 2px solid white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        display:flex; align-items:center; justify-content:center;
      "><div style="transform:rotate(45deg); display:flex;">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9l-1.4 2.9A3.7 3.7 0 0 0 2 12v4c0 .6.4 1 1 1h2"/><circle cx="7" cy="17" r="2"/><circle cx="17" cy="17" r="2"/></svg>
      </div></div>`;

    const popup = new mapboxgl.Popup({ offset: 25 })
      .setHTML(`<b>Taxi Start Position</b><br>Lat: ${lat.toFixed(5)}<br>Lon: ${lon.toFixed(5)}`);

    markerStart = new mapboxgl.Marker({ element: el, anchor: 'bottom' })
      .setLngLat([lon, lat])
      .setPopup(popup)
      .addTo(map);
  }

  function applyRoadNetworkData(edges) {
    if (!edges || edges.length === 0) return;
    cachedNetworkEdges = edges;

    if (!map) {
      pendingNetworkData = edges;
      return;
    }

    if (!map.isStyleLoaded()) {
      pendingNetworkData = edges;
      map.once('style.load', () => {
        if (pendingNetworkData) applyRoadNetworkData(pendingNetworkData);
      });
      return;
    }

    const features = edges.map(e => ({
      type: 'Feature',
      geometry: {
        type: 'LineString',
        coordinates: [[e.src_lon, e.src_lat], [e.tgt_lon, e.tgt_lat]]
      },
      properties: { edge_id: e.edge_id }
    }));

    const geojson = { type: 'FeatureCollection', features };
    const lineColor = currentTheme === 'light' ? '#2563eb' : '#00d4ff';
    const lineOpacity = currentTheme === 'light' ? 0.32 : 0.22;

    if (map.getSource('network-source')) {
      map.getSource('network-source').setData(geojson);
      if (map.getLayer('network-layer')) {
        map.setPaintProperty('network-layer', 'line-color', lineColor);
        map.setPaintProperty('network-layer', 'line-opacity', lineOpacity);
      }
    } else {
      map.addSource('network-source', { type: 'geojson', data: geojson });
      map.addLayer({
        id: 'network-layer',
        type: 'line',
        source: 'network-source',
        layout: {
          'line-join': 'round',
          'line-cap': 'round',
          'visibility': networkVisible ? 'visible' : 'none'
        },
        paint: {
          'line-color': lineColor,
          'line-width': 1.0,
          'line-opacity': lineOpacity
        }
      });
    }
    pendingNetworkData = null;
  }

  async function loadRoadNetwork() {
    try {
      const resp = await fetch('/api/network?max_edges=2500');
      const data = await resp.json();
      if (!data.edges || data.edges.length === 0) return 0;

      applyRoadNetworkData(data.edges);
      return data.displayed_edges;
    } catch (err) {
      console.error('Road network load error:', err);
      return 0;
    }
  }

  function applyHeatmapData(points) {
    if (!points || points.length === 0) return;
    cachedHeatmapPoints = points;

    if (!map) {
      pendingHeatmapData = points;
      return;
    }

    if (!map.isStyleLoaded()) {
      pendingHeatmapData = points;
      map.once('style.load', () => {
        if (pendingHeatmapData) applyHeatmapData(pendingHeatmapData);
      });
      return;
    }

    const features = points.map(p => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [p.lon, p.lat]
      },
      properties: { intensity: p.intensity }
    }));

    const geojson = { type: 'FeatureCollection', features };

    if (map.getSource('heatmap-source')) {
      map.getSource('heatmap-source').setData(geojson);
    } else {
      map.addSource('heatmap-source', { type: 'geojson', data: geojson });
      map.addLayer({
        id: 'heatmap-layer',
        type: 'heatmap',
        source: 'heatmap-source',
        layout: {
          'visibility': heatmapVisible ? 'visible' : 'none'
        },
        paint: {
          'heatmap-weight': ['get', 'intensity'],
          'heatmap-intensity': 1.4,
          'heatmap-color': [
            'interpolate',
            ['linear'],
            ['heatmap-density'],
            0, 'rgba(0, 115, 230, 0)',
            0.2, '#00d4ff',
            0.5, '#ff9f00',
            0.8, '#ff4d6d',
            1, '#ffffff'
          ],
          'heatmap-radius': 22,
          'heatmap-opacity': 0.85
        }
      });
    }
    pendingHeatmapData = null;
  }

  async function loadHeatmap() {
    try {
      const resp = await fetch('/api/hotspots?top_n=150');
      const data = await resp.json();
      if (!data.points || data.points.length === 0) return 0;

      applyHeatmapData(data.points);
      return data.points.length;
    } catch (err) {
      console.error('Heatmap load error:', err);
      return 0;
    }
  }

  function clearRoutes() {
    if (!map) return;
    cachedRecommendations = null;

    // Remove route markers
    routeMarkers.forEach(m => m.remove());
    routeMarkers = [];

    // Remove any existing route layers and sources
    for (let i = 0; i < 10; i++) {
      const glowId = `route-glow-${i}`;
      const lineId = `route-line-${i}`;
      const waypointsId = `route-waypoints-${i}`;
      const sourceId = `route-source-${i}`;
      const ptsSourceId = `${sourceId}-pts`;

      if (map.getLayer(glowId)) map.removeLayer(glowId);
      if (map.getLayer(lineId)) map.removeLayer(lineId);
      if (map.getLayer(waypointsId)) map.removeLayer(waypointsId);
      if (map.getSource(ptsSourceId)) map.removeSource(ptsSourceId);
      if (map.getSource(sourceId)) map.removeSource(sourceId);
    }
  }

  function drawRoutes(recommendations) {
    if (!map) return;
    clearRoutes();

    if (!recommendations || recommendations.length === 0) return;
    cachedRecommendations = recommendations;

    const allCoordinates = [];

    recommendations.forEach((rec, idx) => {
      if (!rec.route || rec.route.length < 2) return;

      const color = ROUTE_COLORS[idx % ROUTE_COLORS.length];
      const coords = rec.route.map(pt => [pt.lon, pt.lat]);
      allCoordinates.push(...coords);

      const sourceId = `route-source-${idx}`;
      const glowId = `route-glow-${idx}`;
      const lineId = `route-line-${idx}`;
      const waypointsId = `route-waypoints-${idx}`;

      // GeoJSON LineString
      map.addSource(sourceId, {
        type: 'geojson',
        data: {
          type: 'Feature',
          geometry: { type: 'LineString', coordinates: coords }
        }
      });

      // Glow layer
      map.addLayer({
        id: glowId,
        type: 'line',
        source: sourceId,
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
          'line-color': color,
          'line-width': 10,
          'line-opacity': 0.18
        }
      });

      // Main route line
      map.addLayer({
        id: lineId,
        type: 'line',
        source: sourceId,
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
          'line-color': color,
          'line-width': 4,
          'line-opacity': 0.95
        }
      });

      // Waypoint dots
      const waypointFeatures = rec.route
        .filter((_, ptIdx) => ptIdx === 0 || ptIdx === rec.route.length - 1 || ptIdx % 3 === 0)
        .map(pt => ({
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [pt.lon, pt.lat] },
          properties: {
            title: `${rec.taxi_id} — Segment ${pt.segment_id}`,
            pickup_prob: (pt.pickup_prob * 100).toFixed(1),
            earning: pt.expected_earning.toFixed(2)
          }
        }));

      map.addSource(`${sourceId}-pts`, {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: waypointFeatures }
      });

      map.addLayer({
        id: waypointsId,
        type: 'circle',
        source: `${sourceId}-pts`,
        paint: {
          'circle-radius': 4.5,
          'circle-color': color,
          'circle-stroke-width': 1.5,
          'circle-stroke-color': '#ffffff'
        }
      });

      // Destination pin marker
      const endPt = rec.route[rec.route.length - 1];
      const endEl = document.createElement('div');
      endEl.style.cssText = `
        background:${color}; width:14px; height:14px; border-radius:50%;
        border:2px solid white; box-shadow:0 0 12px ${color}; cursor:pointer;
      `;
      const endPopup = new mapboxgl.Popup({ offset: 12 })
        .setHTML(`<b>${rec.taxi_id} Destination</b><br>Expected Earning: <b>¥${rec.total_expected_earning.toFixed(2)}</b>`);

      const endMarker = new mapboxgl.Marker({ element: endEl })
        .setLngLat([endPt.lon, endPt.lat])
        .setPopup(endPopup)
        .addTo(map);

      routeMarkers.push(endMarker);
    });

    // Fit map bounds to show full route
    if (allCoordinates.length > 0) {
      const bounds = allCoordinates.reduce((b, coord) => b.extend(coord), new mapboxgl.LngLatBounds(allCoordinates[0], allCoordinates[0]));
      map.fitBounds(bounds, { padding: 60, duration: 1000 });
    }
  }

  function toggleHeatmap(visible) {
    heatmapVisible = visible;
    if (map && map.getLayer('heatmap-layer')) {
      map.setLayoutProperty('heatmap-layer', 'visibility', visible ? 'visible' : 'none');
    }
  }

  function toggleNetwork(visible) {
    networkVisible = visible;
    if (map && map.getLayer('network-layer')) {
      map.setLayoutProperty('network-layer', 'visibility', visible ? 'visible' : 'none');
    }
  }

  function reloadAllLayers() {
    if (cachedNetworkEdges) {
      applyRoadNetworkData(cachedNetworkEdges);
    } else {
      loadRoadNetwork();
    }

    if (cachedHeatmapPoints) {
      applyHeatmapData(cachedHeatmapPoints);
    } else {
      loadHeatmap();
    }

    if (cachedRecommendations) {
      drawRoutes(cachedRecommendations);
    }

    if (lastStartPos) {
      setStartMarker(lastStartPos.lat, lastStartPos.lon);
    }
  }

  function setStyle(styleName) {
    if (!map) return;
    currentStyle = styleName;
    const newStyle = getMapStyle(styleName, mapboxToken);

    // Update button states in Mapbox overlay bar
    const btnDark = document.getElementById('btn-style-dark');
    const btnLight = document.getElementById('btn-style-light');
    const btnStreets = document.getElementById('btn-style-streets');
    if (btnDark) btnDark.classList.toggle('active', styleName === 'dark');
    if (btnLight) btnLight.classList.toggle('active', styleName === 'light');
    if (btnStreets) btnStreets.classList.toggle('active', styleName === 'streets');

    map.setStyle(newStyle);

    // Re-apply layers once new style finishes loading
    map.once('style.load', () => {
      reloadAllLayers();
    });
  }

  function setTheme(themeName) {
    currentTheme = themeName;
    if (themeName === 'light') {
      setStyle('light');
    } else {
      setStyle('dark');
    }
  }

  async function searchGeocode(query) {
    if (!query || !query.trim()) return null;
    if (!hasValidToken(mapboxToken)) {
      console.warn('Geocoding search requires MAPBOX_ACCESS_TOKEN in .env');
      return null;
    }
    const token = mapboxToken;

    const endpoint = `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(query)}.json?access_token=${token}&proximity=116.4074,39.9042`;

    try {
      const resp = await fetch(endpoint);
      const data = await resp.json();
      if (data.features && data.features.length > 0) {
        const first = data.features[0];
        const [lon, lat] = first.center;
        if (map) {
          map.flyTo({ center: [lon, lat], zoom: 14, duration: 1500 });
          setStartMarker(lat, lon);
          if (onClickCallback) onClickCallback(lat, lon);
        }
        return first;
      }
      return null;
    } catch (err) {
      console.error('Geocoding error:', err);
      return null;
    }
  }

  return {
    init,
    resize,
    setStartMarker,
    loadHeatmap,
    loadRoadNetwork,
    drawRoutes,
    clearRoutes,
    toggleHeatmap,
    toggleNetwork,
    setStyle,
    setTheme,
    searchGeocode,
  };
})();
