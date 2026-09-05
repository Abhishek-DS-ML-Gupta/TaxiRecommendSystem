"""
streamlit_app.py — Interactive Researcher UI for the Taxi Route Recommender

Run with:
    streamlit run streamlit_app.py

This provides a notebook-style alternative to the FastAPI web app,
useful for researchers who want to explore the data, run experiments,
and visualize results interactively without a full server setup.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
_SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, os.path.abspath(_SRC_DIR))

from data_processor import DataProcessor
from route_optimizer import RouteOptimizer

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Taxi Route Recommender — INFORMS JoC 2021.0112",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stMetric label { font-size: 11px !important; font-weight: 600 !important; letter-spacing: 0.8px; text-transform: uppercase; color: #7aa3c0 !important; }
  .stMetric .metric-container { background: rgba(12,24,48,0.6); border: 1px solid rgba(0,212,255,0.15); border-radius: 12px; }
  div[data-testid="metric-container"] { background: rgba(0,212,255,0.03); border: 1px solid rgba(0,212,255,0.15); border-radius: 12px; padding: 12px; }
  .stButton button { background: linear-gradient(135deg, #00b8d9, #0073e6) !important; color: white !important; border: none !important; font-weight: 600 !important; border-radius: 10px !important; }
  .stButton button:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(0,115,230,0.4) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading (cached) ────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

@st.cache_resource(show_spinner="Loading Beijing road network...")
def load_data():
    processor = DataProcessor(data_dir=DATA_DIR)
    G, edges_df, vertices_df = processor.load_network()

    # Load network properties
    props_path = os.path.join(DATA_DIR, 'network_properties.csv')
    network_props = None
    if os.path.exists(props_path):
        network_props = pd.read_csv(props_path, header=None, names=[
            'source_id', 'target_id', 'src_lat', 'src_lon', 'tgt_lat', 'tgt_lon',
            'pickup_count', 'taxi_count', 'pickup_prob', 'expected_earning'
        ])

    # Build pickup probs
    pickup_probs = {}
    earnings = {}
    edge_coords = {}
    node_coords = {int(r['node_id']): (float(r['lat']), float(r['lon'])) for _, r in vertices_df.iterrows()}

    if network_props is not None:
        for _, row in network_props.iterrows():
            src, tgt = int(row['source_id']), int(row['target_id'])
            prob = float(row['pickup_prob']) if not pd.isna(row['pickup_prob']) else 0.0
            earn = float(row['expected_earning']) if not pd.isna(row['expected_earning']) else 0.0
            mask = (edges_df['source_id'] == src) & (edges_df['target_id'] == tgt)
            matched = edges_df.loc[mask, 'edge_id']
            if len(matched) > 0:
                eid = int(matched.iloc[0])
                pickup_probs[eid] = min(max(prob, 0.0), 1.0)
                earnings[eid] = max(earn, 0.0)
                mid_lat = (float(row['src_lat']) + float(row['tgt_lat'])) / 2.0
                mid_lon = (float(row['src_lon']) + float(row['tgt_lon'])) / 2.0
                edge_coords[eid] = (mid_lat, mid_lon)

    # Fill missing
    for _, row in edges_df.iterrows():
        eid = int(row['edge_id'])
        src, tgt = int(row['source_id']), int(row['target_id'])
        if eid not in pickup_probs:
            pickup_probs[eid] = float(np.random.uniform(0.05, 0.4))
            earnings[eid] = float(np.random.uniform(10.0, 50.0))
        if eid not in edge_coords:
            src_c = node_coords.get(src, (39.9, 116.4))
            tgt_c = node_coords.get(tgt, (39.9, 116.4))
            edge_coords[eid] = ((src_c[0]+tgt_c[0])/2, (src_c[1]+tgt_c[1])/2)

    return G, edges_df, vertices_df, pickup_probs, earnings, edge_coords, node_coords

@st.cache_resource(show_spinner="Building routing graph...")
def build_route_graph(_edges_df, _node_coords):
    import networkx as nx
    route_G = nx.DiGraph()
    travel_times = {}
    edge_ids = _edges_df['edge_id'].unique()
    source_group = _edges_df.groupby('source_id')

    for eid in edge_ids:
        route_G.add_node(int(eid))

    for _, row in _edges_df.iterrows():
        tgt_node = int(row['target_id'])
        curr_edge = int(row['edge_id'])
        src = int(row['source_id'])
        src_c = _node_coords.get(src, (39.9, 116.4))
        tgt_c = _node_coords.get(tgt_node, (39.9, 116.4))
        dist = ((src_c[0]-tgt_c[0])**2 + (src_c[1]-tgt_c[1])**2)**0.5
        t = max(dist * 100.0, 0.5)
        if tgt_node in source_group.groups:
            for _, nr in source_group.get_group(tgt_node).iterrows():
                next_edge = int(nr['edge_id'])
                route_G.add_edge(curr_edge, next_edge)
                travel_times[(curr_edge, next_edge)] = t

    return route_G, travel_times

# ─── Load All ─────────────────────────────────────────────────────────────────

G, edges_df, vertices_df, pickup_probs, earnings, edge_coords, node_coords = load_data()
route_G, travel_times = build_route_graph(edges_df, node_coords)

valid_starts = [n for n, deg in route_G.out_degree() if deg > 0]

# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🚕 Taxi Route Recommender")
st.markdown(
    "**INFORMS Journal on Computing · 2021.0112** · "
    "Liu, Teng, Chen & Xiong (2023) · "
    "[DOI: 10.1287/ijoc.2021.0112](https://doi.org/10.1287/ijoc.2021.0112)"
)
st.divider()

# ─── Sidebar: Controls ────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Parameters")

    time_limit = st.slider("Search Time Limit τ (min)", 2, 30, 10, 1)
    num_taxis = st.slider("Number of Taxis", 1, 10, 1, 1,
                          help="1 = RRS, >1 = SRRS (Sequential Route Recommendation)")
    scenario = st.selectbox("Scenario", ["normal", "peak", "rainy", "night"],
                            format_func=lambda x: {"normal":"☀️ Normal","peak":"🔥 Peak Hour",
                                                    "rainy":"🌧 Rainy","night":"🌙 Night"}[x])

    st.divider()
    st.markdown("**Starting Location**")
    start_lat = st.number_input("Latitude", value=39.9042, format="%.5f", step=0.001)
    start_lon = st.number_input("Longitude", value=116.4074, format="%.5f", step=0.001)

    algo_label = "🔍 RRS" if num_taxis == 1 else f"⚡ SRRS ({num_taxis} taxis)"
    run_btn = st.button(f"Run {algo_label}", use_container_width=True)

    st.divider()
    st.caption("**About:** Research implementation of the GCN-LSTM + recursive route searching system for optimal taxi routing.")

# ─── Tab Layout ───────────────────────────────────────────────────────────────

tab_map, tab_analysis, tab_model, tab_data = st.tabs([
    "🗺 Route Map", "📈 Route Analysis", "🧪 Model Results", "📊 Dataset Explorer"
])

# ─── Helper: Snap to nearest segment ─────────────────────────────────────────

def snap_to_nearest(lat, lon):
    best_eid, best_dist = None, float('inf')
    for eid in valid_starts:
        if eid not in edge_coords:
            continue
        elat, elon = edge_coords[eid]
        d = (lat - elat)**2 + (lon - elon)**2
        if d < best_dist:
            best_dist = d
            best_eid = eid
    return best_eid

# ─── Helper: Apply scenario ───────────────────────────────────────────────────

def apply_scenario(probs, scen):
    if scen == "peak":   return {k: min(v*1.4, 1.0) for k,v in probs.items()}
    if scen == "rainy":  return {k: min(v*1.6, 1.0) for k,v in probs.items()}
    if scen == "night":  return {k: v*0.6 for k,v in probs.items()}
    return probs.copy()

# ─── Helper: Build route result df ───────────────────────────────────────────

def build_route_df(route, pickup_probs, earnings, edge_coords):
    rows = []
    for i, seg_id in enumerate(route):
        lat, lon = edge_coords.get(seg_id, (0, 0))
        rows.append({
            "Step": i+1,
            "Segment ID": seg_id,
            "Latitude": round(lat, 6),
            "Longitude": round(lon, 6),
            "Pickup Probability": round(pickup_probs.get(seg_id, 0), 4),
            "Expected Earning (¥)": round(earnings.get(seg_id, 0), 2),
        })
    return pd.DataFrame(rows)

# ─── Run Recommendation ───────────────────────────────────────────────────────

if run_btn or "last_result" not in st.session_state:
    if run_btn:
        optimizer = RouteOptimizer(route_G, time_limit_tau=time_limit)
        mod_probs = apply_scenario(pickup_probs, scenario)
        start_seg = snap_to_nearest(start_lat, start_lon)

        if start_seg is None:
            st.error("Could not find a nearby road segment. Try adjusting coordinates.")
        else:
            with st.spinner("Computing optimal route..."):
                t0 = time.time()
                if num_taxis == 1:
                    route, profit = optimizer.RRS(start_seg, 0.0, mod_probs, earnings, travel_times)
                    recs = [{"taxi_id": "Taxi_1", "route": route, "profit": profit}]
                else:
                    taxis_q = [(f"Taxi_{i+1}", start_seg) for i in range(num_taxis)]
                    srrs_out = optimizer.SRRS(taxis_q, mod_probs, earnings, travel_times)
                    recs = []
                    for tid, route in srrs_out.items():
                        _, p = optimizer.RRS(route[0] if route else start_seg, 0.0, mod_probs, earnings, travel_times)
                        recs.append({"taxi_id": tid, "route": route, "profit": p})
                elapsed = time.time() - t0

            st.session_state["last_result"] = {
                "recs": recs,
                "start_lat": start_lat,
                "start_lon": start_lon,
                "start_seg": start_seg,
                "elapsed": elapsed,
                "algorithm": "RRS" if num_taxis == 1 else "SRRS",
            }

# ─── TAB: Route Map ───────────────────────────────────────────────────────────

with tab_map:
    COLORS = ['#00d4ff','#ff9f00','#00ff9d','#b97aff','#ff4d6d','#ffdf00','#47c6ff','#ff7eb3']

    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        recs = res["recs"]

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Algorithm", res["algorithm"])
        col2.metric("Expected Earning", f"¥{recs[0]['profit']:.2f}" if recs else "—")
        col3.metric("Route Segments", len(recs[0]['route']) if recs else 0)
        col4.metric("Compute Time", f"{res['elapsed']*1000:.0f}ms")

        # Folium map
        m = folium.Map(
            location=[res["start_lat"], res["start_lon"]],
            zoom_start=14,
            tiles='CartoDB dark_matter',
        )

        # Start marker
        folium.Marker(
            [res["start_lat"], res["start_lon"]],
            popup=f"Start: ({res['start_lat']:.5f}, {res['start_lon']:.5f})",
            tooltip="🚕 Taxi Start",
            icon=folium.Icon(color='blue', icon='taxi', prefix='fa')
        ).add_to(m)

        # Draw routes
        for i, rec in enumerate(recs):
            color = COLORS[i % len(COLORS)]
            coords = [edge_coords[s] for s in rec['route'] if s in edge_coords]
            if len(coords) >= 2:
                folium.PolyLine(
                    [(lat, lon) for lat, lon in coords],
                    color=color,
                    weight=4,
                    opacity=0.9,
                    tooltip=f"{rec['taxi_id']} · ¥{rec['profit']:.2f}"
                ).add_to(m)
                # End dot
                last = coords[-1]
                folium.CircleMarker(
                    last, radius=6, fill=True, fill_color=color, color='white',
                    fill_opacity=1, tooltip=f"{rec['taxi_id']} route end"
                ).add_to(m)

        # Sample hotspots as heatmap
        top_hotspots = sorted(
            [(eid, pickup_probs[eid]) for eid in pickup_probs if eid in edge_coords],
            key=lambda x: x[1], reverse=True
        )[:100]

        from folium.plugins import HeatMap
        heat_data = [[edge_coords[eid][0], edge_coords[eid][1], prob]
                     for eid, prob in top_hotspots if edge_coords[eid][0] != 0]
        HeatMap(heat_data, radius=14, blur=18,
                gradient={'0.3':'#0073e6','0.6':'#00d4ff','0.8':'#ff9f00','1.0':'#ff4d6d'}
                ).add_to(m)

        st_folium(m, width=None, height=520, use_container_width=True)

    else:
        st.info("👈 Set parameters in the sidebar and click **Run** to get route recommendations.")
        # Show base map
        m = folium.Map(location=[39.9042, 116.4074], zoom_start=13, tiles='CartoDB dark_matter')
        heat_data = []
        for eid, prob in list(pickup_probs.items())[:100]:
            if eid in edge_coords:
                lat, lon = edge_coords[eid]
                if lat != 0:
                    heat_data.append([lat, lon, prob])
        from folium.plugins import HeatMap
        HeatMap(heat_data, radius=14, blur=18).add_to(m)
        st_folium(m, width=None, height=520, use_container_width=True)

# ─── TAB: Route Analysis ──────────────────────────────────────────────────────

with tab_analysis:
    if "last_result" not in st.session_state:
        st.info("Run a recommendation first.")
    else:
        res = st.session_state["last_result"]
        recs = res["recs"]

        st.subheader("Route Details")
        for i, rec in enumerate(recs):
            with st.expander(f"🚕 {rec['taxi_id']} — ¥{rec['profit']:.2f} expected", expanded=(i==0)):
                df = build_route_df(rec['route'], pickup_probs, earnings, edge_coords)
                st.dataframe(df, use_container_width=True)

                if len(df) > 0:
                    fig = px.bar(
                        df, x='Step', y='Pickup Probability',
                        color='Pickup Probability',
                        color_continuous_scale=['#0073e6','#00d4ff','#ff9f00','#ff4d6d'],
                        title=f"{rec['taxi_id']} — Pickup Probability Along Route",
                        labels={'Step':'Route Step','Pickup Probability':'P(pickup)'},
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#7aa3c0',
                        showlegend=False,
                        margin=dict(l=0, r=0, t=40, b=0),
                        height=220,
                    )
                    fig.update_xaxes(showgrid=False)
                    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
                    st.plotly_chart(fig, use_container_width=True)

# ─── TAB: Model Results ───────────────────────────────────────────────────────

with tab_model:
    results_path = os.path.join(os.path.dirname(__file__), 'results', 'summary_result_of_gcn_lstm_split.csv')
    if not os.path.exists(results_path):
        st.warning("Results CSV not found at results/summary_result_of_gcn_lstm_split.csv")
    else:
        df_results = pd.read_csv(results_path)

        st.subheader("GCN-LSTM Hyperparameter Search Results")
        st.caption(f"{len(df_results)} experiment runs | Lower MAE = better model")

        col1, col2, col3 = st.columns(3)
        col1.metric("Best MAE", f"{df_results['mae'].min():.5f}")
        col2.metric("Mean MAE", f"{df_results['mae'].mean():.5f}")
        col3.metric("Total Experiments", len(df_results))

        # MAE scatter
        fig = px.scatter(
            df_results,
            x=df_results.index,
            y='mae',
            color='learning_rate',
            size_max=8,
            color_continuous_scale='teal',
            title='MAE Across All Experiment Runs',
            labels={'index': 'Experiment #', 'mae': 'MAE'},
            hover_data=['date','gc_dims','lstm_dims','learning_rate'],
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#7aa3c0', height=300, margin=dict(l=0,r=0,t=40,b=0)
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig, use_container_width=True)

        # Best by date
        best_by_date = df_results.loc[df_results.groupby('date')['mae'].idxmin()]
        st.subheader("Best Configuration per Date")
        st.dataframe(
            best_by_date[['date','mae','learning_rate','gc_dims','lstm_dims','prior_days','prior_hours']],
            use_container_width=True
        )

# ─── TAB: Dataset Explorer ────────────────────────────────────────────────────

with tab_data:
    st.subheader("Dataset Explorer")

    dataset = st.selectbox("Select Dataset", [
        "Road Network Edges", "Road Network Vertices",
        "Network Properties (Pickup Data)", "Weather Data",
    ])

    @st.cache_data
    def load_dataset(name):
        if name == "Road Network Edges":
            return pd.read_csv(os.path.join(DATA_DIR, 'refined_edges.csv'),
                               header=None, names=['edge_id','source_id','target_id'])
        elif name == "Road Network Vertices":
            return pd.read_csv(os.path.join(DATA_DIR, 'refined_vertice.csv'),
                               header=None, names=['node_id','lat','lon'])
        elif name == "Network Properties (Pickup Data)":
            return pd.read_csv(os.path.join(DATA_DIR, 'network_properties.csv'),
                               header=None, names=['source_id','target_id','src_lat','src_lon',
                                                   'tgt_lat','tgt_lon','pickup_count','taxi_count',
                                                   'pickup_prob','expected_earning'])
        elif name == "Weather Data":
            return pd.read_csv(os.path.join(DATA_DIR, 'weather.csv'))
        return None

    df_show = load_dataset(dataset)
    if df_show is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df_show):,}")
        col2.metric("Columns", len(df_show.columns))
        col3.metric("Memory", f"{df_show.memory_usage(deep=True).sum() / 1e6:.1f} MB")

        st.dataframe(df_show.head(500), use_container_width=True)
        st.caption(f"Showing first 500 of {len(df_show):,} rows")

        if dataset == "Network Properties (Pickup Data)":
            st.subheader("Pickup Probability Distribution")
            fig = px.histogram(
                df_show[df_show['pickup_prob'] > 0],
                x='pickup_prob', nbins=50,
                color_discrete_sequence=['#00d4ff'],
                title='Distribution of Pickup Probabilities (real data)',
                labels={'pickup_prob': 'Pickup Probability'},
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#7aa3c0', height=280, margin=dict(l=0,r=0,t=40,b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
