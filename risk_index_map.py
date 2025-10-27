import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
import io

# -------------------------
# App Configuration
# -------------------------

st.set_page_config(page_title="Termite Risk Index Viewer", layout="wide")

st.title("🏠 Termite Risk Index Viewer")

# -------------------------
# Global Style + Toast
# -------------------------
st.markdown("""
<style>
@media (max-width: 600px) {
    h1 {font-size: 1.3rem !important;}
    .stRadio label, .stSelectbox label {font-size: 0.9rem !important;}
    .stDataFrame {font-size: 0.8rem !important;}
}
.fab {
    position: fixed;
    bottom: 70px;
    right: 20px;
    background-color: #0078D4;
    color: white;
    padding: 14px;
    border-radius: 50%;
    font-size: 20px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    text-decoration: none;
    z-index: 9999;
}
.fab:hover {background-color: #005fa3;}
#toast {
    visibility: hidden;
    min-width: 280px;
    background-color: #333;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 12px;
    position: fixed;
    z-index: 9999;
    left: 50%;
    bottom: 40px;
    transform: translateX(-50%);
    font-size: 14px;
}
#toast.show {
    visibility: visible;
    animation: fadein 0.5s, fadeout 0.5s 3s;
}
@keyframes fadein {from {bottom: 0; opacity: 0;} to {bottom: 40px; opacity: 1;}}
@keyframes fadeout {from {bottom: 40px; opacity: 1;} to {bottom: 0; opacity: 0;}}
</style>
<div id="toast">⚠️</div>
<script>
function showToast(msg) {
  var x = document.getElementById("toast");
  x.innerText = msg;
  x.className = "show";
  setTimeout(function(){ x.className = x.className.replace("show", ""); }, 4000);
}
</script>
""", unsafe_allow_html=True)

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("master_with_inspection_counts_sm202510.csv")

df = load_data()

# -------------------------
# Auto-detect columns
# -------------------------
def find_col(cols, candidates):
    cols_l = [c.lower() for c in cols]
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols[cols_l.index(cand.lower())]
    for cand in candidates:
        for i, c in enumerate(cols):
            if c.lower().startswith(cand.lower()):
                return c
    return None

lat_col = find_col(df.columns, ["latitude", "lat"])
lon_col = find_col(df.columns, ["longitude", "lon", "lng"])
addr_col = find_col(df.columns, ["matched_address", "address", "full_address"])
street_col = find_col(df.columns, ["street", "street_name"])
risk_col = find_col(df.columns, ["risk_level", "risk", "category"])
risk_score_col = find_col(df.columns, ["risk_score", "score"])
recent_insp_col = find_col(df.columns, ["most recent inspection", "most_recent_insp"])
num_insp_col = find_col(df.columns, ["# of inspections", "num_inspections", "total_inspections"])

if not lat_col or not lon_col or not risk_col:
    st.error("CSV must contain latitude, longitude, and risk_level columns.")
    st.stop()

# -------------------------
# Sidebar Controls
# -------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    with st.expander("Search and Radius", expanded=False):
        radius_toggle = st.radio("Select radius (ft)", [200, 300], horizontal=True)
        radius_m = radius_toggle * 0.3048
        search_opts = sorted(df[addr_col].dropna().unique()) if addr_col else []
        search_choice = st.selectbox("🔍 Search Address", [""] + search_opts)

# -------------------------
# Prepare Data
# -------------------------
df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
df = df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

COLOR = {"Very High": "#8B0000", "High": "#FF0000", "Moderate": "#FFA500", "Low": "#FFFF00"}
center_lat, center_lon = df[lat_col].mean(), df[lon_col].mean()

# -------------------------
# Session State
# -------------------------
st.session_state.setdefault("selected", None)
st.session_state.setdefault("nearby_df", pd.DataFrame())
st.session_state.setdefault("active_tab", "map")

# -------------------------
# Distance Function
# -------------------------
def haversine_vec(lat0, lon0, lats, lons):
    R = 6371000.0
    phi1, phi2 = np.radians(lat0), np.radians(lats)
    dphi, dlambda = np.radians(lats - lat0), np.radians(lons - lon0)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

# -------------------------
# Handle Search
# -------------------------
if search_choice:
    sel_row = df[df[addr_col] == search_choice]
    if not sel_row.empty:
        st.session_state.selected = sel_row.iloc[0].to_dict()
        st.session_state.active_tab = "table"

# -------------------------
# Map Builders
# -------------------------
def build_base_map():
    return folium.Map(location=[center_lat, center_lon], zoom_start=13,
                      tiles="https://mt1.google.com/vt/lyrs=y,h&x={x}&y={y}&z={z}", attr="Google")

def build_focused_map_and_nearby(selected_dict):
    lat, lon = float(selected_dict[lat_col]), float(selected_dict[lon_col])
    risk_val = selected_dict.get(risk_col, "")
    risk_color = COLOR.get(risk_val, "gray")
    m = folium.Map(location=[lat, lon], zoom_start=19,
                   tiles="https://mt1.google.com/vt/lyrs=y,h&x={x}&y={y}&z={z}", attr="Google")
    folium.Circle(location=[lat, lon], radius=radius_m, color="blue", fill=False, weight=2).add_to(m)

    # Pulsating center
    html = f"""
    <div style="background-color:{risk_color};
                width:20px;height:20px;
                border-radius:50%;border:2px solid black;
                animation:pulse 1s infinite;"></div>
    <style>
    @keyframes pulse {{
        0% {{transform:scale(0.8);opacity:0.7;}}
        50% {{transform:scale(1.4);opacity:0.4;}}
        100% {{transform:scale(0.8);opacity:0.7;}}
    }}
    </style>
    """
    folium.Marker(location=[lat, lon], icon=folium.DivIcon(html=html)).add_to(m)

    distances = haversine_vec(lat, lon, df[lat_col].values, df[lon_col].values)
    df["dist_m"] = distances
    nearby_df = df[df["dist_m"] <= radius_m].copy()
    if nearby_df.empty:
        return m, nearby_df

    nearby_df["Distance (ft)"] = (nearby_df["dist_m"] * 3.28084).round(0).astype("Int64")

    # Add neighbors + animated lines
    for i, r in nearby_df.iterrows():
        if np.isclose(float(r[lat_col]), lat) and np.isclose(float(r[lon_col]), lon):
            continue
        rc = COLOR.get(r.get(risk_col, ""), "gray")
        # neighbor marker
        folium.CircleMarker(location=[r[lat_col], r[lon_col]],
                            radius=5, color="white", weight=1,
                            fill=True, fill_color=rc, fill_opacity=0.9,
                            popup=f"{r.get(street_col,'')}<br>Risk: {r.get(risk_col,'')}").add_to(m)
        # connection line
        folium.PolyLine([(r[lat_col], r[lon_col]), (lat, lon)],
                        color=rc, weight=1.2, opacity=0.6).add_to(m)

    # Add animation script for lines
    m.get_root().html.add_child(folium.Element("""
    <style>
    path.leaflet-interactive {
        stroke-dasharray: 5;
        animation: draw 2s linear infinite;
    }
    @keyframes draw {
        0% { stroke-dashoffset: 10; }
        100% { stroke-dashoffset: 0; }
    }
    </style>
    """))
    return m, nearby_df

# -------------------------
# Responsive Map Sizing
# -------------------------
def get_map_dimensions():
    try:
        ua = st.runtime.scriptrunner.script_run_context.session_info.user_agent
        if "Mobile" in ua:
            return 350, 500
    except Exception:
        pass
    return 900, 600

map_width, map_height = get_map_dimensions()

# -------------------------
# Tabs
# -------------------------
if st.session_state.active_tab == "table":
    tab2, tab1 = st.tabs(["📊 Nearby Addresses", "🗺️ Map"])
else:
    tab1, tab2 = st.tabs(["🗺️ Map", "📊 Nearby Addresses"])

with tab1:
    if st.session_state.selected is None:
        m = build_base_map()
    else:
        m, nearby = build_focused_map_and_nearby(st.session_state.selected)
        st.session_state.nearby_df = nearby
        if nearby.empty:
            addr = st.session_state.selected.get(addr_col, "this location")
            st.components.v1.html(f"<script>showToast('⚠️ No nearby addresses found within {radius_toggle} ft of {addr}')</script>", height=0)
    map_data = st_folium(m, width=map_width, height=map_height, use_container_width=True)

    if map_data and map_data.get("last_clicked") is not None:
        click_lat = map_data["last_clicked"]["lat"]
        click_lon = map_data["last_clicked"]["lng"]
        distances = haversine_vec(click_lat, click_lon, df[lat_col].values, df[lon_col].values)
        nearest_idx = np.argmin(distances)
        st.session_state.selected = df.iloc[nearest_idx].to_dict()
        st.session_state.active_tab = "table"
        st.rerun()

with tab2:
    if not st.session_state.nearby_df.empty:
        nearby_df = st.session_state.nearby_df
        table_cols = [street_col, risk_col]
        if risk_score_col in nearby_df.columns:
            table_cols.append(risk_score_col)
        table_cols.append("Distance (ft)")
        for c in [recent_insp_col, num_insp_col]:
            if c and c in nearby_df.columns:
                table_cols.append(c)
        display_df = nearby_df[table_cols].copy().fillna("")

        sel_street = st.session_state.selected.get(street_col, "")
        risk_val = st.session_state.selected.get(risk_col, "")
        risk_color = COLOR.get(risk_val, "gray")

        st.markdown(
            f"<div style='background:{risk_color};color:black;font-size:16px;padding:8px;border-radius:6px;text-align:center;'>"
            f"{len(display_df)} Addresses within {radius_toggle} ft of {sel_street} (Risk: {risk_val})</div>",
            unsafe_allow_html=True)

        st.dataframe(display_df.astype(str), use_container_width=True, hide_index=True)

# -------------------------
# Floating Back-to-Top Button
# -------------------------
st.markdown('<a href="#top" class="fab">⬆️</a>', unsafe_allow_html=True)
