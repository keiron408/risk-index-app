import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Termite Risk Index Viewer", layout="wide")
st.title("🏠 Termite Risk Index Viewer (DEBUG ENABLED)")

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv("master_with_inspection_counts_202512.csv")

df = load_data()

# ============================================================
# AUTO-DETECT COLUMNS
# ============================================================
def find_col(cols, candidates):
    cols_lower = [c.lower() for c in cols]
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols[cols_lower.index(cand.lower())]
    for cand in candidates:
        for col in cols:
            if col.lower().startswith(cand.lower()):
                return col
    return None

lat_col = find_col(df.columns, ["latitude", "lat"])
lon_col = find_col(df.columns, ["longitude", "lon", "lng"])
addr_col = find_col(df.columns, ["matched_address", "address", "full_address", "search address"])
street_col = find_col(df.columns, ["street", "street_name", "fulladdress"])
risk_col = find_col(df.columns, ["risk_level", "risk"])
risk_score_col = find_col(df.columns, ["risk_score"])
recent_insp_col = find_col(df.columns, ["most recent inspection"])
num_insp_col = find_col(df.columns, ["# of inspections"])

if not street_col:
    street_col = addr_col

# Normalize risk levels
df[risk_col] = (
    df[risk_col]
    .astype(str)
    .str.strip()
    .str.replace("_", " ")
    .str.replace("-", " ")
    .str.title()
)

df[risk_col] = df[risk_col].replace({
    "High Risk": "High",
    "Moderate Risk": "Moderate",
    "Low Risk": "Low",
    "Very High Risk": "Very High"
})

COLOR = {
    "Very High": "#8B0000",
    "High": "#FF0000",
    "Moderate": "#FFA500",
    "Low": "#FFFF00"
}

# Clean data
df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
df = df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

center_lat, center_lon = df[lat_col].mean(), df[lon_col].mean()

# ============================================================
# SESSION STATE
# ============================================================
st.session_state.setdefault("selected", None)
st.session_state.setdefault("nearby_df", pd.DataFrame())
st.session_state.setdefault("map_last_click", None)
st.session_state.setdefault("_last_click_was_from_search", False)

# ============================================================
# DISTANCE
# ============================================================
def haversine_vec(lat0, lon0, lats, lons):
    R = 6371000.0
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    lats, lons = np.radians(lats), np.radians(lons)
    dlat = lats - lat0
    dlon = lons - lon0
    a = np.sin(dlat/2)**2 + np.cos(lat0)*np.cos(lats)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ============================================================
# MAP BUILDERS (DEBUG ENABLED)
# ============================================================
def build_base_map():
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="https://mt1.google.com/vt/lyrs=y,h&x={x}&y={y}&z={z}",
        attr="Google"
    )

    # 🔥 Enable click capture for DEBUG
    m.add_child(folium.LatLngPopup())

    return m


def build_focused_map_and_nearby(selected):
    lat = float(selected[lat_col])
    lon = float(selected[lon_col])
    risk_val = selected.get(risk_col, "")
    risk_color = COLOR.get(risk_val, "gray")

    m = folium.Map(
        location=[lat, lon],
        zoom_start=18,
        tiles="https://mt1.google.com/vt/lyrs=y,h&x={x}&y={y}&z={z}",
        attr="Google"
    )

    # 🔥 Enable click capture for DEBUG
    m.add_child(folium.LatLngPopup())

    # Compute nearby
    df_copy = df.copy()
    df_copy["dist_m"] = haversine_vec(lat, lon, df_copy[lat_col], df_copy[lon_col])

    radius_ft = st.session_state.get("radius", 200)
    radius_m = radius_ft * 0.3048

    near = df_copy[df_copy["dist_m"] <= radius_m].copy()
    near["Distance (ft)"] = (near["dist_m"] * 3.28084).round(0)

    return m, near.sort_values("dist_m").reset_index(drop=True)

# ============================================================
# CLICK HANDLER (DEBUG VERSION)
# ============================================================
def handle_map_click(map_data):
    st.write("DEBUG: RAW MAP_DATA →", map_data)  # 🔥 Critical debug line

    if not map_data:
        return

    # Try all known formats
    click = None
    if isinstance(map_data, dict):
        if "last_clicked" in map_data:
            click = map_data.get("last_clicked")
        elif "last_object_clicked" in map_data:
            click = map_data.get("last_object_clicked")
        elif "clicked" in map_data:
            click = map_data.get("clicked")

    st.write("DEBUG: CLICK PARSED →", click)

    if not click:
        return

    lat = click.get("lat")
    lon = click.get("lng") or click.get("lon")

    st.write("DEBUG: lat/lon extracted →", lat, lon)

    if lat is None or lon is None:
        return

    # Record click
    st.session_state.map_last_click = {"lat": lat, "lon": lon}

    # Select nearest
    distances = haversine_vec(lat, lon, df[lat_col], df[lon_col])
    nearest_idx = np.argmin(distances)
    st.session_state.selected = df.iloc[nearest_idx].to_dict()

    st.experimental_rerun()


# ============================================================
# SEARCH UI
# ============================================================
st.write("### 🔍 Search Address")

all_addresses = sorted(df[addr_col].unique())
search = st.selectbox("Search", [""] + all_addresses)

radius_ft = st.radio("Radius (ft)", [200, 300], horizontal=True)
st.session_state.radius = radius_ft

if search:
    sel = df[df[addr_col] == search]
    if not sel.empty:
        st.session_state.selected = sel.iloc[0].to_dict()
        st.session_state.map_last_click = {"lat": None, "lon": None}
        st.session_state._last_click_was_from_search = True

# ============================================================
# INITIAL VIEW
# ============================================================
if st.session_state.selected is None:
    st.write("### 🗺️ Initial Map (Click to Debug)")
    m = build_base_map()
    map_data = st_folium(m, height=600, width=1100, use_container_width=True)

    st.write("DEBUG MAP DATA:", map_data)
    handle_map_click(map_data)

    st.stop()

# ============================================================
# SPLIT VIEW
# ============================================================
left, right = st.columns([1.3, 1])

with left:
    m, near_df = build_focused_map_and_nearby(st.session_state.selected)
    map_data = st_folium(m, height=600, width=1100, use_container_width=True)

    st.write("DEBUG MAP DATA:", map_data)
    handle_map_click(map_data)

with right:
    st.write("### 📋 Nearby Results (Debug Mode)")
    st.dataframe(near_df)
