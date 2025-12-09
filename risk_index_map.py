import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Termite Risk Index Viewer", layout="wide")
st.title("🏠 Termite Risk Index Viewer")

# ============================================================
# LOAD DATA (your CSV)
# ============================================================
df = pd.read_csv("master_with_inspection_counts_202512.csv")

# ============================================================
# FIXED, EXACT COLUMN NAMES FROM YOUR CSV
# ============================================================
addr_col = "FullAddress"
search_col = "search address"
lat_col = "Latitude"
lon_col = "Longitude"
risk_col = "risk_level"
risk_score_col = "risk_score"
recent_insp_col = "most recent inspection"
num_insp_col = "# of inspections"

# ============================================================
# SANITY CLEANUP (critical)
# ============================================================
df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
df = df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

# Drop rows with invalid search address
df = df[~df[search_col].isna()].reset_index(drop=True)

# Center map
center_lat = df[lat_col].mean()
center_lon = df[lon_col].mean()

# Normalize risk
df[risk_col] = (
    df[risk_col].astype(str).str.strip().str.title()
)

df[risk_col] = df[risk_col].replace({
    "High Risk" : "High",
    "Moderate Risk": "Moderate",
    "Low Risk": "Low",
    "Very High Risk": "Very High"
})

COLOR = {
    "Very High": "#8B0000",
    "High": "#FF0000",
    "Moderate": "#FFA500",
    "Low": "#FFFF00",
}

# ============================================================
# SESSION STATE
# ============================================================
st.session_state.setdefault("selected", None)
st.session_state.setdefault("nearby_df", pd.DataFrame())
st.session_state.setdefault("map_last_click", None)
st.session_state.setdefault("pending_map_click", False)

# ============================================================
# DISTANCE FUNCTION
# ============================================================
def haversine(lat0, lon0, lats, lons):
    R = 6371000
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)
    lat = np.radians(lats)
    lon = np.radians(lons)
    a = np.sin((lat-lat0)/2)**2 + np.cos(lat0)*np.cos(lat)*np.sin((lon-lon0)/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

# ============================================================
# SIDEBAR
# ============================================================
st.subheader("🔍 Search")

search_list = sorted(df[search_col].dropna().unique())
search_choice = st.selectbox("Search address", [""] + search_list)

radius_ft = st.radio("Radius (ft)", [200, 300], horizontal=True)
radius_m = radius_ft * 0.3048

if search_choice:
    row = df[df[search_col] == search_choice].iloc[0]
    st.session_state.selected = row.to_dict()
    st.session_state.pending_map_click = True

# ============================================================
# MAP BUILDERS
# ============================================================
def build_base_map():
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap"   # ✔ ALWAYS works
    )
    return m

def build_selected_map(selected):
    lat = float(selected[lat_col])
    lon = float(selected[lon_col])

    m = folium.Map(
        location=[lat, lon],
        zoom_start=18,
        tiles="OpenStreetMap"  # ✔ ALWAYS works
    )

    # Circle showing range
    folium.Circle(
        location=(lat, lon),
        radius=radius_m,
        color="blue",
        fill=False,
        weight=2
    ).add_to(m)

    # Marker for selected
    folium.CircleMarker(
        location=(lat, lon),
        radius=10,
        color="black",
        fill=True,
        fill_color=COLOR.get(selected[risk_col], "gray"),
        weight=2
    ).add_to(m)

    # Compute neighbors
    temp = df.copy()
    temp["dist"] = haversine(lat, lon, temp[lat_col], temp[lon_col])
    near = temp[temp["dist"] <= radius_m].copy()
    near["Distance (ft)"] = (near["dist"] * 3.28084).round().astype(int)
    near = near.sort_values("dist").reset_index(drop=True)

    # Add neighbors
    for _, r in near.iterrows():
        c = COLOR.get(r[risk_col], "gray")
        folium.CircleMarker(
            location=(r[lat_col], r[lon_col]),
            radius=6,
            color="white",
            fill=True,
            fill_color=c,
            weight=1
        ).add_to(m)

    return m, near

# ============================================================
# CLICK HANDLER
# ============================================================
def handle_map_click(map_data):
    click = map_data.get("last_clicked")
    if not click:
        return

    lat = click.get("lat")
    lon = click.get("lng")

    if lat is None or lon is None:
        return

    # Ignore first click after search
    if st.session_state.pending_map_click:
        st.session_state.pending_map_click = False
        st.session_state.map_last_click = {"lat": lat, "lon": lon}
        return

    # Avoid duplicate clicks
    last = st.session_state.map_last_click
    if last and abs(last["lat"]-lat) < 1e-9 and abs(last["lon"]-lon) < 1e-9:
        return

    st.session_state.map_last_click = {"lat": lat, "lon": lon}

    # Find nearest row
    d = haversine(lat, lon, df[lat_col], df[lon_col])
    idx = int(np.argmin(d))
    st.session_state.selected = df.iloc[idx].to_dict()

    st.rerun()

# ============================================================
# MAP + RESULTS
# ============================================================
if st.session_state.selected is None:
    m = build_base_map()
    map_data = st_folium(m, height=600, use_container_width=True,
                         returned_objects=["last_clicked"])
    handle_map_click(map_data)
    st.stop()

# ELSE: show full map + results
col_map, col_table = st.columns([1.3, 1])

with col_map:
    m, near = build_selected_map(st.session_state.selected)
    st.session_state.nearby_df = near

    map_data = st_folium(
        m,
        height=600,
        use_container_width=True,
        returned_objects=["last_clicked"]
    )
    handle_map_click(map_data)

with col_table:
    df2 = st.session_state.nearby_df

    if df2.empty:
        st.warning("No nearby addresses.")
        st.stop()

    table_cols = [addr_col, risk_col, "Distance (ft)"]
    if risk_score_col in df2.columns: table_cols.insert(2, risk_score_col)
    if recent_insp_col in df2.columns: table_cols.append(recent_insp_col)
    if num_insp_col in df2.columns: table_cols.append(num_insp_col)

    df2 = df2[table_cols].fillna("")

    st.dataframe(df2, use_container_width=True, height=550)
