import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
import io
from datetime import datetime

st.set_page_config(page_title="Termite Risk Index Mapper", layout="wide")
st.title("🏠 Local Termite Risk Index Mapper")

# Load the CSV directly
@st.cache_data
def load_data():
    df = pd.read_csv("master_with_inspection_counts_sm202510.csv")
    return df

df = load_data()

# -------------------------
# Upload CSV
# -------------------------
#uploaded_file = st.file_uploader("Upload CSV (include lat/lon, street, risk_level, inspection info)", type=["csv"])
#if uploaded_file is None:
#    st.stop()

#df = pd.read_csv(uploaded_file)

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
# Controls
# -------------------------
radius_toggle = st.radio("Select radius (ft)", [200, 300], horizontal=True)
radius_m = radius_toggle * 0.3048

# -------------------------
# Clean coordinates
# -------------------------
df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
df = df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

# -------------------------
# Colors and map center
# -------------------------
COLOR = {"Very High": "#8B0000", "High": "#FF0000", "Moderate": "#FFA500", "Low": "#FFFF00"}
center_lat = df[lat_col].mean()
center_lon = df[lon_col].mean()

# -------------------------
# Session state
# -------------------------
if "selected" not in st.session_state:
    st.session_state.selected = None
if "nearby_df" not in st.session_state:
    st.session_state.nearby_df = pd.DataFrame()

# -------------------------
# Haversine helper
# -------------------------
def haversine_vec(lat0, lon0, lats, lons):
    R = 6371000.0
    phi1 = np.radians(lat0)
    phi2 = np.radians(lats)
    dphi = np.radians(lats - lat0)
    dlambda = np.radians(lons - lon0)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

# -------------------------
# Clear / Reset
# -------------------------
col1, _ = st.columns([1, 9])
with col1:
    if st.button("Clear Selection / Reset Map"):
        st.session_state.selected = None
        st.session_state.nearby_df = pd.DataFrame()

# -------------------------
# Search box
# -------------------------
search_opts = sorted(df[addr_col].dropna().unique()) if addr_col else []
search_choice = st.selectbox("Search / select an address", [""] + search_opts)

if search_choice:
    sel_row = df[df[addr_col] == search_choice]
    if not sel_row.empty:
        st.session_state.selected = sel_row.iloc[0].to_dict()

# -------------------------
# Map build functions
# -------------------------
def build_base_map():
    return folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="https://mt1.google.com/vt/lyrs=y,h&x={x}&y={y}&z={z}",
        attr="Google"
    )

def build_focused_map_and_nearby(selected_dict):
    lat = float(selected_dict[lat_col])
    lon = float(selected_dict[lon_col])
    risk_val = selected_dict.get(risk_col, "")
    risk_color = COLOR.get(risk_val, "gray")

    m = folium.Map(
        location=[lat, lon],
        zoom_start=20,
        tiles="https://mt1.google.com/vt/lyrs=y,h&x={x}&y={y}&z={z}",
        attr="Google"
    )

    # Circle showing radius
    folium.Circle(
        location=[lat, lon],
        radius=radius_m,
        color="blue",
        fill=False,
        weight=2
    ).add_to(m)

    # Pulsating selected address
    html = f"""
    <div style="
        background-color:{risk_color};
        width:20px;
        height:20px;
        border-radius:50%;
        border: 2px solid black;
        animation: pulse 1s infinite;
    "></div>
    <style>
    @keyframes pulse {{
        0% {{transform: scale(0.8); opacity:0.7;}}
        50% {{transform: scale(1.5); opacity:0.4;}}
        100% {{transform: scale(0.8); opacity:0.7;}}
    }}
    </style>
    """
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=html)
    ).add_to(m)

    # Nearby addresses within radius
    distances = haversine_vec(lat, lon, df[lat_col].values, df[lon_col].values)
    df["dist_m"] = distances
    nearby_df = df[df["dist_m"] <= radius_m].copy()
    nearby_df["Distance (ft)"] = (nearby_df["dist_m"] * 3.28084).round(0).astype("Int64")

    for _, r in nearby_df.iterrows():
        if np.isclose(float(r[lat_col]), lat) and np.isclose(float(r[lon_col]), lon):
            continue
        rc = COLOR.get(r.get(risk_col, ""), "gray")
        folium.CircleMarker(
            location=[r[lat_col], r[lon_col]],
            radius=5,
            color="white",
            weight=1,
            fill=True,
            fill_color=rc,
            fill_opacity=0.9,
            popup=f"{r.get(street_col,'')}<br>Risk: {r.get(risk_col,'')}"
        ).add_to(m)

    return m, nearby_df

# -------------------------
# Render map
# -------------------------
if st.session_state.selected is None:
    base_map = build_base_map()
    map_data = st_folium(base_map, width=900, height=600)
else:
    focused_map, nearby = build_focused_map_and_nearby(st.session_state.selected)
    st.session_state.nearby_df = nearby
    map_data = st_folium(focused_map, width=900, height=600)

# -------------------------
# Handle map click
# -------------------------
if map_data and map_data.get("last_clicked") is not None:
    click_lat = map_data["last_clicked"]["lat"]
    click_lon = map_data["last_clicked"]["lng"]
    distances = haversine_vec(click_lat, click_lon, df[lat_col].values, df[lon_col].values)
    nearest_idx = np.argmin(distances)
    st.session_state.selected = df.iloc[nearest_idx].to_dict()
    focused_map, nearby = build_focused_map_and_nearby(st.session_state.selected)
    st.session_state.nearby_df = nearby
    st_folium(focused_map, width=900, height=600)

# -------------------------
# Results table
# -------------------------
if st.session_state.selected is not None:
    nearby_df = st.session_state.nearby_df

    table_cols = [street_col, risk_col]
    if risk_score_col in nearby_df.columns:
        table_cols.append(risk_score_col)
    table_cols.append("Distance (ft)")
    for extra_col in [recent_insp_col, num_insp_col]:
        if extra_col and extra_col in nearby_df.columns:
            table_cols.append(extra_col)

    display_df = nearby_df[table_cols].copy()

    count_within = len(nearby_df)
    sel_street = st.session_state.selected.get(street_col, "")
    risk_val = st.session_state.selected.get(risk_col, "")
    risk_color = COLOR.get(risk_val, "gray")

    if recent_insp_col and recent_insp_col in nearby_df.columns:
        recent_dates = pd.to_datetime(nearby_df[recent_insp_col], errors='coerce')
        recent_dates = recent_dates.dropna()
        recent_date = recent_dates.max().strftime("%m/%d/%Y") if not recent_dates.empty else "N/A"
    else:
        recent_date = "N/A"

    st.markdown(
        f"<div style='background-color:{risk_color}; color:black; font-size:20px; padding:10px;'>{count_within} Addresses within {radius_toggle} ft of {sel_street} (Risk Level: {risk_val})</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='background-color:#f2f2f2; color:black; font-size:14px; padding:8px;'>Most Recent Termite Inspection Within {radius_toggle} ft: {recent_date}</div>",
        unsafe_allow_html=True
    )

    # -------------------------
    # Display table (view-only, non-sortable, no inline download)
    # -------------------------
    st.dataframe(display_df.astype(str), use_container_width=True, hide_index=True)

    # -------------------------
    # Download Termite Risk Index CSV
    # -------------------------
    if not display_df.empty:
        download_df = display_df.copy()
        header1 = f"{len(download_df)} Addresses within {radius_toggle} ft of {sel_street} (Risk Level: {risk_val})"
        header2 = f"Most Recent Termite Inspection Within {radius_toggle} ft: {recent_date}"
        run_date = f"Run Date: {datetime.now().strftime('%m/%d/%Y %H:%M:%S')}"

        output = io.StringIO()
        output.write(header1 + "\n")
        output.write(header2 + "\n")
        download_df.to_csv(output, index=False)
        output.write(run_date + "\n")
        csv_bytes = output.getvalue().encode("utf-8")

        file_name = f"termite_risk_index_{sel_street.replace(' ','_')}.csv"

        st.download_button(
            label="📥 Download Termite Risk Index CSV",
            data=csv_bytes,
            file_name=file_name,
            mime="text/csv"
        )
