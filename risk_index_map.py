import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np

# -------------------------
# App Configuration
# -------------------------
st.set_page_config(page_title="Termite Risk Index Viewer", layout="wide")
st.title("🏠 Termite Risk Index Viewer")

# -------------------------
# Global Style + Toast + Legend
# -------------------------
st.markdown("""
<style>
@media (max-width: 600px) {
    h1 {font-size: 1.3rem !important;}
    .stRadio label, .stSelectbox label {font-size: 0.9rem !important;}
    .stDataFrame {font-size: 0.8rem !important;}
}

/* Toast Notification */
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
#toast.show { visibility: visible; animation: fadein 0.5s, fadeout 0.5s 3s; }
@keyframes fadein { from {bottom: 0; opacity: 0;} to {bottom: 40px; opacity: 1;} }
@keyframes fadeout { from {bottom: 40px; opacity: 1;} to {bottom: 0; opacity: 0;} }

/* Risk Legend Layout */
.legend-container {
    margin-top: 10px;
    display: flex;
    justify-content: center;
    gap: 14px;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
}
.legend-box {
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 1px solid #555;
}

/* Mobile: stack legend vertically */
@media (max-width: 768px) {
    .legend-container {
        flex-direction: column;
        align-items: flex-start;
        margin-left: 6px;
    }
}
</style>

<div id="toast">⚠️</div>

<script>
function showToast(msg, bg) {
  var x = document.getElementById("toast");
  x.innerText = msg;
  if (bg) x.style.backgroundColor = bg;
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
    return pd.read_csv("master_with_inspection_counts_202512.csv")

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
addr_col = find_col(df.columns, ["matched_address", "address", "full_address", "search address"])
street_col = find_col(df.columns, ["street", "street_name", "fulladdress"])
risk_col = find_col(df.columns, ["risk_level", "risk", "category"])
risk_score_col = find_col(df.columns, ["risk_score", "score"])
recent_insp_col = find_col(df.columns, ["most recent inspection", "most_recent_insp"])
num_insp_col = find_col(df.columns, ["# of inspections", "num_inspections", "total_inspections"])

if not lat_col or not lon_col or not risk_col:
    st.error("CSV must contain latitude, longitude, and risk_level columns.")
    st.stop()

# -------------------------
# Normalize risk values (FULL FIX)
# -------------------------
df[risk_col] = (
    df[risk_col]
    .astype(str)
    .str.strip()
    .str.replace("_", " ", regex=False)
    .str.replace("-", " ", regex=False)
    .str.title()
)

df[risk_col] = df[risk_col].replace({
    "High Risk": "High",
    "Moderate Risk": "Moderate",
    "Low Risk": "Low",
    "Very High Risk": "Very High",
    "Veryhigh": "Very High",
    "Very Highrisk": "Very High",
})

# -------------------------
# Color Mapping
# -------------------------
COLOR = {
    "Very High": "#8B0000",
    "High": "#FF0000",
    "Moderate": "#FFA500",
    "Low": "#FFFF00"
}

# -------------------------
# Search Options
# -------------------------
st.markdown("### 🔍 Search Options")

@st.cache_data
def get_search_options(df_in, addr_column):
    return sorted(df_in[addr_column].dropna().unique())

col1, col2 = st.columns([2, 1])

with col1:
    search_opts = get_search_options(df, addr_col)
    search_placeholder = "Enter address / select from map..."
    search_choice = st.selectbox(
        "Search Address",
        [search_placeholder] + search_opts,
        key="search_box",
        label_visibility="collapsed"
    )
    if search_choice == search_placeholder:
        search_choice = ""

with col2:
    radius_toggle = st.radio("Radius (ft)", [200, 300], horizontal=True)

radius_m = radius_toggle * 0.3048

# -------------------------
# Data cleanup
# -------------------------
df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
df = df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

center_lat, center_lon = df[lat_col].mean(), df[lon_col].mean()

# -------------------------
# Session State
# -------------------------
st.session_state.setdefault("selected", None)
st.session_state.setdefault("nearby_df", pd.DataFrame())

# -------------------------
# Distance Function
# -------------------------
def haversine_vec(lat0, lon0, lats, lons):
    R = 6371000.0
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat0_rad) * np.cos(lat_rad) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# -------------------------
# Search selection
# -------------------------
if search_choice:
    sel_row = df[df[addr_col] == search_choice]
    if not sel_row.empty:
        st.session_state.selected = sel_row.iloc[0].to_dict()

# -------------------------
# Map Builders
# -------------------------
def build_base_map():
    return folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="https://mt1.google.com/vt/lyrs=y,h&x={x}&y={y}&z={z}",
        attr="Google"
    )

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

    draw_radius_m = radius_m * (1.25 if radius_toggle == 200 else 1.1667)
    folium.Circle(
        location=[lat, lon],
        radius=draw_radius_m,
        color="blue",
        fill=False,
        weight=2
    ).add_to(m)

    temp_df = df.copy()
    temp_df["dist_m"] = haversine_vec(lat, lon, temp_df[lat_col], temp_df[lon_col])
    nearby_df = temp_df[temp_df["dist_m"] <= radius_m].copy()

    if nearby_df.empty:
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f"""
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
            )
        ).add_to(m)
        return m, nearby_df

    nearby_df["Distance (ft)"] = (nearby_df["dist_m"] * 3.28084).round(0).astype("Int64")
    nearby_df = nearby_df.sort_values("dist_m").reset_index(drop=True)

    for _, r in nearby_df.iterrows():
        rc = COLOR.get(r.get(risk_col, ""), "gray")

        folium.PolyLine(
            [(lat, lon), (r[lat_col], r[lon_col])],
            color=rc,
            weight=1.2,
            opacity=0.4
        ).add_to(m)

        folium.CircleMarker(
            location=[r[lat_col], r[lon_col]],
            radius=6,
            color="white",
            weight=1,
            fill=True,
            fill_color=rc,
            fill_opacity=0.95
        ).add_to(m)

    # Pulsating center marker
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(
            html=f"""
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
        )
    ).add_to(m)

    return m, nearby_df

# -------------------------
# Responsive Map Dimensions
# -------------------------
def get_map_dimensions():
    try:
        ua = st.runtime.scriptrunner.script_run_context.session_info.user_agent
        if "Mobile" in ua:
            return 360, 420
        elif "Tablet" in ua:
            return 720, 520
    except:
        pass
    return 1000, 600

map_width, map_height = get_map_dimensions()

# ============================================================
# 🚨 NEW LAYOUT LOGIC HERE
# ============================================================

# CASE 1 — BEFORE SELECTION: SHOW FULL-WIDTH MAP
if st.session_state.selected is None:

    m = build_base_map()
    map_data = st_folium(m, width=map_width, height=map_height, use_container_width=True)

    # Allow clicking to select nearest address
    if map_data and map_data.get("last_clicked") is not None:
        click_lat = map_data["last_clicked"]["lat"]
        click_lon = map_data["last_clicked"]["lng"]

        distances = haversine_vec(click_lat, click_lon, df[lat_col], df[lon_col])
        nearest_idx = np.argmin(distances)
        st.session_state.selected = df.iloc[nearest_idx].to_dict()

        st.experimental_rerun()

    st.stop()

# -------------------------
# CASE 2 — AFTER SELECTION → SPLIT LAYOUT
# -------------------------
map_col, table_col = st.columns([1.3, 1])

# -------------------------
# LEFT COLUMN — MAP
# -------------------------
with map_col:

    m, nearby = build_focused_map_and_nearby(st.session_state.selected)
    st.session_state.nearby_df = nearby

    map_data = st_folium(m, width=map_width, height=map_height, use_container_width=True)

    # Always select nearest address on click
    if map_data and map_data.get("last_clicked") is not None:

        click_lat = map_data["last_clicked"]["lat"]
        click_lon = map_data["last_clicked"]["lng"]

        distances = haversine_vec(click_lat, click_lon, df[lat_col], df[lon_col])
        nearest_idx = np.argmin(distances)

        st.session_state.selected = df.iloc[nearest_idx].to_dict()
        st.experimental_rerun()

    # RISK LEGEND BELOW MAP
    st.markdown("""
    <div class="legend-container">
        <div class="legend-item"><div class="legend-box" style="background:#8B0000;"></div> Very High</div>
        <div class="legend-item"><div class="legend-box" style="background:#FF0000;"></div> High</div>
        <div class="legend-item"><div class="legend-box" style="background:#FFA500;"></div> Moderate</div>
        <div class="legend-item"><div class="legend-box" style="background:#FFFF00;"></div> Low</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# RIGHT COLUMN — TABLE
# -------------------------
with table_col:

    nearby_df = st.session_state.nearby_df

    if nearby_df.empty:
        st.warning("No nearby addresses found.")
        st.stop()

    table_cols = [street_col, risk_col, "Distance (ft)"]
    if risk_score_col in nearby_df.columns:
        table_cols.insert(2, risk_score_col)
    if recent_insp_col in nearby_df.columns:
        table_cols.append(recent_insp_col)
    if num_insp_col in nearby_df.columns:
        table_cols.append(num_insp_col)

    sort_df = nearby_df.copy()
    sort_df["_dist"] = pd.to_numeric(sort_df["Distance (ft)"], errors="coerce")
    sort_df = sort_df.sort_values("_dist")

    display_df = sort_df[table_cols].copy().fillna("")

    sel_street = st.session_state.selected.get(street_col, "")
    sel_risk = st.session_state.selected.get(risk_col, "")
    risk_color = COLOR.get(sel_risk, "gray")
    text_color = "white" if sel_risk in ["High", "Very High"] else "black"

    # Header Banner
    st.markdown(
        f"""
        <div style="background:{risk_color};color:{text_color};
                    padding:8px;border-radius:6px;
                    text-align:center;font-size:16px;">
            {len(display_df)} addresses within {radius_toggle} ft of {sel_street}
            (Risk: {sel_risk})
        </div>
        """,
        unsafe_allow_html=True
    )

    # Most recent inspection
    if recent_insp_col in sort_df.columns:
        recent_vals = pd.to_datetime(sort_df[recent_insp_col], errors='coerce').dropna()
        max_date = recent_vals.max().strftime("%m/%d/%Y") if not recent_vals.empty else "N/A"
    else:
        max_date = "N/A"

    st.markdown(
        f"""
        <div style="background:#f3f3f3;color:black;padding:6px;
                    border-radius:6px;text-align:center;
                    margin-bottom:8px;">
            Most recent inspection: {max_date}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Row Highlighting
    selected_street_val = sel_street.strip()

    def lighten(hex_color, factor=0.82):
        hex_color = hex_color.lstrip("#")
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        return f"rgb({r},{g},{b})"

    def highlight(row):
        street = str(row.get(street_col, "")).strip()
        level = row.get(risk_col, "")
        base = COLOR.get(level, "#CCC")

        if street == selected_street_val:
            txt = "white" if level in ["High", "Very High"] else "black"
            return [f"background:{base};color:{txt};font-weight:bold;"] * len(row)

        return [f"background:{lighten(base)};color:black;"] * len(row)

    styled_df = (
        display_df.style
        .apply(highlight, axis=1)
        .set_table_styles(
            [{
                "selector": "thead th",
                "props": [("background-color", risk_color),
                          ("color", text_color),
                          ("font-weight", "bold")]
            }]
        )
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=550)
