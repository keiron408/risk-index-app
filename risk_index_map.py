import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np

# ============================================================
# APP CONFIG
# ============================================================
st.set_page_config(page_title="Termite Risk Index Viewer", layout="wide")
st.title("🏠 Termite Risk Index Viewer")

# ============================================================
# GLOBAL CSS: TOAST + LEGEND + MOBILE BACK-TO-TOP BUTTON
# ============================================================
st.markdown("""
<style>

@media (max-width: 600px) {
    h1 {font-size: 1.3rem !important;}
    .stRadio label, .stSelectbox label {font-size: 0.9rem !important;}
    .stDataFrame {font-size: 0.8rem !important;}
}

/* Toast */
#toast {
    visibility: hidden;
    min-width: 260px;
    background-color: #333;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 12px;
    position: fixed;
    z-index: 99999;
    left: 50%;
    bottom: 40px;
    transform: translateX(-50%);
    font-size: 14px;
}
#toast.show { visibility: visible; animation: fadein 0.5s, fadeout 0.5s 3s; }
@keyframes fadein { from {bottom:0; opacity:0;} to {bottom:40px; opacity:1;} }
@keyframes fadeout { from {bottom:40px; opacity:1;} to {bottom:0; opacity:0;} }

/* Legend container */
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

/* Scroll-to-top FLOATING BUTTON */
#scrollTopBtn {
    display: none;
    position: fixed;
    bottom: 90px;
    right: 16px;
    z-index: 9999;
    background-color: #333;
    color: white;
    padding: 12px 14px;
    border-radius: 50%;
    font-size: 18px;
    cursor: pointer;
}
#scrollTopBtn:hover {
    background-color: #555;
}
</style>

<!-- Toast -->
<div id="toast">⚠️</div>

<!-- Scroll-to-Top Button -->
<button onclick="window.scrollTo({top:0, behavior:'smooth'});" id="scrollTopBtn">↑</button>

<script>
// Toast function
function showToast(msg, bg) {
  var x = document.getElementById("toast");
  x.innerText = msg;
  if (bg) x.style.backgroundColor = bg;
  x.className = "show";
  setTimeout(() => x.className = x.className.replace("show",""), 4000);
}

// Show scroll-to-top button when user scrolls down
window.onscroll = function() {
    let btn = document.getElementById("scrollTopBtn");
    if (document.documentElement.scrollTop > 300) {
        btn.style.display = "block";
    } else {
        btn.style.display = "none";
    }
};
</script>
""", unsafe_allow_html=True)

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
    cols_l = [c.lower() for c in cols]
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols[cols_l.index(cand.lower())]
    for cand in candidates:
        for i, col in enumerate(cols):
            if col.lower().startswith(cand.lower()):
                return col
    return None

lat_col = find_col(df.columns, ["latitude", "lat"])
lon_col = find_col(df.columns, ["longitude", "lon", "lng"])
addr_col = find_col(df.columns, ["matched_address", "address", "full_address", "search address"])
street_col = find_col(df.columns, ["street", "street_name", "fulladdress"])
risk_col = find_col(df.columns, ["risk_level", "risk", "category"])
risk_score_col = find_col(df.columns, ["risk_score", "score"])
recent_insp_col = find_col(df.columns, ["most recent inspection", "most_recent_insp"])
num_insp_col = find_col(df.columns, ["# of inspections", "num_inspections", "total_inspections"])

if not street_col:
    street_col = addr_col

if not lat_col or not lon_col or not risk_col:
    st.error("CSV must contain latitude, longitude, and risk_level columns.")
    st.stop()

# ============================================================
# NORMALIZE RISK VALUES
# ============================================================
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
})

# ============================================================
# COLOR MAPPING
# ============================================================
COLOR = {
    "Very High": "#8B0000",
    "High":     "#FF0000",
    "Moderate": "#FFA500",
    "Low":      "#FFFF00"
}

# ============================================================
# SEARCH + CONTROLS
# ============================================================
st.markdown("### 🔍 Search Options")

@st.cache_data
def get_search_options(df_in, col):
    return sorted(df_in[col].dropna().unique())

colA, colB = st.columns([2, 1])

with colA:
    opts = get_search_options(df, addr_col)
    placeholder = "Enter address / select from map..."
    search_choice = st.selectbox(
        "Search",
        [placeholder] + opts,
        key="search_box",
        label_visibility="collapsed"
    )
    if search_choice == placeholder:
        search_choice = ""

with colB:
    radius_toggle = st.radio("Radius (ft)", [200, 300], horizontal=True)

radius_m = radius_toggle * 0.3048

# ============================================================
# DATA CLEANUP
# ============================================================
df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
df = df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

center_lat, center_lon = df[lat_col].mean(), df[lon_col].mean()

# ============================================================
# SESSION STATE
# ============================================================
st.session_state.setdefault("selected", None)
st.session_state.setdefault("nearby_df", pd.DataFrame())
st.session_state.setdefault("map_last_click", None)   # IMPORTANT FIX

# ============================================================
# DISTANCE FUNCTION
# ============================================================
def haversine_vec(lat0, lon0, lats, lons):
    R = 6371000.0
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad
    a = np.sin(dlat/2)**2 + np.cos(lat0_rad)*np.cos(lat_rad)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ============================================================
# SEARCH HANDLER (RESET MAP CLICK HISTORY)
# ============================================================
if search_choice:
    match = df[df[addr_col] == search_choice]
    if not match.empty:
        st.session_state.selected = match.iloc[0].to_dict()
        st.session_state.map_last_click = None   # <-- CRITICAL FIX

# ============================================================
# SAFE MAP CLICK HANDLER
# ============================================================
def handle_map_click(map_data):
    if not map_data:
        return
    click = map_data.get("last_clicked")
    if not click:
        return

    lat = click.get("lat")
    lon = click.get("lng")
    if lat is None or lon is None:
        return

    # If click is identical to last handled click → ignore
    last = st.session_state.map_last_click
    if last and abs(last["lat"] - lat) < 1e-9 and abs(last["lon"] - lon) < 1e-9:
        return

    st.session_state.map_last_click = {"lat": lat, "lon": lon}

    # Find nearest address
    distances = haversine_vec(lat, lon, df[lat_col], df[lon_col])
    nearest_idx = np.argmin(distances)
    st.session_state.selected = df.iloc[nearest_idx].to_dict()
    st.rerun()

# ============================================================
# MAP BUILDERS
# ============================================================
def build_base_map():
    return folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="https://mt1.google.com/vt/lyrs=y,h&x={x}&y={y}&z={z}",
        attr="Google"
    )

def build_focused_map_and_nearby(sel):
    lat = float(sel[lat_col])
    lon = float(sel[lon_col])
    risk_val = sel.get(risk_col, "")
    risk_color = COLOR.get(risk_val, "gray")

    m = folium.Map(
        location=[lat, lon],
        zoom_start=18,
        tiles="https://mt1.google.com/vt/lyrs=y,h&x={x}&y={y}&z={z}",
        attr="Google"
    )

    draw_radius_m = radius_m * (1.25 if radius_toggle == 200 else 1.1667)
    folium.Circle(
        (lat, lon),
        draw_radius_m,
        color="blue",
        fill=False,
        weight=2
    ).add_to(m)

    temp = df.copy()
    temp["dist_m"] = haversine_vec(lat, lon, temp[lat_col], temp[lon_col])
    near = temp[temp["dist_m"] <= radius_m].copy()

    if near.empty:
        folium.Marker(
            (lat, lon),
            icon=folium.DivIcon(
                html=f"""
                <div style="background:{risk_color};
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
        return m, near

    near["Distance (ft)"] = (near["dist_m"] * 3.28084).round(0).astype("Int64")
    near = near.sort_values("dist_m").reset_index(drop=True)

    for _, r in near.iterrows():
        c = COLOR.get(r.get(risk_col, ""), "gray")
        folium.PolyLine(
            [(lat, lon), (r[lat_col], r[lon_col])],
            color=c,
            weight=1.2,
            opacity=0.4
        ).add_to(m)
        folium.CircleMarker(
            (r[lat_col], r[lon_col]],
            radius=6,
            color="white",
            weight=1,
            fill=True,
            fill_color=c,
            fill_opacity=0.95
        ).add_to(m)

    folium.Marker(
        (lat, lon),
        icon=folium.DivIcon(
            html=f"""
            <div style="background:{risk_color};
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

    return m, near

# ============================================================
# RESPONSIVE MAP SIZE
# ============================================================
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
# INITIAL LOAD — FULL-WIDTH MAP
# ============================================================
if st.session_state.selected is None:
    m = build_base_map()
    map_data = st_folium(m, width=map_width, height=map_height, use_container_width=True)

    handle_map_click(map_data)

    # Legend
    st.markdown("""
    <div class="legend-container">
        <div class="legend-item"><div class="legend-box" style="background:#8B0000;"></div> Very High</div>
        <div class="legend-item"><div class="legend-box" style="background:#FF0000;"></div> High</div>
        <div class="legend-item"><div class="legend-box" style="background:#FFA500;"></div> Moderate</div>
        <div class="legend-item"><div class="legend-box" style="background:#FFFF00;"></div> Low</div>
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ============================================================
# SPLIT LAYOUT — MAP + TABLE
# ============================================================
map_col, table_col = st.columns([1.3, 1])

# -------------------------
# MAP
# -------------------------
with map_col:
    m, nearby_df = build_focused_map_and_nearby(st.session_state.selected)
    st.session_state.nearby_df = nearby_df

    map_data = st_folium(m, width=map_width, height=map_height, use_container_width=True)

    handle_map_click(map_data)

    # Legend
    st.markdown("""
    <div class="legend-container">
        <div class="legend-item"><div class="legend-box" style="background:#8B0000;"></div> Very High</div>
        <div class="legend-item"><div class="legend-box" style="background:#FF0000;"></div> High</div>
        <div class="legend-item"><div class="legend-box" style="background:#FFA500;"></div> Moderate</div>
        <div class="legend-item"><div class="legend-box" style="background:#FFFF00;"></div> Low</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# TABLE
# -------------------------
with table_col:
    nearby_df = st.session_state.nearby_df

    if nearby_df.empty:
        st.warning("No nearby addresses found within the selected radius.")
        st.stop()

    table_cols = [street_col, risk_col, "Distance (ft)"]
    if risk_score_col in nearby_df.columns:
        table_cols.insert(2, risk_score_col)
    if recent_insp_col in nearby_df.columns:
        table_cols.append(recent_insp_col)
    if num_insp_col in nearby_df.columns:
        table_cols.append(num_insp_col)

    df2 = nearby_df.copy()
    df2["_dist"] = pd.to_numeric(df2["Distance (ft)"], errors="coerce")
    df2 = df2.sort_values("_dist")
    display_df = df2[table_cols].fillna("")

    sel_street = str(st.session_state.selected.get(street_col, "")).strip()
    sel_risk = st.session_state.selected.get(risk_col, "")
    risk_color = COLOR.get(sel_risk, "gray")
    header_text_color = "white" if sel_risk in ["High", "Very High"] else "black"

    st.markdown(
        f"""
        <div style="background:{risk_color};color:{header_text_color};
                    padding:8px;border-radius:6px;text-align:center;font-size:16px;">
            {len(display_df)} addresses within {radius_toggle} ft of {sel_street}
            (Risk: {sel_risk})
        </div>
        """,
        unsafe_allow_html=True
    )

    if recent_insp_col in df2.columns:
        dts = pd.to_datetime(df2[recent_insp_col], errors="coerce").dropna()
        recent_val = dts.max().strftime("%m/%d/%Y") if not dts.empty else "N/A"
    else:
        recent_val = "N/A"

    st.markdown(
        f"""
        <div style="background:#f3f3f3;color:black;
                    padding:6px;border-radius:6px;text-align:center;margin-bottom:8px;">
            Most recent inspection within {radius_toggle} ft: {recent_val}
        </div>
        """,
        unsafe_allow_html=True
    )

    def lighten(hex_color, factor=0.82):
        hex_color = hex_color.lstrip("#")
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r += int((255 - r) * factor)
        g += int((255 - g) * factor)
        b += int((255 - b) * factor)
        return f"rgb({r},{g},{b})"

    def highlight_rows(row):
        street = str(row.get(street_col, "")).strip()
        level = row.get(risk_col, "")
        base = COLOR.get(level, "#CCCCCC")

        if street == sel_street:
            txt = "white" if level in ["High", "Very High"] else "black"
            return [f"background-color:{base};color:{txt};font-weight:bold;"] * len(row)

        return [f"background-color:{lighten(base)};color:black;"] * len(row)

    styled_df = (
        display_df.style
        .apply(highlight_rows, axis=1)
        .set_table_styles(
            [{
                "selector": "thead th",
                "props": [
                    ("background-color", risk_color),
                    ("color", header_text_color),
                    ("font-weight", "bold")
                ]
            }]
        )
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=550)
