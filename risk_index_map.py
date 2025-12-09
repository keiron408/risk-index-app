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
# CSS (Legend + scroll button)
# ============================================================
st.markdown("""
<style>
@media (max-width: 600px) {
    h1 {font-size: 1.3rem !important;}
    .stRadio label, .stSelectbox label {font-size: 0.9rem !important;}
    .stDataFrame {font-size: 0.8rem !important;}
}

.legend-container {
    margin-top: 10px;
    display: flex;
    justify-content: center;
    gap: 14px;
}
.legend-item {display: flex;align-items: center;gap: 6px;font-size: 13px;}
.legend-box {width: 18px;height: 18px;border-radius: 3px;border: 1px solid #555;}

@media (max-width: 768px) {
    .legend-container {flex-direction: column;align-items: flex-start;margin-left: 6px;}
}

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
#scrollTopBtn:hover {background-color: #555;}
</style>

<button onclick="window.scrollTo({top:0, behavior:'smooth'});" id="scrollTopBtn">↑</button>

<script>
window.onscroll = function() {
    let btn = document.getElementById("scrollTopBtn");
    if (document.documentElement.scrollTop > 300) btn.style.display = "block";
    else btn.style.display = "none";
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
    lower = [c.lower() for c in cols]
    for cand in candidates:
        if cand.lower() in lower:
            return cols[lower.index(cand.lower())]
    for cand in candidates:
        for col in cols:
            if col.lower().startswith(cand.lower()):
                return col
    return None

lat_col = find_col(df.columns, ["latitude", "lat"])
lon_col = find_col(df.columns, ["longitude", "lon", "lng"])
addr_col = find_col(df.columns, ["matched_address", "address", "full_address"])
street_col = find_col(df.columns, ["street", "street_name", "fulladdress"])
risk_col = find_col(df.columns, ["risk_level", "risk"])
risk_score_col = find_col(df.columns, ["risk_score"])
recent_insp_col = find_col(df.columns, ["most recent inspection"])
num_insp_col = find_col(df.columns, ["# of inspections"])
search_col = find_col(df.columns, ["search address", "search_address"])

if not street_col:
    street_col = addr_col

# ============================================================
# RISK NORMALIZATION (UI FIX)
# ============================================================
df[risk_col] = (
    df[risk_col]
    .astype(str)
    .str.strip()
    .str.replace("_", " ", regex=False)
    .str.replace("-", " ", regex=False)
    .str.upper()
)

df[risk_col] = df[risk_col].replace({
    "VERY HIGH": "Very High",
    "HIGH": "High",
    "MODERATE": "Moderate",
    "LOW": "Low",
})

COLOR = {
    "Very High": "#8B0000",
    "High": "#FF0000",
    "Moderate": "#FFA500",
    "Low": "#FFFF00",
}

# ============================================================
# CLEAN LAT/LON
# ============================================================
df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
df = df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

center_lat = df[lat_col].mean()
center_lon = df[lon_col].mean()

# ============================================================
# SESSION STATE
# ============================================================
st.session_state.setdefault("selected", None)
st.session_state.setdefault("nearby_df", pd.DataFrame())
st.session_state.setdefault("map_last_click", None)
st.session_state.setdefault("pending_first_click", False)

# ============================================================
# DISTANCE FUNCTION
# ============================================================
def haversine_vec(lat0, lon0, lats, lons):
    R = 6371000.0
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)
    lat = np.radians(lats)
    lon = np.radians(lons)
    a = np.sin((lat - lat0)/2)**2 + np.cos(lat0)*np.cos(lat)*np.sin((lon - lon0)/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ============================================================
# SEARCH UI
# ============================================================
st.markdown("### 🔍 Search Options")

@st.cache_data
def get_search_options(df, col):
    return sorted(df[col].dropna().unique())

colA, colB = st.columns([2, 1])

with colA:
    opts = get_search_options(df, search_col if search_col else addr_col)
    placeholder = "Enter address / select from map..."
    search_choice = st.selectbox(
        "Search",
        [placeholder] + opts,
        key="search_box",
        label_visibility="collapsed",
    )
    if search_choice == placeholder:
        search_choice = ""

with colB:
    radius_ft = st.radio("Radius (ft)", [200, 300], horizontal=True)

radius_m = radius_ft * 0.3048

# Handle search selection
if search_choice:
    match = df[df[search_col] == search_choice] if search_col else df[df[addr_col] == search_choice]

    if not match.empty:
        st.session_state.selected = match.iloc[0].to_dict()
        st.session_state.map_last_click = None
        st.session_state.pending_first_click = True  # FIX flashing after search

# ============================================================
# MAP BUILDERS (unchanged)
# ============================================================
def build_base_map():
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="https://mt1.google.com/vt/lyrs=y,h&x={x}&y={y}&z={z}",
        attr="Google",
    )
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
        attr="Google",
    )
    m.add_child(folium.LatLngPopup())

    draw_radius_m = radius_m * 1.20
    folium.Circle(
        (lat, lon),
        draw_radius_m,
        color="blue",
        fill=False,
        weight=2,
    ).add_to(m)

    temp = df.copy()
    temp["dist_m"] = haversine_vec(lat, lon, temp[lat_col], temp[lon_col])
    near = temp[temp["dist_m"] <= radius_m].copy()

    if near.empty:
        return m, near

    near["Distance (ft)"] = (near["dist_m"] * 3.28084).round(0).astype("Int64")
    near = near.sort_values("dist_m").reset_index(drop=True)

    for _, r in near.iterrows():
        c = COLOR.get(r.get(risk_col, ""), "gray")
        folium.CircleMarker(
            (r[lat_col], r[lon_col]),
            radius=6,
            color="white",
            weight=1,
            fill=True,
            fill_color=c,
            fill_opacity=0.95,
        ).add_to(m)

    # Center marker
    folium.CircleMarker(
        (lat, lon),
        radius=10,
        color="black",
        weight=2,
        fill=True,
        fill_color=risk_color,
        fill_opacity=1,
    ).add_to(m)

    return m, near

# ============================================================
# CLICK HANDLER (with first-click-after-search fix)
# ============================================================
def handle_map_click(map_data):
    if not isinstance(map_data, dict):
        return

    click = map_data.get("last_clicked")
    if not click:
        return

    lat = click.get("lat")
    lon = click.get("lng")
    if lat is None or lon is None:
        return

    # FIRST CLICK AFTER SEARCH → ignore (UI improvement)
    if st.session_state.pending_first_click:
        st.session_state.pending_first_click = False
        st.session_state.map_last_click = {"lat": lat, "lon": lon}
        return

    # Duplicate click guard
    last = st.session_state.map_last_click
    if isinstance(last, dict):
        if abs(last["lat"] - lat) < 1e-9 and abs(last["lon"] - lon) < 1e-9:
            return

    st.session_state.map_last_click = {"lat": lat, "lon": lon}

    # Snap to nearest parcel
    d = haversine_vec(lat, lon, df[lat_col], df[lon_col])
    idx = int(np.argmin(d))
    st.session_state.selected = df.iloc[idx].to_dict()

    st.rerun()

# ============================================================
# LEGEND
# ============================================================
def legend():
    st.markdown("""
    <div class="legend-container">
        <div class="legend-item"><div class="legend-box" style="background:#8B0000;"></div> Very High</div>
        <div class="legend-item"><div class="legend-box" style="background:#FF0000;"></div> High</div>
        <div class="legend-item"><div class="legend-box" style="background:#FFA500;"></div> Moderate</div>
        <div class="legend-item"><div class="legend-box" style="background:#FFFF00;"></div> Low</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# NO SELECTION → INITIAL MAP
# ============================================================
if st.session_state.selected is None:
    m = build_base_map()
    map_data = st_folium(
        m,
        height=600,
        use_container_width=True
    )
    handle_map_click(map_data)
    legend()
    st.stop()

# ============================================================
# MAP + TABLE LAYOUT
# ============================================================
map_col, table_col = st.columns([1.3, 1])

with map_col:
    m, near = build_focused_map_and_nearby(st.session_state.selected)
    st.session_state.nearby_df = near

    map_data = st_folium(
        m,
        height=600,
        use_container_width=True
    )
    handle_map_click(map_data)
    legend()

with table_col:
    df2 = st.session_state.nearby_df

    if df2.empty:
        st.warning("No nearby addresses.")
        st.stop()

    table_cols = [street_col, risk_col, "Distance (ft)"]
    if risk_score_col in df2.columns: table_cols.insert(2, risk_score_col)
    if recent_insp_col in df2.columns: table_cols.append(recent_insp_col)
    if num_insp_col in df2.columns: table_cols.append(num_insp_col)

    df2 = df2[table_cols].fillna("")

    # --- FORCE HEADER ROW TO APPEAR ---
    df2 = df2.rename(columns={col: col for col in df2.columns})

    # ============================================================
    # ROW COLORING (selected + lightened for nearby rows)
    # ============================================================
    def lighten(hex_color, factor=0.82):
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i+2], 16) for i in (0,2,4))
        r = int(r + (255 - r)*factor)
        g = int(g + (255 - g)*factor)
        b = int(b + (255 - b)*factor)
        return f"rgb({r},{g},{b})"

    sel_addr = st.session_state.selected.get(street_col, "")

    def highlight_rows(row):
        addr = str(row.get(street_col, ""))
        level = row.get(risk_col, "")
        base = COLOR.get(level, "#DDD")

        if addr == sel_addr:
            txt = "white" if level in ["High","Very High"] else "black"
            return [f"background-color:{base};color:{txt};font-weight:bold;"] * len(row)

        return [f"background-color:{lighten(base)};color:black;"] * len(row)

    styled = (
        df2.style
        .apply(highlight_rows, axis=1)
        .set_table_styles([{
            "selector": "thead th",
            "props": [("background-color", "#444"), ("color", "white"), ("font-weight", "bold")]
        }])
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=550,
        column_config={col: st.column_config.Column(col) for col in df2.columns}
    )
