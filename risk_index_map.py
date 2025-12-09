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

/* Scroll-to-top */
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
# COLUMN DETECTION
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

if not street_col:
    street_col = addr_col

# Normalize risk
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
    "Very High Risk": "Very High",
})

COLOR = {
    "Very High": "#8B0000",
    "High": "#FF0000",
    "Moderate": "#FFA500",
    "Low": "#FFFF00",
}

# Clean and center
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
st.session_state.setdefault("pending_map_click", False)   # ⭐ critical fix

# ============================================================
# DISTANCE
# ============================================================
def haversine_vec(lat0, lon0, lats, lons):
    R = 6371000.0
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)
    lat = np.radians(lats)
    lon = np.radians(lons)
    dlat = lat - lat0
    dlon = lon - lon0
    a = np.sin(dlat/2)**2 + np.cos(lat0)*np.cos(lat)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ============================================================
# SEARCH UI
# ============================================================
st.markdown("### 🔍 Search Options")

@st.cache_data
def get_search_options(df, col):
    return sorted(df[col].dropna().unique())

colA, colB = st.columns([2,1])
with colA:
    addresses = get_search_options(df, addr_col)
    placeholder = "Enter address / select from map..."
    search_choice = st.selectbox(
        "Search", [placeholder] + addresses,
        key="search_box", label_visibility="collapsed"
    )
    if search_choice == placeholder:
        search_choice = ""

with colB:
    radius_ft = st.radio("Radius (ft)", [200, 300], horizontal=True)

radius_m = radius_ft * 0.3048

# Handle search
if search_choice:
    matched = df[df[addr_col] == search_choice]
    if not matched.empty:
        st.session_state.selected = matched.iloc[0].to_dict()
        st.session_state.map_last_click = None
        st.session_state.pending_map_click = True     # ⭐ ignore first click
        

# ============================================================
# MAP BUILD
# ============================================================
def build_base_map():
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="https://mt1.google.com/vt/lyrs=y,h",
        attr="Google"
    )
    m.add_child(folium.LatLngPopup())
    return m


def build_focused_map_and_nearby(selected):
    lat = float(selected[lat_col])
    lon = float(selected[lon_col])

    m = folium.Map(
        location=[lat, lon],
        zoom_start=18,
        tiles="https://mt1.google.com/vt/lyrs=y,h",
        attr="Google"
    )
    m.add_child(folium.LatLngPopup())

    temp = df.copy()
    temp["dist_m"] = haversine_vec(lat, lon, temp[lat_col], temp[lon_col])
    nearby = temp[temp["dist_m"] <= radius_m].copy()
    nearby["Distance (ft)"] = (nearby["dist_m"] * 3.28084).round(0).astype("Int64")
    nearby = nearby.sort_values("dist_m").reset_index(drop=True)

    # Draw radius
    folium.Circle(
        (lat, lon),
        radius_m * 1.15, color="blue", fill=False, weight=2
    ).add_to(m)

    # Center marker
    center_color = COLOR.get(selected[risk_col], "gray")
    folium.CircleMarker(
        (lat, lon), radius=10, fill=True,
        color="black", fill_color=center_color, weight=2
    ).add_to(m)

    # Nearby markers + lines
    for _, r in nearby.iterrows():
        c = COLOR.get(r[risk_col], "#AAA")
        folium.PolyLine(
            [(lat, lon), (r[lat_col], r[lon_col])],
            color=c, weight=1.2, opacity=0.4
        ).add_to(m)
        folium.CircleMarker(
            (r[lat_col], r[lon_col]),
            radius=6, fill=True,
            color="white", fill_color=c, weight=1
        ).add_to(m)

    return m, nearby

# ============================================================
# CLICK HANDLER (pending_map_click integrated)
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

    # ⭐ Ignore the first click after search (critical fix)
    if st.session_state.pending_map_click:
        st.session_state.pending_map_click = False
        st.session_state.map_last_click = {"lat": lat, "lon": lon}
        return

    last = st.session_state.map_last_click
    if isinstance(last, dict) and last.get("lat") is not None:
        if abs(last["lat"] - lat) < 1e-9 and abs(last["lon"] - lon) < 1e-9:
            return

    st.session_state.map_last_click = {"lat": lat, "lon": lon}

    distances = haversine_vec(lat, lon, df[lat_col], df[lon_col])
    nearest_idx = int(np.argmin(distances))
    st.session_state.selected = df.iloc[nearest_idx].to_dict()
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
# INITIAL MAP (NO SELECTION)
# ============================================================
if st.session_state.selected is None:
    m = build_base_map()
    map_data = st_folium(
        m,
        height=600,
        use_container_width=True,
        returned_objects=["last_clicked"]
    )
    handle_map_click(map_data)
    legend()
    st.stop()

# ============================================================
# SPLIT: MAP + TABLE
# ============================================================
map_col, table_col = st.columns([1.3,1])

with map_col:
    m, nearby = build_focused_map_and_nearby(st.session_state.selected)
    st.session_state.nearby_df = nearby
    map_data = st_folium(
        m,
        height=600,
        use_container_width=True,
        returned_objects=["last_clicked"]
    )
    handle_map_click(map_data)
    legend()

with table_col:
    df2 = st.session_state.nearby_df
    if df2 is None or df2.empty:
        st.warning("No nearby addresses.")
        st.stop()

    table_cols = [street_col, risk_col, "Distance (ft)"]
    if risk_score_col in df2.columns: table_cols.insert(2, risk_score_col)
    if recent_insp_col in df2.columns: table_cols.append(recent_insp_col)
    if num_insp_col in df2.columns: table_cols.append(num_insp_col)

    df2 = df2.sort_values("Distance (ft)")

    sel_street = st.session_state.selected.get(street_col, "")
    sel_risk = st.session_state.selected.get(risk_col, "")
    risk_color = COLOR.get(sel_risk, "gray")
    text_color = "white" if sel_risk in ["High","Very High"] else "black"

    st.markdown(
        f"""
        <div style="background:{risk_color};color:{text_color};
                    padding:8px;border-radius:6px;text-align:center;">
            {len(df2)} addresses within {radius_ft} ft of {sel_street}
            (Risk: {sel_risk})
        </div>
        """,
        unsafe_allow_html=True
    )

    if recent_insp_col in df2.columns:
        d = pd.to_datetime(df2[recent_insp_col], errors="coerce").dropna()
        recent_val = d.max().strftime("%m/%d/%Y") if not d.empty else "N/A"
        st.markdown(
            f"""
            <div style="background:#eee;padding:6px;border-radius:6px;
                        text-align:center;margin-bottom:8px;">
                Most recent inspection: {recent_val}
            </div>
            """,
            unsafe_allow_html=True
        )

    def lighten(hex_color, factor=0.82):
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i+2], 16) for i in (0,2,4))
        r = int(r + (255 - r)*factor)
        g = int(g + (255 - g)*factor)
        b = int(b + (255 - b)*factor)
        return f"rgb({r},{g},{b})"

    def highlight_rows(row):
        street = str(row.get(street_col, ""))
        level = row.get(risk_col, "")
        base = COLOR.get(level, "#DDD")
        if street == sel_street:
            txt = "white" if level in ["High", "Very High"] else "black"
            return [f"background-color:{base};color:{txt};font-weight:bold;"] * len(row)
        return [f"background-color:{lighten(base)};color:black;"] * len(row)

    styled = (
        df2[table_cols].style
        .apply(highlight_rows, axis=1)
        .set_table_styles([{
            "selector": "thead th",
            "props": [
                ("background-color", risk_color),
                ("color", text_color),
                ("font-weight", "bold")
            ]
        }])
    )

    st.dataframe(styled, use_container_width=True, hide_index=True, height=550)
