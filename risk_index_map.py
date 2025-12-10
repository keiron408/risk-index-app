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
# CSS (legend + scroll button)
# ============================================================
st.markdown("""
<style>
.legend-container {
    margin-top: 10px;
    display: flex;
    justify-content: center;
    gap: 12px;
}
.legend-item {display: flex;align-items: center;gap: 6px;font-size: 13px;}
.legend-box {width: 18px;height: 18px;border-radius: 4px;border: 1px solid #444;}

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
def load_file():
    return pd.read_csv("master_with_inspection_counts_202512.csv")

df = load_file()

# ============================================================
# COLUMN DEFINITIONS (based on your screenshot)
# ============================================================
lat_col = "Latitude"
lon_col = "Longitude"
addr_col = "FullAddress"
search_col = "search address"
risk_col = "risk_level"
risk_score_col = "risk_score"
recent_insp_col = "most recent inspection"
num_insp_col = "# of inspections"

# ============================================================
# NORMALIZE RISK LEVELS
# ============================================================
df[risk_col] = (
    df[risk_col]
    .astype(str)
    .str.strip()
    .str.upper()
    .replace({
        "VERY HIGH": "Very High",
        "HIGH": "High",
        "MODERATE": "Moderate",
        "LOW": "Low",
    })
)

COLOR = {
    "Very High": "#8B0000",
    "High": "#FF0000",
    "Moderate": "#FFA500",
    "Low": "#FFFF00",
}

# Clean lat/lon
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

# ============================================================
# DISTANCE FUNCTION
# ============================================================
def haversine_vec(lat0, lon0, lats, lons):
    R = 6371000
    lat0, lon0 = np.radians([lat0, lon0])
    lat, lon = np.radians(lats), np.radians(lons)
    a = np.sin((lat-lat0)/2)**2 + np.cos(lat0)*np.cos(lat)*np.sin((lon-lon0)/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ============================================================
# SEARCH BAR
# ============================================================
st.markdown("### 🔍 Search")

@st.cache_data
def get_opts():
    return sorted(df[search_col].dropna().unique())

colA, colB = st.columns([2,1])

with colA:
    opts = get_opts()
    placeholder = "Type or select address…"
    choice = st.selectbox("Search", [placeholder] + opts, label_visibility="collapsed")
    search_choice = None if choice == placeholder else choice

with colB:
    radius_ft = st.radio("Radius (ft)", [200,300], horizontal=True)

radius_m = radius_ft * 0.3048

# APPLY SEARCH SELECTION
if search_choice:
    row = df[df[search_col] == search_choice]
    if not row.empty:
        st.session_state.selected = row.iloc[0].to_dict()
        st.session_state.map_last_click = None

# ============================================================
# MAP BUILDERS
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

def build_focused_map(selected):
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

    folium.Circle(
        (lat, lon), radius_m*1.20, color="blue", fill=False, weight=2
    ).add_to(m)

    temp = df.copy()
    temp["dist_m"] = haversine_vec(lat, lon, temp[lat_col], temp[lon_col])
    near = temp[temp["dist_m"] <= radius_m].copy().reset_index(drop=True)

    if near.empty:
        return m, near

    near["Distance (ft)"] = (near["dist_m"] * 3.28084).round(0)
    near = near.sort_values("dist_m")

    # Radial dashed lines
    for _, r in near.iterrows():
        rc = COLOR.get(r[risk_col], "gray")
        folium.PolyLine(
            [(lat,lon),(r[lat_col],r[lon_col])],
            color=rc,
            weight=1.3,
            opacity=0.5,
            dash_array="4,4"
        ).add_to(m)

    # Nearby markers
    for _, r in near.iterrows():
        folium.CircleMarker(
            (r[lat_col], r[lon_col]),
            radius=6,
            color="white",
            fill=True,
            fill_color=COLOR.get(r[risk_col], "gray"),
            fill_opacity=0.97,
            weight=1,
        ).add_to(m)

    # Center marker
    folium.CircleMarker(
        (lat,lon),
        radius=10,
        color="black",
        fill=True,
        fill_color=risk_color,
        weight=2,
        fill_opacity=1
    ).add_to(m)

    return m, near

# ============================================================
# CLICK HANDLER
# ============================================================
def handle_click(md):
    if not isinstance(md, dict): return
    click = md.get("last_clicked")
    if not click: return

    lat, lon = click.get("lat"), click.get("lng")
    if lat is None or lon is None: return

    # prevent duplicate ghost click
    last = st.session_state.map_last_click
    if last and abs(last["lat"]-lat)<1e-9 and abs(last["lng"]-lon)<1e-9:
        return

    st.session_state.map_last_click = {"lat":lat,"lng":lon}

    # snap to nearest parcel
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
# SINGLE st_folium() LOCATION (critical fix)
# ============================================================
if st.session_state.selected is None:
    # FULL-WIDTH MAP on initial load
    m = build_base_map()
   
    map_data = st_folium(m, height=600, use_container_width=True, key="mainmap")
    legend()

else:
    # MAP + TABLE LAYOUT after selection
    map_col, table_col = st.columns([1.3,1])

    with map_col:
        m, near = build_focused_map(st.session_state.selected)
        st.session_state.nearby_df = near

        map_data = st_folium(m, height=600, use_container_width=True, key="mainmap")
        legend()

    with table_col:
        near = st.session_state.nearby_df
        if near.empty:
            st.warning("No nearby addresses.")
            st.stop()

        sel_addr = st.session_state.selected.get(addr_col, "")

        # Summary banner
        risk_val = st.session_state.selected.get(risk_col, "")
        banner_color = COLOR.get(risk_val, "#444")
        txt = "white" if risk_val in ["High","Very High"] else "black"

        st.markdown(
            f"""
            <div style="padding:10px;border-radius:6px;
                        background:{banner_color};color:{txt};
                        text-align:center;font-weight:bold;">
                {len(near)} nearby addresses within {radius_ft} ft of {sel_addr}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Table
        table_cols = [addr_col, risk_col, "Distance (ft)", risk_score_col, num_insp_col, recent_insp_col]
        table_cols = [c for c in table_cols if c in near.columns]

        df_table = near[table_cols].copy()

        # Row shading
        def lighten(hex_color, factor=0.82):
            hex_color = hex_color.lstrip("#")
            r, g, b = (int(hex_color[i:i+2], 16) for i in (0,2,4))
            r = int(r + (255-r)*factor)
            g = int(g + (255-g)*factor)
            b = int(b + (255-b)*factor)
            return f"rgb({r},{g},{b})"

        def highlight(row):
            addr = row[addr_col]
            level = row[risk_col]
            base = COLOR.get(level,"#DDD")
            if addr == sel_addr:
                txtcol = "white" if level in ["High","Very High"] else "black"
                return [f"background:{base};color:{txtcol};font-weight:bold"]*len(row)
            else:
                return [f"background:{lighten(base)};color:black"]*len(row)

        styled = (
            df_table.style
            .apply(highlight, axis=1)
            .set_table_styles([{
                "selector": "thead th",
                "props": [("background-color","#333"),("color","white")]
            }])
        )

        st.dataframe(styled, height=550, use_container_width=True, hide_index=True)

# ============================================================
# CLICK HANDLING (works for both initial & focused map)
# ============================================================
if map_data and map_data.get("last_clicked"):
    handle_click(map_data)
