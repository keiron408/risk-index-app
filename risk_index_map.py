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
# AUTO-DETECT COLUMNS (KEEPS MAP WORKING)
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
# RISK NORMALIZATION
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
    "High":     "#FF0000",
    "Moderate": "#FFA500",
    "Low":      "#FFFF00",
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
st.session_state.setdefault("last_search_value", "")

# Sanity check: if selected is not a dict with lat/lon, reset it
sel = st.session_state.get("selected", None)
if (
    not isinstance(sel, dict)
    or sel is None
    or lat_col not in sel
    or lon_col not in sel
):
    st.session_state.selected = None

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
def get_search_options(df):
    return sorted(df["search address"].dropna().unique())

colA, colB = st.columns([2, 1])

with colA:
    opts = get_search_options(df)
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

# Only update selected from SEARCH if user actually changed the dropdown
user_changed_search = (search_choice != st.session_state.last_search_value)
st.session_state.last_search_value = search_choice

if user_changed_search and search_choice:
    match = df[df["search address"] == search_choice]
    if not match.empty:
        st.session_state.selected = match.iloc[0].to_dict()
        st.session_state.map_last_click = None  # reset map click

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

    # Radius ring
    ring_radius = radius_m * 1.20
    folium.Circle(
        (lat, lon),
        ring_radius,
        color="blue",
        fill=False,
        weight=2,
    ).add_to(m)

    # Compute nearby parcels
    temp = df.copy()
    temp["dist_m"] = haversine_vec(lat, lon, temp[lat_col], temp[lon_col])
    near = temp[temp["dist_m"] <= radius_m].copy()

    if near.empty:
        return m, near

    near["Distance (ft)"] = (near["dist_m"] * 3.28084).round(0).astype("Int64")
    near = near.sort_values("dist_m").reset_index(drop=True)

    # Radial dashed lines inward
    for _, r in near.iterrows():
        rc = COLOR.get(r.get(risk_col, ""), "gray")
        folium.PolyLine(
            [(lat, lon), (r[lat_col], r[lon_col])],
            color=rc,
            weight=1.5,
            opacity=0.5,
            dash_array="5, 5"
        ).add_to(m)

    # Nearby markers
    for _, r in near.iterrows():
        c = COLOR.get(r.get(risk_col, ""), "gray")
        folium.CircleMarker(
            (r[lat_col], r[lon_col]),
            radius=6,
            color="white",
            fill=True,
            fill_color=c,
            fill_opacity=0.95,
            weight=1,
        ).add_to(m)

    # Center marker
    folium.CircleMarker(
        (lat, lon),
        radius=10,
        color="black",
        fill=True,
        fill_color=risk_color,
        weight=2,
        fill_opacity=1,
    ).add_to(m)

    return m, near

# ============================================================
# CLICK HANDLER — FIRST CLICK AFTER SEARCH UPDATES IMMEDIATELY
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

    # Prevent ghost-click duplicates
    last = st.session_state.get("map_last_click")
    if last and abs(last["lat"] - lat) < 1e-9 and abs(last["lon"] - lon) < 1e-9:
        return

    st.session_state.map_last_click = {"lat": lat, "lon": lon}

    # Snap to nearest parcel
    d = haversine_vec(lat, lon, df[lat_col], df[lon_col])
    nearest_idx = int(np.argmin(d))
    st.session_state.selected = df.iloc[nearest_idx].to_dict()

    # When user clicks map, clear search intent AND dropdown
    st.session_state.search_box = ""            # <— NEW AND CRITICAL
    st.session_state.last_search_value = ""     # <— previously added, still needed

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
# INITIAL MAP (FULL WIDTH)
# ============================================================
if st.session_state.selected is None:

    m = build_base_map()
    map_data = st_folium(
        m,
        height=600,
        use_container_width=True,
        key="mainmap"
    )
    legend()

    # CLICK HANDLING FOR INITIAL MAP
    if map_data and map_data.get("last_clicked"):
        handle_map_click(map_data)

# ============================================================
# MAP + TABLE LAYOUT (SIDE BY SIDE)
# ============================================================
else:
    map_col, table_col = st.columns([1.3, 1])

    with map_col:
        m, near = build_focused_map_and_nearby(st.session_state.selected)
        st.session_state.nearby_df = near

        map_data = st_folium(
            m,
            height=600,
            use_container_width=True,
            key="mainmap"
        )

        # CLICK HANDLING FOR FOCUSED MAP
        if map_data and map_data.get("last_clicked"):
            handle_map_click(map_data)

        legend()

    with table_col:

        df2 = st.session_state.nearby_df

        if df2.empty:
            st.warning("No nearby addresses within the selected radius.")
            st.session_state.nearby_df = pd.DataFrame()
        else:
            # Selected columns (FullAddress for display, risk, distance, etc.)
            table_cols = [street_col, risk_col, "Distance (ft)"]
            if risk_score_col in df2.columns:
                table_cols.insert(2, risk_score_col)
            if recent_insp_col in df2.columns:
                table_cols.append(recent_insp_col)
            if num_insp_col in df2.columns:
                table_cols.append(num_insp_col)

            df2 = df2[table_cols].fillna("")

            # SUMMARY BANNER
            sel_addr = st.session_state.selected.get(street_col, "")
            sel_risk = st.session_state.selected.get(risk_col, "")
            banner_color = COLOR.get(sel_risk, "#444")
            text_color = "white" if sel_risk in ["High", "Very High"] else "black"

            st.markdown(
                f"""
                <div style="padding:10px 14px;
                            margin-bottom:8px;
                            border-radius:6px;
                            background:{banner_color};
                            color:{text_color};
                            font-size:15px;
                            font-weight:bold;
                            text-align:center;">
                    {len(df2)} nearby addresses within {radius_ft} ft of {sel_addr}<br>
                    (Risk level: {sel_risk})
                </div>
                """,
                unsafe_allow_html=True
            )

            # RECENT INSPECTION BAR
            if recent_insp_col in df2.columns:
                recent_vals = pd.to_datetime(df2[recent_insp_col], errors='coerce')
                recent_text = str(recent_vals.max().date()) if not recent_vals.isna().all() else "N/A"

                st.markdown(
                    f"""
                    <div style="padding:8px;
                                margin-bottom:6px;
                                border-radius:6px;
                                background:#eee;
                                color:#333;
                                text-align:center;
                                font-size:13px;">
                        Most recent inspection among nearby addresses: {recent_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # FORCE HEADER ROW
            df2 = df2.rename(columns={col: col for col in df2.columns})

            # ROW COLORING
            def lighten(hex_color, factor=0.82):
                hex_color = hex_color.lstrip("#")
                r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                r = int(r + (255 - r)*factor)
                g = int(g + (255 - g)*factor)
                b = int(b + (255 - b)*factor)
                return f"rgb({r},{g},{b})"

            selected_address = st.session_state.selected.get(street_col, "")

            def highlight_rows(row):
                addr = str(row.get(street_col, ""))
                level = row.get(risk_col, "")
                base = COLOR.get(level, "#DDD")

                if addr == selected_address:
                    txt = "white" if level in ["High", "Very High"] else "black"
                    return [f"background-color:{base};color:{txt};font-weight:bold;"] * len(row)

                return [f"background-color:{lighten(base)};color:black;"] * len(row)

            styled = (
                df2.style
                .apply(highlight_rows, axis=1)
                .set_table_styles([{
                    "selector": "thead th",
                    "props": [("background-color", "#333"), ("color", "white"), ("font-weight", "bold")]
                }])
            )

            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                height=550,
                column_config={col: st.column_config.Column(col) for col in df2.columns}
            )
