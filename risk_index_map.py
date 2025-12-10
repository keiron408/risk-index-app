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
street_col = find_col(df.columns, ["fulladdress", "street", "street_name"])
risk_col = find_col(df.columns, ["risk_level", "risk"])
risk_score_col = find_col(df.columns, ["risk_score"])
recent_insp_col = find_col(df.columns, ["most recent inspection"])
num_insp_col = find_col(df.columns, ["# of inspections"])
search_addr_col = "search address"

if not street_col:
    street_col = addr_col

# ============================================================
# FIX "# OF INSPECTIONS" TO INTEGER
# ============================================================
if num_insp_col and num_insp_col in df.columns:
    df[num_insp_col] = (
        pd.to_numeric(df[num_insp_col], errors="coerce")
        .fillna(0)
        .astype(int)
    )

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
st.session_state.setdefault("clear_search", False)

sel = st.session_state.get("selected", None)
if (
    not isinstance(sel, dict)
    or sel is None
    or lat_col not in sel
    or lon_col not in sel
):
    st.session_state.selected = None

# ============================================================
# DISTANCE
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
    return sorted(df[search_addr_col].dropna().unique())

colA, colB = st.columns([2, 1])

if st.session_state.clear_search:
    st.session_state.search_box = ""
    st.session_state.last_search_value = ""
    st.session_state.clear_search = False

with colA:
    opts = get_search_options(df)
    placeholder = "Enter address / select from map..."

    def fmt(v):
        return placeholder if v == "" else v

    search_choice = st.selectbox(
        "Search",
        [""] + opts,
        key="search_box",
        format_func=fmt,
        label_visibility="collapsed",
    )

with colB:
    radius_ft = st.radio("Radius (ft)", [200, 300], horizontal=True)

radius_m = radius_ft * 0.3048

user_changed_search = (search_choice != st.session_state.last_search_value)
st.session_state.last_search_value = search_choice

if user_changed_search and search_choice != "":
    match = df[df[search_addr_col] == search_choice]
    if not match.empty:
        st.session_state.selected = match.iloc[0].to_dict()
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

    folium.Circle(
        (lat, lon),
        radius_m * 1.20,
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

    # numbering
    selected_address = selected.get(street_col, "")
    numbers = {}
    counter = 1

    for _, r in near.iterrows():
        addr = r.get(street_col, "")
        if addr == selected_address:
            numbers[addr] = ""
        else:
            numbers[addr] = str(counter)
            counter += 1

    # dashed lines
    for _, r in near.iterrows():
        col = COLOR.get(r.get(risk_col, ""), "gray")
        folium.PolyLine(
            [(lat, lon), (r[lat_col], r[lon_col])],
            color=col,
            weight=1.5,
            opacity=0.5,
            dash_array="5, 5",
        ).add_to(m)

    # numbered markers
    for _, r in near.iterrows():
        addr = r.get(street_col, "")
        marker_num = numbers.get(addr, "")
        col = COLOR.get(r.get(risk_col, ""), "gray")

        folium.map.Marker(
            location=(r[lat_col], r[lon_col]),
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    background:{col};
                    width:22px;height:22px;
                    border-radius:50%;
                    border:2px solid black;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-size:12px;
                    font-weight:bold;
                    color:black;">
                    {marker_num}
                </div>
                """
            ),
        ).add_to(m)

    # center marker
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
# CLICK HANDLER
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

    last = st.session_state.get("map_last_click")
    if last and abs(last["lat"] - lat) < 1e-9 and abs(last["lon"] - lon) < 1e-9:
        return

    st.session_state.map_last_click = {"lat": lat, "lon": lon}

    d = haversine_vec(lat, lon, df[lat_col], df[lon_col])
    nearest_idx = int(np.argmin(d))
    st.session_state.selected = df.iloc[nearest_idx].to_dict()

    st.session_state.clear_search = True
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

    if map_data and map_data.get("last_clicked"):
        handle_map_click(map_data)

# ============================================================
# MAP + TABLE LAYOUT
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

        if map_data and map_data.get("last_clicked"):
            handle_map_click(map_data)

        legend()

    # ================= TABLE ====================
    with table_col:
        df2 = st.session_state.nearby_df.copy()

        if df2.empty:
            st.warning("No nearby addresses within the selected radius.")
            st.session_state.nearby_df = pd.DataFrame()
        else:
            # numbering column
            if "No." in df2.columns:
                df2 = df2.drop(columns=["No."])
            df2.insert(0, "No.", "")

            selected_address = st.session_state.selected.get(street_col, "")
            counter = 1
            for i in df2.index:
                if df2.at[i, street_col] == selected_address:
                    df2.at[i, "No."] = ""
                else:
                    df2.at[i, "No."] = counter
                    counter += 1

            # columns for table
            table_cols = ["No.", street_col, risk_col, "Distance (ft)"]

            if risk_score_col in df2.columns:
                table_cols.insert(3, risk_score_col)

            if recent_insp_col in df2.columns:
                table_cols.append(recent_insp_col)

            if num_insp_col in df2.columns:
                table_cols.append(num_insp_col)

            df2 = df2[table_cols].fillna("")

            # summary banner (neighbor count excludes selected itself)
            sel_addr = st.session_state.selected.get(street_col, "")
            sel_risk = st.session_state.selected.get(risk_col, "")
            banner_color = COLOR.get(sel_risk, "#444")
            text_color = "white" if sel_risk in ["High", "Very High"] else "black"

            neighbor_count = max(len(df2) - 1, 0)

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
                    {neighbor_count} nearby addresses within {radius_ft} ft of {sel_addr}
                </div>
                """,
                unsafe_allow_html=True
            )

            # optional “most recent inspection” summary
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

            # row coloring
            def lighten(hex_color, factor=0.82):
                hex_color = hex_color.lstrip("#")
                r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                r = int(r + (255 - r)*factor)
                g = int(g + (255 - g)*factor)
                b = int(b + (255 - b)*factor)
                return f"rgb({r},{g},{b})"

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
                    "props": [("background-color", "#333"),
                              ("color", "white"),
                              ("font-weight", "bold")]
                }])
            )

            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                height=550,
            )
