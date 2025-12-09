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
# Global Style + Toast
# -------------------------
st.markdown("""
<style>
@media (max-width: 600px) {
    h1 {font-size: 1.3rem !important;}
    .stRadio label, .stSelectbox label {font-size: 0.9rem !important;}
    .stDataFrame {font-size: 0.8rem !important;}
}
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
#toast.show {
    visibility: visible;
    animation: fadein 0.5s, fadeout 0.5s 3s;
}
@keyframes fadein {from {bottom: 0; opacity: 0;} to {bottom: 40px; opacity: 1;}}
@keyframes fadeout {from {bottom: 40px; opacity: 1;} to {bottom: 0; opacity: 0;}}
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

st.markdown("""
<style>
/* Make controls and map stack cleanly on mobile */
@media (max-width: 768px) {
    div[data-testid="column"] {
        display: block !important;
        width: 100% !important;
    }
    .stSelectbox, .stRadio {
        width: 100% !important;
    }
    iframe, .stIFrame {
        width: 100% !important;
    }
}
</style>
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
# Normalize risk values (handles UPPERCASE, etc.)
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
    "Very Highrisk": "Very High",
    "Veryhigh": "Very High",
    "Moderaterisk": "Moderate",
    "Lowrisk": "Low"
})

# -------------------------
# Colors
# -------------------------
COLOR = {
    "Very High": "#8B0000",  # dark red
    "High": "#FF0000",       # bright red
    "Moderate": "#FFA500",   # orange
    "Low": "#FFFF00"         # yellow
}

# -------------------------
# Search Box & Controls
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
# Data Cleanup
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
    """Vectorized Haversine distance in meters."""
    R = 6371000.0
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0_rad) * np.cos(lat_rad) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

# -------------------------
# Handle Search Selection
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

def build_focused_map_and_nearby(selected_dict):
    lat = float(selected_dict[lat_col])
    lon = float(selected_dict[lon_col])
    risk_val = selected_dict.get(risk_col, "")
    risk_color = COLOR.get(risk_val, "gray")

    m = folium.Map(
        location=[lat, lon],
        zoom_start=18,
        tiles="https://mt1.google.com/vt/lyrs=y,h&x={x}&y={y}&z={z}",
        attr="Google"
    )

    # Slightly larger circle for visual radius
    draw_radius_m = radius_m * (1.25 if radius_toggle == 200 else 1.1667)
    folium.Circle(
        location=[lat, lon],
        radius=draw_radius_m,
        color="blue",
        fill=False,
        weight=2
    ).add_to(m)

    temp_df = df.copy()
    temp_df["dist_m"] = haversine_vec(lat, lon, temp_df[lat_col].values, temp_df[lon_col].values)
    nearby_df = temp_df[temp_df["dist_m"] <= radius_m].copy()

    if nearby_df.empty:
        # Still put a pulsating center, but no nearby lines/markers
        html = f"""
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
        folium.Marker(location=[lat, lon], icon=folium.DivIcon(html=html)).add_to(m)
        return m, nearby_df

    nearby_df["Distance (ft)"] = (nearby_df["dist_m"] * 3.28084).round(0).astype("Int64")
    nearby_df = nearby_df.sort_values("dist_m", ascending=True).reset_index(drop=True)
    nearby_df["Distance Rank"] = nearby_df.index + 1

    # Lines + markers colored by risk
    for _, r in nearby_df.iterrows():
        rc = COLOR.get(r.get(risk_col, ""), "gray")

        folium.PolyLine(
            [(lat, lon), (r[lat_col], r[lon_col])],
            color=rc,
            weight=1.2,
            opacity=0.45
        ).add_to(m)

        folium.CircleMarker(
            location=[r[lat_col], r[lon_col]],
            radius=6,
            color="white",
            weight=1,
            fill=True,
            fill_color=rc,
            fill_opacity=0.95,
            popup=f"<b>{r.get(street_col, '')}</b><br>Risk: {r.get(risk_col, '')}"
        ).add_to(m)

    # Pulsating center marker
    html = f"""
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
    folium.Marker(location=[lat, lon], icon=folium.DivIcon(html=html)).add_to(m)

    return m, nearby_df

# -------------------------
# Responsive Map Size
# -------------------------
def get_map_dimensions():
    try:
        ua = st.runtime.scriptrunner.script_run_context.session_info.user_agent
        if "Mobile" in ua:
            return 360, 420
        elif "Tablet" in ua:
            return 720, 520
    except Exception:
        pass
    return 1000, 600

map_width, map_height = get_map_dimensions()

# -------------------------
# SIDE-BY-SIDE LAYOUT
# -------------------------
map_col, table_col = st.columns([1.3, 1])

# -------------------------
# LEFT: MAP
# -------------------------
with map_col:
    if st.session_state.selected is None:
        m = build_base_map()
        st.session_state.nearby_df = pd.DataFrame()
    else:
        m, nearby = build_focused_map_and_nearby(st.session_state.selected)
        st.session_state.nearby_df = nearby
        if nearby.empty:
            addr = st.session_state.selected.get(addr_col, "this location")
            st.components.v1.html(
                f"<script>showToast('⚠️ No nearby addresses found near {addr}', '#e6b800');</script>",
                height=0
            )

    map_data = st_folium(m, width=map_width, height=map_height, use_container_width=True)

    # Map click → always select nearest address (Option A)
    if map_data and map_data.get("last_clicked") is not None:
        try:
            click_lat = map_data["last_clicked"]["lat"]
            click_lon = map_data["last_clicked"]["lng"]

            distances = haversine_vec(
                click_lat,
                click_lon,
                df[lat_col].values,
                df[lon_col].values
            )
            nearest_idx = np.argmin(distances)
            nearest_row = df.iloc[nearest_idx]

            # Always select nearest, even if far away
            st.session_state.selected = nearest_row.to_dict()
            st.experimental_rerun()

        except Exception as e:
            msg = str(e).replace("'", "").replace('"', "")
            st.components.v1.html(f"""
                <script>
                showToast("❌ Error processing click: {msg}", "#cc0000");
                </script>
            """, height=0)

# -------------------------
# RIGHT: TABLE
# -------------------------
with table_col:
    if st.session_state.selected is None:
        st.markdown("#### 👉 Select an address from the search bar or click on the map.")
    else:
        nearby_df = st.session_state.nearby_df

        if nearby_df.empty:
            st.warning("No nearby addresses found within the selected radius.")
        else:
            # Build table columns
            table_cols = [street_col, risk_col]
            if risk_score_col in nearby_df.columns:
                table_cols.append(risk_score_col)
            table_cols.append("Distance (ft)")
            for c in [recent_insp_col, num_insp_col]:
                if c and c in nearby_df.columns:
                    table_cols.append(c)

            sort_df = nearby_df.copy()
            sort_df["_d"] = pd.to_numeric(sort_df["Distance (ft)"], errors="coerce")
            sort_df = sort_df.sort_values("_d", ascending=True)
            nearby_df = sort_df
            display_df = nearby_df[table_cols].copy().fillna("")

            sel_street = st.session_state.selected.get(street_col, "")
            sel_risk = st.session_state.selected.get(risk_col, "")
            risk_color = COLOR.get(sel_risk, "gray")
            header_text_color = "white" if sel_risk in ["Very High", "High"] else "black"

            # Top banner
            st.markdown(
                f"<div style='background:{risk_color};color:{header_text_color};font-size:16px;padding:8px;"
                f"border-radius:6px;text-align:center;'>"
                f"{len(display_df)} addresses within {radius_toggle} ft of {sel_street} (Risk: {sel_risk})"
                f"</div>",
                unsafe_allow_html=True
            )

            # Most recent insp date
            if recent_insp_col and recent_insp_col in nearby_df.columns:
                recent_dates = pd.to_datetime(nearby_df[recent_insp_col], errors="coerce").dropna()
                recent_date = recent_dates.max().strftime("%m/%d/%Y") if not recent_dates.empty else "N/A"
            else:
                recent_date = "N/A"

            st.markdown(
                f"<div style='background:#f9f9f9;color:black;font-size:13px;padding:6px;"
                f"border-radius:6px;margin-bottom:8px;text-align:center;'>"
                f"Most recent termite inspection within {radius_toggle} ft: {recent_date}</div>",
                unsafe_allow_html=True
            )

            # Row highlighting
            selected_street_val = str(st.session_state.selected.get(street_col, "")).strip()

            def lighten(color_hex, factor=0.82):
                color_hex = color_hex.lstrip("#")
                r = int(color_hex[0:2], 16)
                g = int(color_hex[2:4], 16)
                b = int(color_hex[4:6], 16)
                r = int(r + (255 - r) * factor)
                g = int(g + (255 - g) * factor)
                b = int(b + (255 - b) * factor)
                return f"rgb({r},{g},{b})"

            def highlight_rows(row):
                street_val = str(row.get(street_col, "")).strip()
                row_risk = row.get(risk_col, "")
                base_color = COLOR.get(row_risk, "#CCCCCC")

                if street_val == selected_street_val:
                    txt_color = "white" if row_risk in ["Very High", "High"] else "black"
                    return [f"background-color:{base_color};color:{txt_color};font-weight:bold;"] * len(row)

                light_color = lighten(base_color)
                return [f"background-color:{light_color};color:black;"] * len(row)

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
