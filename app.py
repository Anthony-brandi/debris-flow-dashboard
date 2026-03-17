import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import ee
import json
from datetime import datetime, timedelta

# ==========================================
# 1. PAGE SETUP & DATA SOURCES
# ==========================================
st.set_page_config(page_title="Wildfire Recovery & SGMA Analysis", layout="wide", page_icon="💧")

# Official CA DWR Groundwater Basins (Bulletin 118) via GeoJSON URL
GW_BASINS_URL = "https://opendata.arcgis.com/datasets/5da310c66bc649f0bc4ad21820b36873_0.geojson"

@st.cache_data
def load_gw_basins():
    # Pulls directly from CA State servers to avoid local file errors
    return gpd.read_file(GW_BASINS_URL)

@st.cache_data
def load_fire_perimeters():
    # Relative path works for both Local VS Code and GitHub Cloud
    path = 'CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view/CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view.shp'
    fires = gpd.read_file(path)
    # Cleaning data for easy selection
    fires = fires.dissolve(by='incident_n').reset_index()
    return fires.to_crs(epsg=4326)

# ==========================================
# 2. GEE INITIALIZATION
# ==========================================
if 'ee_initialized' not in st.session_state:
    try:
        # Check for secrets (Cloud) or local secrets.toml
        if "EARTHENGINE_JSON" in st.secrets:
            creds_dict = json.loads(st.secrets["EARTHENGINE_JSON"])
            credentials = ee.ServiceAccountCredentials(creds_dict['client_email'], key_data=st.secrets["EARTHENGINE_JSON"])
            ee.Initialize(credentials, project='gee-streamlit-app-490500')
        else:
            # Fallback for local testing
            ee.Initialize(project='gee-streamlit-app-490500')
        st.session_state['ee_initialized'] = True
    except Exception as e:
        st.error(f"Google Earth Engine Initialization Error: {e}")

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Interactive Risk Map", "Technical Documentation"])

if page == "Interactive Risk Map":
    try:
        cal_fires = load_fire_perimeters()
        
        st.sidebar.title("Analysis Control")
        fire_list = sorted(cal_fires['incident_n'].fillna(cal_fires['mission']).dropna().unique())
        selected_fire = st.sidebar.selectbox("Select Wildfire Perimeter", fire_list)
        fire_data = cal_fires[cal_fires['incident_n'] == selected_fire]
        
        # --- RECOVERY MONITOR ---
        st.sidebar.markdown("---")
        st.sidebar.subheader(" Vegetation Recovery Monitor")
        recovery_months = st.sidebar.select_slider(
            "Months Post-Fire to Analyze",
            options=[1, 6, 12, 18, 24],
            value=1,
            help="Higher months show how the landscape 'greens up' over time."
        )
        
        analyze_btn = st.sidebar.checkbox("Run Spatial Analysis", value=False)
        slope_limit = st.sidebar.slider("Slope Threshold (Degrees)", 10, 45, 27)

        # --- LAYER CONTROLS ---
        st.sidebar.markdown("---")
        with st.sidebar.expander("⚙️ Map Layer Visibility", expanded=True):
            show_recovery = st.checkbox("Show Recovery (dNBR)", value=True)
            show_basins = st.checkbox("CA Groundwater Basins (SGMA)", value=True)
            show_risk = st.checkbox("Critical Debris Hazard (Orange)", value=False)
            show_infra = st.checkbox("Infrastructure (Roads)", value=True)
            
            basemap_opt = st.radio("Basemap Style", ["Google Satellite", "Google Terrain"])
            basemap_url = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}' if basemap_opt == "Google Satellite" else 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}'

        # --- MAP INITIALIZATION ---
        st.title(f"{selected_fire} Debris Flow & Aquifer Recharge Dashboard")
        centroid = fire_data.geometry.centroid.iloc[0]
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles=basemap_url, attr=basemap_opt)
        folium.GeoJson(fire_data.geometry, style_function=lambda x: {'fillColor': 'transparent', 'color': 'red', 'weight': 3}).add_to(m)

        if analyze_btn:
            with st.spinner(f"Processing {recovery_months} months of satellite data..."):
                area = ee.FeatureCollection(fire_data.__geo_interface__)
                
                # --- AUTOMATED DATES ---
                fire_start_raw = fire_data['ALARM_DATE'].iloc[0] if 'ALARM_DATE' in fire_data.columns else "2021-06-01"
                fire_date_ee = ee.Date(pd.to_datetime(fire_start_raw).strftime('%Y-%m-%d'))
                
                # Compare 1 year before fire to X months after fire
                pre_fire_date = fire_date_ee.advance(-1, 'year')
                post_fire_date = fire_date_ee.advance(recovery_months, 'month')

                def get_nbr_image(date):
                    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                        .filterBounds(area)\
                        .filterDate(date.advance(-1, 'month'), date.advance(1, 'month'))\
                        .median().clip(area).normalizedDifference(['B8', 'B12'])

                nbr_pre = get_nbr_image(pre_fire_date)
                nbr_post = get_nbr_image(post_fire_date)
                dnbr = nbr_pre.subtract(nbr_post)

                # --- RENDER DNBR RECOVERY ---
                if show_recovery:
                    dnbr_vis = {'min': -0.1, 'max': 0.5, 'palette': ['ffffff', '7ad071', 'f9e072', 'ff0000']}
                    dnbr_id = dnbr.getMapId(dnbr_vis)
                    folium.TileLayer(tiles=dnbr_id['tile_fetcher'].url_format, attr='Sentinel-2', name='Recovery Status', overlay=True, opacity=0.6).add_to(m)

                # --- RENDER GROUNDWATER BASINS ---
                if show_basins:
                    basins = load_gw_basins()
                    folium.GeoJson(
                        basins,
                        name="SGMA Basins",
                        style_function=lambda x: {'fillColor': '#3498db', 'color': 'blue', 'weight': 1, 'fillOpacity': 0.15},
                        tooltip=folium.GeoJsonTooltip(fields=['Basin_Name'], aliases=['Basin Name:'])
                    ).add_to(m)

                # --- RENDER HAZARD (Slope + No Recovery) ---
                if show_risk:
                    dem = ee.Image("USGS/SRTMGL1_003")
                    slope = ee.Terrain.slope(dem).clip(area)
                    # Hazard exists where slope is steep AND vegetation hasn't recovered (dnbr > 0.1)
                    hazard_mask = slope.gte(slope_limit).And(dnbr.gt(0.1))
                    h_id = hazard_mask.updateMask(hazard_mask).getMapId({'palette': ['#ff7b00'], 'opacity': 0.9})
                    folium.TileLayer(tiles=h_id['tile_fetcher'].url_format, attr='GEE', name='Active Hazard', overlay=True).add_to(m)

                # --- RENDER INFRASTRUCTURE ---
                if show_infra:
                    roads = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(area)
                    roads_img = ee.Image(0).mask(0).paint(roads, 1, 2)
                    infra_id = roads_img.getMapId({'palette': ['#2ecc71']})
                    folium.TileLayer(tiles=infra_id['tile_fetcher'].url_format, attr='TIGER', name='Roads', overlay=True).add_to(m)

                # --- STATISTICS ---
                avg_dnbr = dnbr.reduceRegion(reducer=ee.Reducer.mean(), geometry=area.geometry(), scale=30).getInfo().get('nd', 0)
                st.sidebar.markdown("---")
                st.sidebar.metric(f"Avg. Burn Severity ({recovery_months} mo)", f"{avg_dnbr:.3f}")
                if avg_dnbr < 0.1: st.sidebar.success("✅ Significant recovery detected.")
                else: st.sidebar.warning("⚠️ High severity persists.")

        # FINAL MAP RENDER
        st_folium(m, use_container_width=True, height=750, key="map_main")

    except Exception as e:
        st.error(f"Application Runtime Error: {e}")

# ==========================================
# 4. TECHNICAL DOCUMENTATION
# ==========================================
elif page == "Technical Documentation":
    st.title("Scientific & Policy Framework")
    st.markdown("---")
    st.header("1. The SGMA Connection")
    st.write("Post-fire recovery is a vital metric for Groundwater Sustainability Agencies (GSAs). Hydrophobic soils in severely burned areas prevent natural recharge, impacting basin sustainability goals.")
    
    st.header("2. Methodology (dNBR)")
    st.latex(r"dNBR = NBR_{pre-fire} - NBR_{post-fire}")
    st.write("This dashboard monitors recovery over a 24-month horizon. As vegetation returns, debris flow risk decreases and infiltration capacity increases.")
