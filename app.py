import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import ee
import json
import requests
from datetime import datetime, timedelta
import altair as alt
import zipfile
import os

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(page_title="Watershed Risk Portal", layout="wide")

# ==========================================
# 2. DATA LOADERS
# ==========================================
@st.cache_data
def load_fire_perimeters():
    try:
        possible_names = ['Master_Fire_Dataset.zip', 'Master_Fire_Dataset.geojson.zip']
        actual_zip = next((name for name in possible_names if os.path.exists(name)), None)
        if not actual_zip: return None
        with zipfile.ZipFile(actual_zip, 'r') as zip_ref:
            geojson_filename = [f for f in zip_ref.namelist() if f.endswith('.geojson')][0]
            zip_ref.extract(geojson_filename)
        fires = gpd.read_file(geojson_filename)
        fires['final_date'] = pd.to_datetime(fires['final_date'])
        return fires
    except: return None

# ==========================================
# 3. GEE INITIALIZATION
# ==========================================
if 'ee_initialized' not in st.session_state:
    try:
        if "EARTHENGINE_JSON" in st.secrets:
            creds = json.loads(st.secrets["EARTHENGINE_JSON"])
            ee.Initialize(ee.ServiceAccountCredentials(creds['client_email'], key_data=st.secrets["EARTHENGINE_JSON"]), project='gee-streamlit-app-490500')
        else: ee.Initialize(project='gee-streamlit-app-490500')
        st.session_state['ee_initialized'] = True
    except Exception as e: st.error(f"GEE Error: {e}")

# ==========================================
# 4. SIDEBAR & NAVIGATION
# ==========================================
st.sidebar.title("Risk Portal Navigation")
page = st.sidebar.selectbox("Select View", ["1. Incident Briefing", "2. Interactive Analysis", "3. Statistical Report"])

all_fires = load_fire_perimeters()
if all_fires is not None:
    fire_names = sorted(all_fires['incident_n'].dropna().unique())
    selected_name = st.sidebar.selectbox("Choose Wildfire Incident", fire_names)
    fire_subset = all_fires[all_fires['incident_n'] == selected_name]
    default_alarm_dt = fire_subset['final_date'].iloc[0]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Parameters")
    manual_baseline = st.sidebar.date_input("Pre-Fire Baseline Date", value=default_alarm_dt - timedelta(days=365))
    recovery_months = st.sidebar.select_slider("Observation Window (Months)", options=[1, 6, 12, 18, 24], value=1)
    dnbr_limit = st.sidebar.slider("Burn Severity Threshold (dNBR)", 0.05, 0.70, 0.20, 0.05)
    slope_limit = st.sidebar.slider("Critical Slope Threshold (Deg)", 5, 45, 20)

# ==========================================
# PAGE 2: INTERACTIVE ANALYSIS
# ==========================================
if page == "2. Interactive Analysis" and all_fires is not None:
    st.title("Interactive GIS Lab")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Layer Toggles")
    show_k = st.sidebar.checkbox("Soil Erodibility (K-Factor)", value=True)
    show_recovery = st.sidebar.checkbox("Burn Severity (dNBR)", value=True)
    show_risk = st.sidebar.checkbox("Hazard Intersection (Orange)", value=True)
    show_streams = st.sidebar.checkbox("Stream Networks", value=True)
    show_roads = st.sidebar.checkbox("Major Roads (TIGER)", value=True)
    
    run_analysis = st.toggle("Activate Spatial Modeling Engine", value=True)

    if run_analysis:
        with st.spinner("Analyzing multispectral and soil datasets..."):
            area = ee.FeatureCollection(fire_subset.__geo_interface__)
            pre_date = ee.Date(manual_baseline.strftime('%Y-%m-%d'))
            fire_start = ee.Date(default_alarm_dt.strftime('%Y-%m-%d'))
            target_date = fire_start.advance(recovery_months, 'month')

            # Layers
            def get_nbr(d): return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area).filterDate(d.advance(-2, 'month'), d.advance(2, 'month')).median().clip(area).normalizedDifference(['B8', 'B12'])
            dnbr = get_nbr(pre_date).subtract(get_nbr(target_date))
            slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003").clip(area))
            
            # Refined K-Factor Heatmap logic
            soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0').clip(area)
            k_factor = soil.remap([1,2,3,4,5,6,7,8,9,10,11,12], [15,25,15,30,35,20,30,40,25,45,10,5]).divide(100.0)
            
            hazard_mask = slope.gte(slope_limit).And(dnbr.gt(dnbr_limit))
            streams = ee.Image(0).mask(0).paint(ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(area), 1, 2)
            roads = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(area)

            centroid = fire_subset.geometry.centroid.iloc[0]
            m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr="Google")
            
            # PERMANENT LEGEND
            legend_html = f"""
            <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; background-color: rgba(255, 255, 255, 0.9); border:2px solid black; z-index:9999; font-size:12px; padding: 10px;">
            <b>Map Legend</b><br>
            <i style="background:#ff7b00; width:12px; height:12px; float:left; margin-right:5px;"></i> Hazard Intersection<br>
            <i style="background:#bd0026; width:12px; height:12px; float:left; margin-right:5px;"></i> Burn Scar (Severe)<br>
            <i style="background:#8c510a; width:12px; height:12px; float:left; margin-right:5px;"></i> High Soil Erodibility<br>
            <i style="background:#3498db; width:12px; height:3px; float:left; margin-right:5px; margin-top:5px;"></i> Stream Flow<br>
            <i style="background:white; border:1px solid black; width:12px; height:2px; float:left; margin-right:5px; margin-top:5px;"></i> Major Roads
            </div>"""
            m.get_root().html.add_child(folium.Element(legend_html))

            if show_k:
                folium.TileLayer(tiles=k_factor.getMapId({'min': 0.1, 'max': 0.45, 'palette': ['#f6e8c3','#dfc27d','#bf812d','#8c510a']})['tile_fetcher'].url_format, attr='Soil', name='Soil Erodibility', opacity=0.5).add_to(m)
            if show_recovery:
                folium.TileLayer(tiles=dnbr.updateMask(dnbr.gt(0.1)).getMapId({'min': 0.1, 'max': 0.5, 'palette': ['#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026']})['tile_fetcher'].url_format, attr='S2', name='Burn Severity', opacity=0.6).add_to(m)
            if show_risk:
                folium.TileLayer(tiles=hazard_mask.updateMask(hazard_mask).getMapId({'palette':['#ff7b00']})['tile_fetcher'].url_format, attr='GEE', name='Hazard Intersection').add_to(m)
            if show_streams:
                folium.TileLayer(tiles=streams.getMapId({'palette':['#3498db']})['tile_fetcher'].url_format, attr='Hydro', name='Streams').add_to(m)
            if show_roads:
                folium.GeoJson(roads.getInfo(), style_function=lambda x: {'color': 'white', 'weight': 2, 'opacity': 0.8}, name='Roads').add_to(m)

            st_folium(m, use_container_width=True, height=700)
