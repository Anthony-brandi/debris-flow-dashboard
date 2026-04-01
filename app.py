import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import ee
import json
from datetime import datetime, timedelta
import zipfile
import os

# ==========================================
# 1. PAGE SETUP & ARCHITECTURE
# ==========================================
st.set_page_config(page_title="PF-WRP | Post-Fire Watershed Risk Portal", layout="wide")

st.sidebar.title("PF-WRP Navigation")
page = st.sidebar.radio("Select Module:", [
    "1. Incident Briefing", 
    "2. Spatial Modeling Lab", 
    "3. Watershed Loading (Phase 2)"
])

# ==========================================
# 2. GEE INITIALIZATION
# ==========================================
if 'ee_initialized' not in st.session_state:
    try:
        if "EARTHENGINE_JSON" in st.secrets:
            creds_dict = json.loads(st.secrets["EARTHENGINE_JSON"])
            credentials = ee.ServiceAccountCredentials(creds_dict['client_email'], key_data=st.secrets["EARTHENGINE_JSON"])
            # Corrected Project ID to match your service account
            ee.Initialize(credentials, project='gee-streamlit-app-490500')
        else:
            ee.Initialize(project='gee-streamlit-app-490500')
        st.session_state['ee_initialized'] = True
    except Exception as e:
        st.error(f"Earth Engine Initialization Error: {e}")

# ==========================================
# 3. ROBUST CLOUD DATA LOADER
# ==========================================
@st.cache_data
def fetch_and_extract_fire_data():
    zip_path = 'Master_Fire_Dataset.geojson.zip'
    extract_dir = 'temp_fire_data_v4'
    
    try:
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    if not member.startswith('__MACOSX') and member.endswith('.geojson'):
                        zip_ref.extract(member, extract_dir)
        
        for file in os.listdir(extract_dir):
            if file.endswith('.geojson'):
                geojson_path = os.path.join(extract_dir, file)
                fires = gpd.read_file(geojson_path)
                fires = fires.dissolve(by='incident_n').reset_index()
                return fires.to_crs(epsg=4326)
        raise FileNotFoundError("No .geojson file found in archive.")
    except Exception as e:
        st.error(f"Failed to load perimeter data: {e}")
        return gpd.GeoDataFrame()

# ==========================================
# GLOBAL FIRE SELECTION & DATE PARSING
# ==========================================
cal_fires = fetch_and_extract_fire_data()

if not cal_fires.empty:
    name_col = 'incident_n' if 'incident_n' in cal_fires.columns else cal_fires.columns[0]
    fire_series = cal_fires[name_col]
    if 'mission' in cal_fires.columns:
        fire_series = fire_series.fillna(cal_fires['mission'])
    fire_list = sorted(fire_series.dropna().astype(str).unique())
    selected_fire = st.sidebar.selectbox("Select Wildfire Perimeter", fire_list)
    fire_data = cal_fires[cal_fires[name_col] == selected_fire]
    
    ignition_date = datetime(2021, 1, 1)
    for col in ['START_DATE', 'ALARM_DATE', 'alarm_date', 'cont_date']:
        if col in fire_data.columns and not pd.isna(fire_data[col].iloc[0]):
            try:
                ignition_date = pd.to_datetime(fire_data[col].iloc[0]).to_pydatetime()
                break
            except: continue

    pre_fire_start = (ignition_date - timedelta(days=365)).strftime('%Y-%m-%d')
    pre_fire_end = (ignition_date - timedelta(days=1)).strftime('%Y-%m-%d')
    post_fire_start = (ignition_date + timedelta(days=90)).strftime('%Y-%m-%d')
    post_fire_end = (ignition_date + timedelta(days=180)).strftime('%Y-%m-%d')

    area = ee.FeatureCollection(fire_data.__geo_interface__)
    centroid = fire_data.to_crs(epsg=3310).geometry.centroid.to_crs(epsg=4326).iloc[0]
else:
    st.error("No fire perimeters loaded.")
    st.stop()

def mask_s2_clouds(image):
    qa = image.select('QA60')
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask).divide(10000)

# ==========================================
# PAGE 1: INCIDENT BRIEFING
# ==========================================
if page == "1. Incident Briefing":
    st.title(f"Incident Briefing: {selected_fire}")
    total_ac = (fire_data.to_crs(epsg=3310).area.sum()) * 0.000247105
    st.metric("Total Acres Burned", f"{total_ac:,.0f} ac")
    st.metric("Estimated Ignition Date", ignition_date.strftime('%B %d, %Y'))
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles="CartoDB positron")
    folium.GeoJson(fire_data.geometry, style_function=lambda x: {'fillColor': 'red', 'color': 'darkred', 'weight': 2, 'fillOpacity': 0.4}).add_to(m)
    st_folium(m, use_container_width=True, height=500)

# ==========================================
# PAGE 2: SPATIAL MODELING LAB
# ==========================================
elif page == "2. Spatial Modeling Lab":
    st.title("Spatial Modeling Lab (Engineering View)")
    SLOPE_LIMIT = 25
    DNBR_THRESHOLD = 0.25
    st.sidebar.info(f"**Critical Slope:** > {SLOPE_LIMIT}°\n\n**Severity (dNBR):** > {DNBR_THRESHOLD}")
    
    with st.sidebar.expander("Methodology & Reasoning"):
        st.write("Slopes > 25° provide gravitational energy for debris initiation. dNBR > 0.25 isolates moderate-to-high severity zones where hydrophobic soil sealing is likely.")

    show_risk = st.sidebar.checkbox("Hazard Intersection (Risk)", value=True)
    show_slope = st.sidebar.checkbox("Topographic Velocity (Slope)", value=False)
    show_severity = st.sidebar.checkbox("Burn Severity (dNBR)", value=False)
    show_soils = st.sidebar.checkbox("Soil Erodibility (K-Factor)", value=False)
    show_streams = st.sidebar.checkbox("HydroSHEDS Stream Routing", value=True)
    show_roads = st.sidebar.checkbox("TIGER Roads", value=True)

    with st.spinner("Compiling Spatial Intersection Data..."):
        dem = ee.Image("USGS/SRTMGL1_003")
        slope = ee.Terrain.slope(dem).clip(area)
        slope_mask = slope.gte(SLOPE_LIMIT)

        s2_pre = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(area).filterDate(pre_fire_start, pre_fire_end).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).map(mask_s2_clouds).median().clip(area)
        s2_post = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(area).filterDate(post_fire_start, post_fire_end).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).map(mask_s2_clouds).median().clip(area)
        dnbr = s2_pre.normalizedDifference(['B8', 'B12']).subtract(s2_post.normalizedDifference(['B8', 'B12']))
        severity_mask = dnbr.gte(DNBR_THRESHOLD)

        erodible_soils = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0').clip(area).lt(11).selfMask()
        hazard_intersection = slope_mask.And(severity_mask).And(erodible_soils).selfMask()

        roads_img = ee.Image(0).mask(0).paint(ee.FeatureCollection("TIGER/2016/Roads").filterBounds(area), 1, 2)
        streams_img = ee.Image(0).mask(0).paint(ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(area), 1, 1)

        m2 = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Hybrid')
        if show_slope: folium.TileLayer(tiles=slope_mask.selfMask().getMapId({'palette':['yellow'],'opacity':0.4})['tile_fetcher'].url_format, attr='USGS', name='Slope').add_to(m2)
        if show_severity: folium.TileLayer(tiles=severity_mask.selfMask().getMapId({'palette':['red'],'opacity':0.4})['tile_fetcher'].url_format, attr='ESA', name='Severity').add_to(m2)
        if show_soils: folium.TileLayer(tiles=erodible_soils.getMapId({'palette':['#800026'],'opacity':0.4})['tile_fetcher'].url_format, attr='Soil', name='Soils').add_to(m2)
        if show_streams: folium.TileLayer(tiles=streams_img.getMapId({'palette':['#3498db']})['tile_fetcher'].url_format, attr='WWF', name='Streams').add_to(m2)
        if show_roads: folium.TileLayer(tiles=roads_img.getMapId({'palette':['#2ecc71']})['tile_fetcher'].url_format, attr='TIGER', name='Roads').add_to(m2)
        if show_risk: folium.TileLayer(tiles=hazard_intersection.getMapId({'palette':['#ff7b00'],'opacity':0.9})['tile_fetcher'].url_format, attr='GEE', name='Risk').add_to(m2)

        toggle_key = f"lab_{selected_fire}_v{show_risk}{show_slope}{show_severity}{show_soils}{show_streams}{show_roads}"
        st_folium(m2, use_container_width=True, height=700, key=toggle_key)

elif page == "3. Watershed Loading (Phase 2)":
    st.warning("Phase 2 Module: Vulnerability Matrix and Sediment Yield Calculations are currently under development.")
