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

@st.cache_data
def fetch_dins_damage(incident_name):
    url = "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/ArcGIS/rest/services/DINS_Public_View/FeatureServer/0/query"
    params = {"where": f"UPPER(INCIDENT_NAME) LIKE '%{str(incident_name).upper()}%'", "returnCountOnly": "true", "f": "json"}
    try: return requests.get(url, params=params).json().get('count', 0)
    except: return 0

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
    dnbr_limit = st.sidebar.slider("Burn Severity Threshold (dNBR)", 0.10, 0.70, 0.25, 0.05)
    slope_limit = st.sidebar.slider("Critical Slope Threshold (Deg)", 10, 45, 27)

# ==========================================
# PAGE 1: INCIDENT BRIEFING
# ==========================================
if page == "1. Incident Briefing" and all_fires is not None:
    st.header(f"Incident Brief: {selected_name}")
    
    # CASE STUDY LOGIC
    case_studies = {
        "DIXIE": {"impact": "Massive infrastructure loss; Town of Greenville destroyed.", "notes": "Atmospheric River event in Oct 2021 triggered significant sediment movement."},
        "THOMAS": {"impact": "Devastating debris flows in Montecito.", "notes": "23 fatalities. This event defines modern post-fire risk management."},
        "CALDOR": {"impact": "Threatened Lake Tahoe basin; Highway 50 closures.", "notes": "High granite/sand content in soil led to massive runoff."}
    }
    
    if selected_name in case_studies:
        with st.expander("Critical Case Study Details", expanded=True):
            c1, c2 = st.columns(2)
            c1.warning(f"Impact: {case_studies[selected_name]['impact']}")
            c2.info(f"Analysis Note: {case_studies[selected_name]['notes']}")

    impacted_count = fetch_dins_damage(selected_name)
    total_acres = (fire_subset.to_crs(epsg=3310).area.sum()) * 0.000247105
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recorded Ignition", default_alarm_dt.strftime('%b %d, %Y'))
    m2.metric("Total Perimeter", f"{total_acres:,.1f} Ac")
    m3.metric("Lead Agency", "Interagency Database")
    m4.metric("Structures Impacted", f"{impacted_count}")

    col1, col2 = st.columns([2, 1])
    with col1:
        centroid = fire_subset.geometry.centroid.iloc[0]
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles='CartoDB positron')
        folium.GeoJson(fire_subset.geometry, style_function=lambda x: {'color': 'red', 'weight': 2}).add_to(m)
        st_folium(m, use_container_width=True, height=500)
    with col2:
        st.subheader("Field Observations")
        st.write(fire_subset.drop(columns=['geometry']).iloc[0])

# ==========================================
# PAGE 2: INTERACTIVE ANALYSIS
# ==========================================
elif page == "2. Interactive Analysis" and all_fires is not None:
    st.title("Interactive GIS Lab")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Layer Toggles")
    show_k = st.sidebar.checkbox("Soil Erodibility (K-Factor Heatmap)", value=False)
    show_recovery = st.sidebar.checkbox("Burn Severity (dNBR)", value=True)
    show_risk = st.sidebar.checkbox("Hazard Intersection (Orange)", value=True)
    
    run_analysis = st.toggle("Activate Spatial Modeling Engine", value=False)

    if run_analysis:
        with st.spinner("Analyzing multispectral and soil datasets..."):
            area = ee.FeatureCollection(fire_subset.__geo_interface__)
            pre_date = ee.Date(manual_baseline.strftime('%Y-%m-%d'))
            fire_start = ee.Date(default_alarm_dt.strftime('%Y-%m-%d'))
            target_date = fire_start.advance(recovery_months, 'month')

            def get_nbr(d): return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area).filterDate(d.advance(-2, 'month'), d.advance(2, 'month')).median().clip(area).normalizedDifference(['B8', 'B12'])
            dnbr = get_nbr(pre_date).subtract(get_nbr(target_date))
            slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003").clip(area))
            
            # Soil K-Factor Extraction
            soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0').clip(area)
            k_factor = soil.remap([1,2,3,4,5,6,7,8,9,10,11,12], [15,25,15,30,35,20,30,40,25,45,10,5]).divide(100.0)
            
            hazard_mask = slope.gte(slope_limit).And(dnbr.gt(dnbr_limit))
            
            raw_hazard = hazard_mask.multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), area.geometry(), 250).getInfo().get('nd', 0)
            hazard_acres = float(raw_hazard) * 0.000247105 if raw_hazard else 0.0

            st.metric("Total High Risk Area", f"{hazard_acres:,.1f} Acres")

            centroid = fire_subset.geometry.centroid.iloc[0]
            m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr="Google")
            
            if show_k:
                folium.TileLayer(tiles=k_factor.getMapId({'min': 0.1, 'max': 0.45, 'palette': ['#f7fcf0','#e0f3db','#ccebc5','#a8ddb5','#7bccc4','#4eb3d3','#2b8cbe','#08589e']})['tile_fetcher'].url_format, attr='OpenLandMap', name='Soil Erodibility', opacity=0.5).add_to(m)
            if show_recovery:
                folium.TileLayer(tiles=dnbr.updateMask(dnbr.gt(0.1)).getMapId({'min': 0.1, 'max': 0.5, 'palette': ['#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026']})['tile_fetcher'].url_format, attr='S2', name='Burn Severity').add_to(m)
            if show_risk:
                folium.TileLayer(tiles=hazard_mask.updateMask(hazard_mask).getMapId({'palette':['#ff7b00']})['tile_fetcher'].url_format, attr='GEE', name='Hazard Zones').add_to(m)

            st_folium(m, use_container_width=True, height=600)

# ==========================================
# PAGE 3: STATISTICAL REPORT
# ==========================================
elif page == "3. Statistical Report" and all_fires is not None:
    st.title("Watershed Statistical Analysis")
    run_stats = st.toggle("Generate Regional Vulnerability Map & Report", value=False)

    if run_stats:
        with st.spinner("Calculating sediment yield and intersection metrics..."):
            area = ee.FeatureCollection(fire_subset.__geo_interface__)
            pre_date = ee.Date(manual_baseline.strftime('%Y-%m-%d'))
            fire_start = ee.Date(default_alarm_dt.strftime('%Y-%m-%d'))
            target_date = fire_start.advance(recovery_months, 'month')

            def get_nbr(d): return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area).filterDate(d.advance(-2, 'month'), d.advance(2, 'month')).median().clip(area).normalizedDifference(['B8', 'B12'])
            dnbr = get_nbr(pre_date).subtract(get_nbr(target_date))
            slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003").clip(area))
            soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0').clip(area)
            k_factor = soil.remap([1,2,3,4,5,6,7,8,9,10,11,12], [15,25,15,30,35,20,30,40,25,45,10,5]).divide(100.0).rename('k_factor')
            
            hazard_mask = slope.gte(slope_limit).And(dnbr.gt(dnbr_limit)).multiply(ee.Image.pixelArea()).rename('hazard_area')
            precip = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterBounds(area).filterDate(target_date.advance(-1, 'month'), target_date).select('precipitation').sum().rename('rainfall')
            
            combined = hazard_mask.addBands(precip).addBands(k_factor)
            huc12 = ee.FeatureCollection("USGS/WBD/2017/HUC12").filterBounds(area.geometry())
            
            stats = combined.reduceRegions(collection=huc12, reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), sharedInputs=True), scale=500).getInfo()
            
            ws_data = []
            for f in stats['features']:
                p = f['properties']
                h_acres = (p.get('hazard_area_sum', 0) or 0) * 0.000247105
                rain = p.get('rainfall_mean', 0) or 0
                k = p.get('k_factor_mean', 0.25) or 0.25
                vol = (p.get('hazard_area_sum', 0) or 0) * (rain/1000.0) * k
                if h_acres > 0.1:
                    ws_data.append({"Watershed": p.get('name', 'Unknown'), "Hazard Area (Ac)": round(h_acres,1), "Rain (mm)": round(rain,1), "K-Factor": round(k,3), "Est Yield (m3)": round(vol,1)})
            
            if ws_data:
                df = pd.DataFrame(ws_data).sort_values(by="Est Yield (m3)", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.altair_chart(alt.Chart(df).mark_bar(color='#ff7b00').encode(x='Est Yield (m3):Q', y=alt.Y('Watershed:N', sort='-x')), use_container_width=True)
            else: st.info("No hazards detected within selected watersheds.")
