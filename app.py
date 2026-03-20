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
    except Exception as e: st.error(f"GEE Failure: {e}")

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Risk Portal Navigation")
page = st.sidebar.selectbox("Select View", ["1. Incident Briefing", "2. Interactive Analysis", "3. Statistical Report"])

all_fires = load_fire_perimeters()

if all_fires is not None:
    fire_names = sorted(all_fires['incident_n'].dropna().unique())
    selected_name = st.sidebar.selectbox("Choose Wildfire Incident", fire_names)
    fire_subset = all_fires[all_fires['incident_n'] == selected_name]
    
    try:
        default_alarm_dt = pd.to_datetime(fire_subset['final_date'].iloc[0])
    except:
        default_alarm_dt = datetime(2021, 1, 1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Parameters")
    manual_baseline = st.sidebar.date_input("Pre-Fire Baseline Date", value=default_alarm_dt - timedelta(days=365))
    recovery_months = st.sidebar.select_slider("Observation Window (Months)", options=[1, 6, 12, 18, 24], value=12)
    dnbr_limit = st.sidebar.slider("Burn Severity Threshold (dNBR)", 0.05, 0.70, 0.15)
    slope_limit = st.sidebar.slider("Critical Slope Threshold (Deg)", 5, 45, 20)

# ==========================================
# PAGE 1: INCIDENT BRIEFING
# ==========================================
if page == "1. Incident Briefing" and all_fires is not None:
    st.header(f"Incident Brief: {selected_name}")
    impacted_count = fetch_dins_damage(selected_name)
    total_acres = (fire_subset.to_crs(epsg=3310).area.sum()) * 0.000247105
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Recorded Ignition", default_alarm_dt.strftime('%b %d, %Y'))
    m2.metric("Total Perimeter", f"{total_acres:,.1f} Ac")
    m3.metric("Structures Impacted", f"{impacted_count}")

    col1, col2 = st.columns([2, 1])
    with col1:
        centroid = fire_subset.geometry.centroid.iloc[0]
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles='CartoDB Positron')
        folium.GeoJson(fire_subset.geometry, style_function=lambda x: {'color': 'red', 'weight': 2, 'fillOpacity': 0.1}).add_to(m)
        st_folium(m, use_container_width=True, height=500)
    with col2:
        st.subheader("Analysis Goals")
        st.info("This project models debris flow 'initiation zones' by identifying where high-severity fire intersects steep, erodible terrain.")

# ==========================================
# PAGE 2: INTERACTIVE ANALYSIS
# ==========================================
elif page == "2. Interactive Analysis" and all_fires is not None:
    st.title("Interactive GIS Lab")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Layer Toggles")
    show_k = st.sidebar.checkbox("Soil Erodibility (K-Factor)", value=True)
    show_burn = st.sidebar.checkbox("Burn Severity (dNBR)", value=True)
    show_risk = st.sidebar.checkbox("Hazard Intersection (Orange)", value=True)
    show_hydro = st.sidebar.checkbox("Stream Networks", value=True)
    show_roads = st.sidebar.checkbox("Major Roads (TIGER)", value=True)
    
    run_analysis = st.toggle("Activate Spatial Modeling Engine", value=True)

    if run_analysis:
        with st.spinner("Processing multispectral imagery..."):
            area = ee.FeatureCollection(fire_subset.__geo_interface__)
            pre_date = ee.Date(manual_baseline.strftime('%Y-%m-%d'))
            target_date = ee.Date(default_alarm_dt.strftime('%Y-%m-%d')).advance(recovery_months, 'month')

            def get_nbr(d): return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area).filterDate(d.advance(-3, 'month'), d.advance(3, 'month')).median().clip(area).normalizedDifference(['B8', 'B12'])
            dnbr = get_nbr(pre_date).subtract(get_nbr(target_date))
            slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003").clip(area))
            soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0').clip(area)
            k_factor = soil.remap([1,2,3,4,5,6,7,8,9,10,11,12], [15,25,15,30,35,20,30,40,25,45,10,5]).divide(100.0)
            hazard_mask = slope.gte(slope_limit).And(dnbr.gt(dnbr_limit))
            streams = ee.Image(0).mask(0).paint(ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(area), 1, 2)
            roads = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(area)

            centroid = fire_subset.geometry.centroid.iloc[0]
            m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr="Google")
            
            legend_html = f"""<div style="position: fixed; bottom: 50px; left: 50px; width: 220px; background-color: white; border:2px solid black; z-index:9999; font-size:12px; padding: 10px; border-radius: 5px;">
            <b>Map Legend</b><br>
            <i style="background:#ff7b00; width:12px; height:12px; float:left; margin-right:5px; border:1px solid black;"></i> Hazard Intersection<br>
            <i style="background:#bd0026; width:12px; height:12px; float:left; margin-right:5px;"></i> Severe Burn Area<br>
            <i style="background:#8c510a; width:12px; height:12px; float:left; margin-right:5px;"></i> High Soil Erodibility<br>
            <i style="background:#00d4ff; width:12px; height:3px; float:left; margin-right:5px; margin-top:5px;"></i> Streams<br>
            <i style="background:white; border:1px solid black; width:12px; height:2px; float:left; margin-right:5px; margin-top:5px;"></i> Roads (TIGER)
            </div>"""
            m.get_root().html.add_child(folium.Element(legend_html))

            if show_k: folium.TileLayer(tiles=k_factor.getMapId({'min': 0.1, 'max': 0.45, 'palette': ['#f6e8c3','#dfc27d','#bf812d','#8c510a']})['tile_fetcher'].url_format, attr='Soil', opacity=0.4).add_to(m)
            if show_burn: folium.TileLayer(tiles=dnbr.updateMask(dnbr.gt(0.1)).getMapId({'min': 0.1, 'max': 0.5, 'palette': ['#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026']})['tile_fetcher'].url_format, attr='S2', opacity=0.6).add_to(m)
            if show_risk: folium.TileLayer(tiles=hazard_mask.updateMask(hazard_mask).getMapId({'palette':['#ff7b00']})['tile_fetcher'].url_format, attr='GEE').add_to(m)
            if show_hydro: folium.TileLayer(tiles=streams.getMapId({'palette':['#00d4ff']})['tile_fetcher'].url_format, attr='Hydro').add_to(m)
            if show_roads: folium.GeoJson(roads.getInfo(), style_function=lambda x: {'color': 'white', 'weight': 1.5}).add_to(m)
            st_folium(m, use_container_width=True, height=700)

# ==========================================
# PAGE 3: STATISTICAL REPORT (RESTRUCTURED)
# ==========================================
elif page == "3. Statistical Report" and all_fires is not None:
    st.title("Watershed Risk Matrix")
    run_stats = st.toggle("Generate Risk Map and Data", value=True)

    if run_stats:
        with st.spinner("Mapping sub-watershed debris loading..."):
            area = ee.FeatureCollection(fire_subset.__geo_interface__)
            target_date = ee.Date(default_alarm_dt.strftime('%Y-%m-%d')).advance(recovery_months, 'month')
            
            # Logic
            def get_nbr(d): return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area).filterDate(d.advance(-3, 'month'), d.advance(3, 'month')).median().clip(area).normalizedDifference(['B8', 'B12'])
            dnbr = get_nbr(ee.Date(manual_baseline.strftime('%Y-%m-%d'))).subtract(get_nbr(target_date))
            slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003").clip(area))
            soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0').clip(area)
            k_factor = soil.remap([1,2,3,4,5,6,7,8,9,10,11,12], [15,25,15,30,35,20,30,40,25,45,10,5]).divide(100.0).rename('k_factor')
            hazard_area = slope.gte(slope_limit).And(dnbr.gt(dnbr_limit)).multiply(ee.Image.pixelArea()).rename('hazard_area')
            precip = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterBounds(area).filterDate(target_date.advance(-1, 'month'), target_date).select('precipitation').sum().rename('rainfall')
            
            combined = hazard_area.addBands(precip).addBands(k_factor)
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
                    ws_data.append({"Watershed": p.get('name', 'Unknown'), "Hazard (Ac)": round(h_acres,1), "Soil K": round(k,3), "Rain (mm)": round(rain,1), "Est Yield (m3)": round(vol,1)})

            if ws_data:
                df = pd.DataFrame(ws_data).sort_values(by="Est Yield (m3)", ascending=False)
                
                # PAGE 3 MAP
                st.subheader("Regional Loading Map")
                st.markdown("Select a watershed from the table to highlight the specific sub-basin and its drainage channels.")
                
                sel_ws = st.selectbox("Highlight Watershed", ["None"] + df['Watershed'].tolist())
                centroid = fire_subset.geometry.centroid.iloc[0]
                m3 = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles='CartoDB Positron')
                
                # Add HUC12 outlines
                folium.GeoJson(huc12.getInfo(), style_function=lambda x: {'color': 'purple', 'weight': 1, 'fillOpacity': 0}).add_to(m3)
                
                if sel_ws != "None":
                    highlight = huc12.filter(ee.Filter.eq('name', sel_ws))
                    folium.GeoJson(highlight.getInfo(), style_function=lambda x: {'color': 'cyan', 'weight': 3, 'fillOpacity': 0.2}).add_to(m3)
                
                # Overlay stream paths on top of the watershed map
                streams_img = ee.Image(0).mask(0).paint(ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(area), 1, 2)
                folium.TileLayer(tiles=streams_img.getMapId({'palette':['#00d4ff']})['tile_fetcher'].url_format, attr='Hydro', name='Streams').add_to(m3)
                
                st_folium(m3, use_container_width=True, height=500)
                
                st.markdown("---")
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.write("Regional Debris Flow Trigger Matrix")
                    st.dataframe(df, use_container_width=True, hide_index=True)
                with c2:
                    st.write("Sediment Mobilization Risk")
                    st.altair_chart(alt.Chart(df).mark_bar(color='#ff7b00').encode(x='Est Yield (m3):Q', y=alt.Y('Watershed:N', sort='-x')), use_container_width=True)
            else: st.warning("No watersheds found meeting the hazard threshold.")
