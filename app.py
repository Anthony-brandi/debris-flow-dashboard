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
        # Check for the zip file in the root directory
        possible_names = ['Master_Fire_Dataset.zip', 'Master_Fire_Dataset.geojson.zip']
        actual_zip = next((name for name in possible_names if os.path.exists(name)), None)
        
        if not actual_zip:
            st.error("CRITICAL ERROR: Master_Fire_Dataset.zip not found in GitHub. Please ensure the file is uploaded to the main folder.")
            return None

        with zipfile.ZipFile(actual_zip, 'r') as zip_ref:
            # Extract the GeoJSON filename from the zip content
            geojson_filename = [f for f in zip_ref.namelist() if f.endswith('.geojson')][0]
            zip_ref.extract(geojson_filename)
            
        fires = gpd.read_file(geojson_filename)
        # Ensure date column is properly formatted
        fires['final_date'] = pd.to_datetime(fires['final_date'], errors='coerce')
        return fires
    except Exception as e:
        st.error(f"Failed to load shapes: {e}")
        return None

@st.cache_data
def fetch_dins_damage(incident_name):
    url = "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/ArcGIS/rest/services/DINS_Public_View/FeatureServer/0/query"
    params = {"where": f"UPPER(INCIDENT_NAME) LIKE '%{str(incident_name).upper()}%'", "returnCountOnly": "true", "f": "json"}
    try:
        response = requests.get(url, params=params, timeout=5).json()
        return response.get('count', 0)
    except:
        return 0

# ==========================================
# 3. GEE INITIALIZATION
# ==========================================
if 'ee_initialized' not in st.session_state:
    try:
        if "EARTHENGINE_JSON" in st.secrets:
            creds = json.loads(st.secrets["EARTHENGINE_JSON"])
            ee.Initialize(ee.ServiceAccountCredentials(creds['client_email'], key_data=st.secrets["EARTHENGINE_JSON"]), project='gee-streamlit-app-490500')
        else:
            ee.Initialize(project='gee-streamlit-app-490500')
        st.session_state['ee_initialized'] = True
    except Exception as e:
        st.error(f"Google Earth Engine failed to initialize: {e}")

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
    
    # Handle missing or null dates gracefully
    try:
        raw_date = fire_subset['final_date'].iloc[0]
        default_alarm_dt = pd.to_datetime(raw_date) if pd.notnull(raw_date) else datetime(2021, 7, 1)
    except:
        default_alarm_dt = datetime(2021, 7, 1)

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
    
    with st.spinner("Calculating infrastructure and damage metrics..."):
        impacted_count = fetch_dins_damage(selected_name)
        total_acres = (fire_subset.to_crs(epsg=3310).area.sum()) * 0.000247105
        
        # Infrastructure Miles (TIGER)
        area_ee = ee.FeatureCollection(fire_subset.__geo_interface__)
        roads_fc = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(area_ee.geometry())
        road_miles = roads_fc.aggregate_sum('length').getInfo() / 1609.34 if roads_fc.size().getInfo() > 0 else 0
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recorded Ignition", default_alarm_dt.strftime('%b %d, %Y'))
        m2.metric("Total Perimeter", f"{total_acres:,.1f} Ac")
        m3.metric("Verified Damage", f"{impacted_count} Struct")
        m4.metric("Road Exposure", f"{road_miles:,.1f} Miles")

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Boundary Overview")
        centroid = fire_subset.geometry.centroid.iloc[0]
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles='CartoDB Positron')
        folium.GeoJson(fire_subset.geometry, style_function=lambda x: {'color': 'red', 'weight': 2, 'fillOpacity': 0.1}).add_to(m)
        st_folium(m, use_container_width=True, height=500)
    with col2:
        st.subheader("Geomorphic Context")
        st.info("Debris flows in California are typically supply-limited. This model identifies initiation zones where 'loose' soil (high K-factor) meets high-velocity terrain (steep slopes) and low infiltration capacity (burn scars).")

# ==========================================
# PAGE 2: INTERACTIVE ANALYSIS
# ==========================================
elif page == "2. Interactive Analysis" and all_fires is not None:
    st.title("Interactive GIS Lab")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Layer Toggles")
    show_k = st.sidebar.checkbox("Soil Erodibility (K-Factor Heatmap)", value=True)
    show_burn = st.sidebar.checkbox("Burn Severity (dNBR)", value=True)
    show_risk = st.sidebar.checkbox("Hazard Intersection (Orange)", value=True)
    show_hydro = st.sidebar.checkbox("Stream Networks", value=True)
    show_roads = st.sidebar.checkbox("Major Roads (TIGER)", value=True)
    
    run_analysis = st.toggle("Activate Spatial Modeling Engine", value=True)

    if run_analysis:
        with st.spinner("Processing satellite and topographic arrays..."):
            try:
                area_ee = ee.FeatureCollection(fire_subset.__geo_interface__)
                pre_date = ee.Date(manual_baseline.strftime('%Y-%m-%d'))
                target_date = ee.Date(default_alarm_dt.strftime('%Y-%m-%d')).advance(recovery_months, 'month')

                # Satellite Engine
                def get_nbr(d):
                    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area_ee).filterDate(d.advance(-3, 'month'), d.advance(3, 'month')).median().clip(area_ee).normalizedDifference(['B8', 'B12'])
                
                dnbr = get_nbr(pre_date).subtract(get_nbr(target_date))
                slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003").clip(area_ee))
                
                # Soil K-Factor Heatmap
                soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0').clip(area_ee)
                k_factor = soil.remap([1,2,3,4,5,6,7,8,9,10,11,12], [15,25,15,30,35,20,30,40,25,45,10,5]).divide(100.0)
                
                hazard_mask = slope.gte(slope_limit).And(dnbr.gt(dnbr_limit))
                streams = ee.Image(0).mask(0).paint(ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(area_ee), 1, 2)
                roads_ee = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(area_ee)

                centroid = fire_subset.geometry.centroid.iloc[0]
                m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr="Google Satellite")
                
                # Fixed Legend
                legend_html = f"""<div style="position: fixed; bottom: 50px; left: 50px; width: 220px; background-color: white; border:2px solid black; z-index:9999; font-size:12px; padding: 10px; border-radius: 5px;">
                <b>Map Legend</b><br>
                <i style="background:#ff7b00; width:12px; height:12px; float:left; margin-right:5px; border:1px solid black;"></i> Hazard Intersection<br>
                <i style="background:#bd0026; width:12px; height:12px; float:left; margin-right:5px;"></i> Severe Burn Area<br>
                <i style="background:#8c510a; width:12px; height:12px; float:left; margin-right:5px;"></i> High Soil Erodibility<br>
                <i style="background:#00d4ff; width:12px; height:3px; float:left; margin-right:5px; margin-top:5px;"></i> Stream Flow<br>
                <i style="background:white; border:1px solid black; width:12px; height:2px; float:left; margin-right:5px; margin-top:5px;"></i> Major Roads
                </div>"""
                m.get_root().html.add_child(folium.Element(legend_html))

                if show_k:
                    folium.TileLayer(tiles=k_factor.getMapId({'min': 0.1, 'max': 0.45, 'palette': ['#f6e8c3','#dfc27d','#bf812d','#8c510a']})['tile_fetcher'].url_format, attr='Soil', name='Soil Stability', opacity=0.5).add_to(m)
                if show_burn:
                    folium.TileLayer(tiles=dnbr.updateMask(dnbr.gt(0.1)).getMapId({'min': 0.1, 'max': 0.5, 'palette': ['#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026']})['tile_fetcher'].url_format, attr='S2', name='Burn Status', opacity=0.7).add_to(m)
                if show_risk:
                    folium.TileLayer(tiles=hazard_mask.updateMask(hazard_mask).getMapId({'palette':['#ff7b00']})['tile_fetcher'].url_format, attr='GEE', name='Hazard Intersection').add_to(m)
                if show_hydro:
                    folium.TileLayer(tiles=streams.getMapId({'palette':['#00d4ff']})['tile_fetcher'].url_format, attr='Hydro', name='Flow Paths').add_to(m)
                if show_roads:
                    folium.GeoJson(roads_ee.getInfo(), style_function=lambda x: {'color': 'white', 'weight': 1.5, 'opacity': 0.8}, name='Infrastructure').add_to(m)

                st_folium(m, use_container_width=True, height=750)
            except Exception as e:
                st.error(f"Geospatial rendering failed: {e}")

# ==========================================
# PAGE 3: STATISTICAL REPORT
# ==========================================
elif page == "3. Statistical Report" and all_fires is not None:
    st.title("Watershed Risk Matrix")
    run_stats = st.toggle("Generate Risk Map and Data", value=True)

    if run_stats:
        with st.spinner("Analyzing sub-watershed debris loading..."):
            try:
                area_ee = ee.FeatureCollection(fire_subset.__geo_interface__)
                target_date = ee.Date(default_alarm_dt.strftime('%Y-%m-%d')).advance(recovery_months, 'month')
                
                def get_nbr(d): return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area_ee).filterDate(d.advance(-3, 'month'), d.advance(3, 'month')).median().clip(area_ee).normalizedDifference(['B8', 'B12'])
                dnbr = get_nbr(ee.Date(manual_baseline.strftime('%Y-%m-%d'))).subtract(get_nbr(target_date))
                slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003").clip(area_ee))
                soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0').clip(area_ee)
                k_factor = soil.remap([1,2,3,4,5,6,7,8,9,10,11,12], [15,25,15,30,35,20,30,40,25,45,10,5]).divide(100.0).rename('k_factor')
                
                hazard_mask = slope.gte(slope_limit).And(dnbr.gt(dnbr_limit)).multiply(ee.Image.pixelArea()).rename('hazard_area')
                precip = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterBounds(area_ee).filterDate(target_date.advance(-1, 'month'), target_date).select('precipitation').sum().rename('rainfall')
                
                combined = hazard_mask.addBands(precip).addBands(k_factor)
                huc12 = ee.FeatureCollection("USGS/WBD/2017/HUC12").filterBounds(area_ee.geometry())
                stats = combined.reduceRegions(collection=huc12, reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), sharedInputs=True), scale=500).getInfo()
                
                ws_data = []
                for f in stats['features']:
                    p = f['properties']
                    h_sqm = p.get('hazard_area_sum', 0) or 0
                    h_acres = h_sqm * 0.000247105
                    k = p.get('k_factor_mean', 0.25) or 0.25
                    vol = h_sqm * ( (p.get('rainfall_mean', 0) or 0) / 1000.0) * k
                    if h_acres > 0.05:
                        ws_data.append({"Watershed": p.get('name', 'Unknown'), "Hazard (Ac)": round(h_acres,1), "Soil K": round(k,3), "Est Yield (m3)": round(vol,1)})

                if ws_data:
                    df = pd.DataFrame(ws_data).sort_values(by="Est Yield (m3)", ascending=False)
                    
                    st.subheader("Regional Vulnerability Map")
                    sel_ws = st.selectbox("Highlight Specific Watershed", ["None"] + df['Watershed'].tolist())
                    centroid = fire_subset.geometry.centroid.iloc[0]
                    m3 = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles='CartoDB Positron')
                    
                    folium.GeoJson(huc12.getInfo(), style_function=lambda x: {'color': 'purple', 'weight': 1, 'fillOpacity': 0}).add_to(m3)
                    if sel_ws != "None":
                        highlight = huc12.filter(ee.Filter.eq('name', sel_ws))
                        folium.GeoJson(highlight.getInfo(), style_function=lambda x: {'color': 'cyan', 'weight': 3, 'fillOpacity': 0.2}).add_to(m3)
                    
                    # Stream networks
                    streams_img = ee.Image(0).mask(0).paint(ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(area_ee), 1, 2)
                    folium.TileLayer(tiles=streams_img.getMapId({'palette':['#00d4ff']})['tile_fetcher'].url_format, attr='Hydro').add_to(m3)
                    st_folium(m3, use_container_width=True, height=500)
                    
                    st.markdown("---")
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.altair_chart(alt.Chart(df).mark_bar(color='#ff7b00').encode(x='Est Yield (m3):Q', y=alt.Y('Watershed:N', sort='-x')), use_container_width=True)
                else:
                    st.warning("Adjust thresholds in the sidebar to generate data for this perimeter.")
            except Exception as e:
                st.error(f"Statistical calculation failed: {e}")
