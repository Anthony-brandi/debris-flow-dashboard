import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import ee
import json
from datetime import datetime, timedelta

# ==========================================
# 1. DATA SOURCES & SETUP
# ==========================================
st.set_page_config(page_title="Wildfire Recovery & SGMA Analysis", layout="wide")

# CA DWR Bulletin 118 URL
GW_BASINS_URL = "https://opendata.arcgis.com/datasets/5da310c66bc649f0bc4ad21820b36873_0.geojson"

@st.cache_data
def load_fire_perimeters():
    path = 'CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view/CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view.shp'
    fires = gpd.read_file(path)
    return fires.dissolve(by='incident_n').reset_index().to_crs(epsg=4326)

# ==========================================
# 2. GEE INITIALIZATION
# ==========================================
if 'ee_initialized' not in st.session_state:
    try:
        if "EARTHENGINE_JSON" in st.secrets:
            creds_dict = json.loads(st.secrets["EARTHENGINE_JSON"])
            credentials = ee.ServiceAccountCredentials(creds_dict['client_email'], key_data=st.secrets["EARTHENGINE_JSON"])
            ee.Initialize(credentials, project='gee-streamlit-app-490500')
        else:
            ee.Initialize(project='gee-streamlit-app-490500')
        st.session_state['ee_initialized'] = True
    except Exception as e:
        st.error(f"GEE Initialization Error: {e}")

# ==========================================
# 3. INTERACTIVE DASHBOARD
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Interactive Risk Map", "Technical Documentation"])

if page == "Interactive Risk Map":
    try:
        cal_fires = load_fire_perimeters()
        fire_list = sorted(cal_fires['incident_n'].dropna().unique())
        selected_fire = st.sidebar.selectbox("Select Wildfire Perimeter", fire_list)
        fire_data = cal_fires[cal_fires['incident_n'] == selected_fire]
        
        # Recovery period selection
        st.sidebar.markdown("---")
        st.sidebar.subheader("Temporal Analysis")
        recovery_months = st.sidebar.select_slider("Post-Fire Interval (Months)", options=[1, 6, 12, 18, 24], value=1)
        
        analyze_btn = st.sidebar.checkbox("Execute Spatial Analysis", value=False)
        slope_limit = st.sidebar.slider("Slope Threshold (Degrees)", 10, 45, 27)

        # Map Layers
        with st.sidebar.expander("Layer Visibility"):
            show_recovery = st.checkbox("Burn Severity (dNBR)", value=True)
            show_basins = st.checkbox("Groundwater Basin Boundaries", value=True)
            show_risk = st.checkbox("Critical Hazard Intersection", value=False)
            basemap = st.radio("Basemap Style", ["Satellite", "Terrain"])
            tile_url = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}' if basemap == "Satellite" else 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}'

        st.title(f"{selected_fire} Debris Flow & Recovery Analysis")
        
        # Map Setup
        centroid = fire_data.geometry.centroid.iloc[0]
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles=tile_url, attr="Google")
        folium.GeoJson(fire_data.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent'}).add_to(m)

        if analyze_btn:
            with st.spinner("Processing geospatial data..."):
                area = ee.FeatureCollection(fire_data.__geo_interface__)
                
                # --- DNBR CALCULATION ---
                fire_date = pd.to_datetime(fire_data['ALARM_DATE'].iloc[0]) if 'ALARM_DATE' in fire_data.columns else datetime(2021,6,1)
                pre_date = ee.Date(fire_date.strftime('%Y-%m-%d')).advance(-1, 'year')
                post_date = ee.Date(fire_date.strftime('%Y-%m-%d')).advance(recovery_months, 'month')

                def get_nbr(date):
                    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area)\
                        .filterDate(date.advance(-1,'month'), date.advance(1,'month'))\
                        .median().clip(area).normalizedDifference(['B8', 'B12'])

                dnbr = get_nbr(pre_date).subtract(get_nbr(target_date))

                # --- STATISTICAL ANALYSIS ---
                # 1. Total Acreage
                total_area_m2 = fire_data.to_crs(epsg=3310).area.sum()
                total_acres = total_area_m2 * 0.000247105

                # 2. Burn Severity Class Distribution (Acreage)
                # Thresholds: High (>0.44), Moderate (0.1 - 0.44), Low/Unburned (<0.1)
                high_sev = dnbr.gt(0.44).multiply(ee.Image.pixelArea()).reduceRegion(reducer=ee.Reducer.sum(), geometry=area.geometry(), scale=30).getInfo().get('nd', 0)
                mod_sev = dnbr.gt(0.1).And(dnbr.lte(0.44)).multiply(ee.Image.pixelArea()).reduceRegion(reducer=ee.Reducer.sum(), geometry=area.geometry(), scale=30).getInfo().get('nd', 0)
                
                high_acres = high_sev * 0.000247105
                mod_acres = mod_sev * 0.000247105

                # 3. Slope Analysis
                dem = ee.Image("USGS/SRTMGL1_003")
                slope = ee.Terrain.slope(dem).clip(area)
                steep_area = slope.gte(slope_limit).multiply(ee.Image.pixelArea()).reduceRegion(reducer=ee.Reducer.sum(), geometry=area.geometry(), scale=30).getInfo().get('slope', 0)
                steep_acres = steep_area * 0.000247105

                # --- SIDEBAR STATISTICS PANEL ---
                st.sidebar.markdown("---")
                st.sidebar.subheader("Quantitative Results")
                st.sidebar.write(f"Total Area: {total_acres:,.1f} ac")
                
                col1, col2 = st.sidebar.columns(2)
                col1.metric("High Severity", f"{high_acres:,.1f} ac")
                col2.metric("Steep Terrain", f"{steep_acres:,.1f} ac")
                
                # --- LAYER RENDERING ---
                if show_basins:
                    # FIX: Filter GeoJSON to specific fire area to avoid HTTP 400 error
                    basins = gpd.read_file(GW_BASINS_URL)
                    local_basins = basins[basins.intersects(fire_data.geometry.iloc[0])]
                    folium.GeoJson(local_basins, name="Groundwater Basins", 
                                   style_function=lambda x: {'fillColor': '#3498db', 'color': 'blue', 'weight': 1, 'fillOpacity': 0.1},
                                   tooltip=folium.GeoJsonTooltip(fields=['Basin_Name'])).add_to(m)

                if show_recovery:
                    dnbr_vis = {'min': -0.1, 'max': 0.5, 'palette': ['ffffff', '7ad071', 'f9e072', 'ff0000']}
                    dnbr_id = dnbr.getMapId(dnbr_vis)
                    folium.TileLayer(tiles=dnbr_id['tile_fetcher'].url_format, attr='GEE', name='dNBR Severity', opacity=0.6).add_to(m)

                if show_risk:
                    hazard = slope.gte(slope_limit).And(dnbr.gt(0.1))
                    h_id = hazard.updateMask(hazard).getMapId({'palette': ['#ff7b00']})
                    folium.TileLayer(tiles=h_id['tile_fetcher'].url_format, attr='GEE', name='Hazard Intersection').add_to(m)

        st_folium(m, use_container_width=True, height=750, key="map_main")

    except Exception as e:
        st.error(f"Analysis Error: {e}")

elif page == "Technical Documentation":
    st.title("Scientific Methodology")
    st.markdown("---")
    st.write("Quantitative analysis is performed using Sentinel-2 Multi-Spectral Instrument (MSI) data and SRTM Digital Elevation Models.")
