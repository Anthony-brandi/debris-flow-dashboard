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

# CA DWR Bulletin 118 Groundwater Basins URL
GW_BASINS_URL = "https://opendata.arcgis.com/datasets/5da310c66bc649f0bc4ad21820b36873_0.geojson"

@st.cache_data
def load_fire_perimeters():
    path = 'CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view/CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view.shp'
    fires = gpd.read_file(path)
    # Dissolve by incident name to consolidate multi-part geometries
    fires = fires.dissolve(by='incident_n').reset_index()
    return fires.to_crs(epsg=4326)

@st.cache_data
def load_filtered_basins(fire_geometry):
    # Pulls the full dataset but filters spatially to prevent browser memory errors (HTTP 400)
    all_basins = gpd.read_file(GW_BASINS_URL)
    return all_basins[all_basins.intersects(fire_geometry)]

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
st.sidebar.title("Analysis Control")
page = st.sidebar.radio("Navigation", ["Interactive Risk Map", "Technical Documentation"])

if page == "Interactive Risk Map":
    try:
        cal_fires = load_fire_perimeters()
        fire_list = sorted(cal_fires['incident_n'].dropna().unique())
        selected_fire = st.sidebar.selectbox("Select Wildfire Perimeter", fire_list)
        fire_data = cal_fires[cal_fires['incident_n'] == selected_fire]
        fire_geom = fire_data.geometry.iloc[0]
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Temporal and Topographic Parameters")
        recovery_months = st.sidebar.select_slider(
            "Post-Fire Interval (Months)", 
            options=[1, 6, 12, 18, 24], 
            value=1,
            help="Select the duration after ignition to evaluate vegetation regrowth and soil stability."
        )
        
        analyze_btn = st.sidebar.checkbox("Execute Spatial Analysis", value=False)
        slope_limit = st.sidebar.slider(
            "Slope Threshold (Degrees)", 
            10, 45, 27,
            help="Specify the minimum gradient for debris flow initiation. USGS standards often utilize 27 degrees."
        )

        with st.sidebar.expander("Layer Visibility Settings"):
            show_recovery = st.checkbox("Burn Severity (dNBR)", value=True)
            show_basins = st.checkbox("Groundwater Basin Boundaries", value=True)
            show_risk = st.checkbox("Critical Hazard Intersection", value=False)
            basemap = st.radio("Basemap Style", ["Satellite", "Terrain"])
            tile_url = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}' if basemap == "Satellite" else 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}'

        st.title(f"{selected_fire} Debris Flow Risk and Aquifer Recharge Analysis")
        
        # Map Initialization
        centroid = fire_data.geometry.centroid.iloc[0]
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles=tile_url, attr="Google")
        folium.GeoJson(fire_data.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 2}).add_to(m)

        if analyze_btn:
            with st.spinner("Processing geospatial datasets..."):
                area = ee.FeatureCollection(fire_data.__geo_interface__)
                
                # --- TEMPORAL DNBR CALCULATION ---
                # Attempt to parse fire start date from attributes
                date_col = next((c for c in ['ALARM_DATE', 'START_DATE', 'alarm_date'] if c in fire_data.columns), None)
                fire_start_dt = pd.to_datetime(fire_data[date_col].iloc[0]) if date_col else datetime(2021, 6, 1)
                
                pre_date = ee.Date(fire_start_dt.strftime('%Y-%m-%d')).advance(-1, 'year')
                post_date = ee.Date(fire_start_dt.strftime('%Y-%m-%d')).advance(recovery_months, 'month')

                def get_nbr_median(target_date):
                    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                        .filterBounds(area)\
                        .filterDate(target_date.advance(-1, 'month'), target_date.advance(1, 'month'))\
                        .median().clip(area).normalizedDifference(['B8', 'B12'])

                nbr_pre = get_nbr_median(pre_date)
                nbr_post = get_nbr_median(post_date)
                dnbr = nbr_pre.subtract(nbr_post)

                # --- TOPOGRAPHIC ANALYSIS ---
                dem = ee.Image("USGS/SRTMGL1_003")
                slope = ee.Terrain.slope(dem).clip(area)
                
                # Statistics Calculation
                total_acres = (fire_data.to_crs(epsg=3310).area.sum()) * 0.000247105
                
                # Calculate acreage of High Severity (dNBR > 0.44)
                high_sev_mask = dnbr.gt(0.44)
                high_acres = high_sev_mask.multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(), geometry=area.geometry(), scale=30
                ).getInfo().get('nd', 0) * 0.000247105

                # Calculate acreage of steep terrain
                steep_mask = slope.gte(slope_limit)
                steep_acres = steep_mask.multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(), geometry=area.geometry(), scale=30
                ).getInfo().get('slope', 0) * 0.000247105

                # Sidebar Metrics
                st.sidebar.markdown("---")
                st.sidebar.subheader("Statistical Summary")
                st.sidebar.write(f"Total Perimeter Area: {total_acres:,.1f} acres")
                st.sidebar.metric("High Severity Area", f"{high_acres:,.1f} ac")
                st.sidebar.metric("Steep Terrain Area", f"{steep_acres:,.1f} ac")

                # --- LAYER RENDERING ---
                if show_basins:
                    local_basins = load_filtered_basins(fire_geom)
                    folium.GeoJson(
                        local_basins, 
                        name="Groundwater Basins",
                        style_function=lambda x: {'fillColor': '#3498db', 'color': 'blue', 'weight': 1, 'fillOpacity': 0.15},
                        tooltip=folium.GeoJsonTooltip(fields=['Basin_Name'], aliases=['Basin Name:'])
                    ).add_to(m)

                if show_recovery:
                    dnbr_vis = {'min': -0.1, 'max': 0.5, 'palette': ['ffffff', '7ad071', 'f9e072', 'ff0000']}
                    dnbr_mapid = dnbr.getMapId(dnbr_vis)
                    folium.TileLayer(tiles=dnbr_mapid['tile_fetcher'].url_format, attr='Sentinel-2', name='Burn Severity', opacity=0.7).add_to(m)

                if show_risk:
                    hazard = steep_mask.And(dnbr.gt(0.1))
                    h_mapid = hazard.updateMask(hazard).getMapId({'palette': ['#ff7b00']})
                    folium.TileLayer(tiles=h_mapid['tile_fetcher'].url_format, attr='GEE', name='Hazard Intersection').add_to(m)

        st_folium(m, use_container_width=True, height=750, key="map_main")

    except Exception as e:
        st.error(f"Analysis Runtime Error: {e}")

# ==========================================
# 4. TECHNICAL DOCUMENTATION
# ==========================================
elif page == "Technical Documentation":
    st.title("Scientific Methodology and Policy Implications")
    st.markdown("---")
    
    st.header("1. Significance of the Slope Threshold")
    st.write("""
    The Slope Threshold slider defines the gravitational potential energy required to initiate a mass wasting event. 
    Post-fire debris flows typically originate in upland 'initiation zones' where gradients exceed the angle of repose for destabilized soil. 
    According to USGS standards, slopes greater than 27 degrees are considered high-risk corridors. Adjusting this slider 
    allows researchers to perform sensitivity analysis on different geomorphic settings.
    """)
    

    st.header("2. Temporal Analysis: The Post-Fire Interval")
    st.write("""
    The Post-Fire Interval slider accounts for two critical variables: Hydrophobicity and Vegetation Succession. 
    Immediately following a fire (1-6 months), soils often exhibit 'hydrophobic' properties, where ash and charred organic 
    matter form a water-repellent layer, drastically increasing runoff. 
    
    As the interval increases (12-24 months), secondary succession occurs as pioneer plant species establish root systems 
    that anchor the soil and increase the 'roughness' of the landscape, reducing velocity and increasing groundwater 
    infiltration. This dashboard uses multi-temporal Sentinel-2 imagery to monitor this recovery in real-time.
    """)
    

    st.header("3. Groundwater Recharge and SGMA")
    st.write("""
    Under the Sustainable Groundwater Management Act (SGMA), Groundwater Sustainability Agencies (GSAs) must maintain 
    the 'recharge potential' of their basins. Severely burned landscapes represent a temporary loss of recharge capacity. 
    By intersecting burn severity with groundwater basin boundaries, this tool identifies areas where restoration 
    is required to protect future water security and prevent subsidence.
    """)
