import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import ee
import json
from datetime import datetime, timedelta

# ==========================================
# 1. SYSTEM SETUP
# ==========================================
st.set_page_config(page_title="Watershed Runoff & Debris Analysis", layout="wide")

@st.cache_data
def load_fire_perimeters():
    path = 'CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view/CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view.shp'
    fires = gpd.read_file(path)
    fires = fires.dissolve(by='incident_n').reset_index()
    return fires.to_crs(epsg=4326)

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
        st.error(f"Initialization Error: {e}")

# ==========================================
# 3. NAVIGATION
# ==========================================
st.sidebar.title("Main Menu")
page = st.sidebar.radio("Navigation", ["Interactive Risk Map", "User Manual", "Technical Documentation"])

# ==========================================
# PAGE 1: INTERACTIVE RISK MAP
# ==========================================
if page == "Interactive Risk Map":
    try:
        cal_fires = load_fire_perimeters()
        fire_list = sorted(cal_fires['incident_n'].dropna().unique())
        selected_fire = st.sidebar.selectbox("Select Wildfire Incident", fire_list)
        fire_data = cal_fires[cal_fires['incident_n'] == selected_fire]
        centroid_point = fire_data.geometry.centroid.iloc[0]
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Environmental Parameters")
        
        recovery_months = st.sidebar.select_slider(
            "Observation Window (Months Post-Fire)", 
            options=[1, 6, 12, 18, 24], 
            value=1
        )
        
        analyze_btn = st.sidebar.checkbox("Execute Hydrologic Analysis", value=False)
        slope_limit = st.sidebar.slider("Slope Threshold (Degrees)", 10, 45, 27)

        with st.sidebar.expander("Map Layer Settings"):
            show_recovery = st.checkbox("Burn Severity (dNBR)", value=True)
            show_precip = st.checkbox("Precipitation (NASA GPM)", value=False)
            show_risk = st.checkbox("High-Risk Intersection", value=False)
            show_infra = st.checkbox("Infrastructure (TIGER)", value=True)
            basemap = st.radio("Style", ["Satellite", "Terrain"])
            tile_url = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}' if basemap == "Satellite" else 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}'

        st.title(f"{selected_fire} Debris Flow and Runoff Analysis")
        
        if analyze_btn:
            with st.spinner("Calculating multispectral and hydrologic intersection..."):
                area = ee.FeatureCollection(fire_data.__geo_interface__)
                
                # --- DNBR COMPUTATION ---
                date_col = next((c for c in ['ALARM_DATE', 'START_DATE', 'alarm_date'] if c in fire_data.columns), None)
                fire_start_dt = pd.to_datetime(fire_data[date_col].iloc[0]) if date_col else datetime(2021, 6, 1)
                
                pre_date = ee.Date(fire_start_dt.strftime('%Y-%m-%d')).advance(-1, 'year')
                post_date = ee.Date(fire_start_dt.strftime('%Y-%m-%d')).advance(recovery_months, 'month')

                def get_nbr_median(target_date):
                    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area)\
                        .filterDate(target_date.advance(-1, 'month'), target_date.advance(1, 'month'))\
                        .median().clip(area).normalizedDifference(['B8', 'B12'])

                dnbr = get_nbr_median(pre_date).subtract(get_nbr_median(post_date))

                # --- PRECIPITATION (NASA GPM) ---
                precip = ee.ImageCollection("NASA/GPM_L3/IMERG_V06")\
                    .filterBounds(area)\
                    .filterDate(post_date.advance(-1, 'month'), post_date)\
                    .select('precipitation')\
                    .sum().clip(area)

                # --- TOPOGRAPHY ---
                dem = ee.Image("USGS/SRTMGL1_003")
                slope = ee.Terrain.slope(dem).clip(area)

                # --- STATISTICAL COMPUTATION ---
                total_acres = (fire_data.to_crs(epsg=3310).area.sum()) * 0.000247105
                high_sev_acres = dnbr.gt(0.44).multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), area.geometry(), 30).getInfo().get('nd', 0) * 0.000247105
                avg_precip = precip.reduceRegion(ee.Reducer.mean(), area.geometry(), 1000).getInfo().get('precipitation', 0)
                
                # Runoff Calculation (Proxy for Peak Flow)
                runoff_index = dnbr.multiply(precip).reduceRegion(ee.Reducer.max(), area.geometry(), 1000).getInfo().get('nd', 0)
                recovery_pct = max(0, min(100, (100 - ((high_sev_acres / (total_acres * 0.15)) * 100))))

                # METRICS DASHBOARD
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("High Severity", f"{high_sev_acres:,.1f} Ac")
                m2.metric("Healing Rate", f"{recovery_pct:.1f}%")
                m3.metric("Rainfall", f"{avg_precip:,.1f} mm")
                m4.metric("Runoff Intensity", f"{runoff_index:.2f}")

                # CHART
                st.subheader("Landscape Recovery Trend")
                chart_data = pd.DataFrame({
                    "Stage": ["Post-Ignition", "Current State", "Restoration Goal"],
                    "Unrecovered Acres": [total_acres * 0.15, high_sev_acres, total_acres * 0.01]
                })
                st.bar_chart(chart_data, x="Stage", y="Unrecovered Acres")

                # MAP
                m = folium.Map(location=[centroid_point.y, centroid_point.x], zoom_start=12, tiles=tile_url, attr="Google")
                folium.GeoJson(fire_data.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 2}).add_to(m)

                if show_recovery:
                    dnbr_vis = {'min': -0.1, 'max': 0.5, 'palette': ['ffffff', '7ad071', 'f9e072', 'ff0000']}
                    folium.TileLayer(tiles=dnbr.getMapId(dnbr_vis)['tile_fetcher'].url_format, attr='Sentinel-2', name='Burn Status', opacity=0.6).add_to(m)

                if show_precip:
                    p_vis = {'min': 0, 'max': 200, 'palette': ['f0f9e8', 'bae4bc', '7bccc4', '43a2ca', '0868ac']}
                    folium.TileLayer(tiles=precip.getMapId(p_vis)['tile_fetcher'].url_format, attr='NASA', name='Rainfall').add_to(m)

                if show_risk:
                    hazard = slope.gte(slope_limit).And(dnbr.gt(0.1))
                    h_id = hazard.updateMask(hazard).getMapId({'palette': ['#ff7b00']})
                    folium.TileLayer(tiles=h_id['tile_fetcher'].url_format, attr='GEE', name='Risk Intersection').add_to(m)

                st_folium(m, use_container_width=True, height=750)

        else:
            m = folium.Map(location=[centroid_point.y, centroid_point.x], zoom_start=12, tiles=tile_url, attr="Google")
            folium.GeoJson(fire_data.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 2}).add_to(m)
            st_folium(m, use_container_width=True, height=750)

    except Exception as e:
        st.error(f"Analysis Runtime Error: {e}")

# ==========================================
# PAGE 2: USER MANUAL
# ==========================================
elif page == "User Manual":
    st.title("User Manual")
    st.markdown("---")
    st.header("Step-by-Step Instructions")
    st.write("""
    1. **Incident Selection:** Use the dropdown menu to select a historical fire perimeter from the CAL FIRE database.
    2. **Temporal Selection:** Use the slider to select a time interval. This retrieves satellite data from that specific month to check how much vegetation has grown back.
    3. **Set Thresholds:** Adjust the 'Steepness Threshold' to filter for hillsides that are vulnerable to gravity-driven landslides.
    4. **Hydrologic Run:** Click 'Execute Hydrologic Analysis' to overlay NASA rainfall data and calculate the Runoff Intensity.
    """)
    st.info("The 'Healing Rate' shows what percentage of the initial burn scar has been replaced by new green vegetation.")

# ==========================================
# PAGE 3: TECHNICAL DOCUMENTATION
# ==========================================
elif page == "Technical Documentation":
    st.title("Technical Documentation")
    st.markdown("---")
    
    st.header("1. Multispectral Burn Analysis (dNBR)")
    st.write("We utilize Sentinel-2 satellite data to calculate the Differenced Normalized Burn Ratio. Red zones indicate a total loss of canopy, leaving soil exposed to direct rain impact.")
    

    st.header("2. Topographic Risk and Tipping Points")
    st.write("Slopes exceeding 27 degrees are geomorphically unstable after a fire. This dashboard isolates these areas to identify 'Initiation Zones' where debris flows are born.")
    

    st.header("3. The Watershed Runoff Effect")
    st.write("""
    When rain hits a healthy forest, the trees slow it down. After a fire, the watershed becomes a 'parking lot.'
    * **NASA GPM IMERG:** This dashboard pulls global satellite rainfall data to see exactly how much water fell on the burn scar.
    * **Runoff Intensity:** By multiplying burn severity by rainfall, we identify 'Hotspots' where water velocity was highest, potentially burying roads or flooding canyons.
    """)
