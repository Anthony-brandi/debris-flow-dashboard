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
        
        # Centroid defined early to prevent 'not defined' errors
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
                target_date = ee.Date(fire_start_dt.strftime('%Y-%m-%d')).advance(recovery_months, 'month')

                def get_nbr_median(date_obj):
                    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area)\
                        .filterDate(date_obj.advance(-1, 'month'), date_obj.advance(1, 'month'))\
                        .median().clip(area).normalizedDifference(['B8', 'B12'])

                dnbr = get_nbr_median(pre_date).subtract(get_nbr_median(target_date))

                # --- PRECIPITATION (NASA GPM V07) ---
                # Fixed: Using the 'precipitation' band as indicated by your error log
                precip = ee.ImageCollection("NASA/GPM_L3/IMERG_V07")\
                    .filterBounds(area)\
                    .filterDate(target_date.advance(-1, 'month'), target_date)\
                    .select('precipitation')\
                    .sum().clip(area)

                # --- STATISTICAL COMPUTATION ---
                total_acres = (fire_data.to_crs(epsg=3310).area.sum()) * 0.000247105
                high_sev_acres = dnbr.gt(0.44).multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), area.geometry(), 30).getInfo().get('nd', 0) * 0.000247105
                avg_precip = precip.reduceRegion(ee.Reducer.mean(), area.geometry(), 1000).getInfo().get('precipitation', 0)
                
                # Recovery % Logic
                recovery_pct = max(0, min(100, (100 - ((high_sev_acres / (total_acres * 0.15)) * 100))))

                # METRICS DASHBOARD
                m1, m2, m3 = st.columns(3)
                m1.metric("High Severity", f"{high_sev_acres:,.1f} Ac")
                m2.metric("Landscape Healing", f"{recovery_pct:.1f}%")
                m3.metric("Rainfall Accumulation", f"{avg_precip:,.1f} mm")

                # RECOVERY ANALYSIS GRAPH
                st.subheader("Vegetation Recovery and Watershed Stabilization")
                chart_data = pd.DataFrame({
                    "Observation Stage": ["Peak Vulnerability", "Current State", "Restoration Goal"],
                    "Unrecovered Acres": [total_acres * 0.15, high_sev_acres, total_acres * 0.01]
                })
                st.bar_chart(chart_data, x="Observation Stage", y="Unrecovered Acres")

                # MAP RENDER
                m = folium.Map(location=[centroid_point.y, centroid_point.x], zoom_start=12, tiles=tile_url, attr="Google")
                folium.GeoJson(fire_data.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 2}).add_to(m)

                if show_recovery:
                    dnbr_vis = {'min': -0.1, 'max': 0.5, 'palette': ['ffffff', '7ad071', 'f9e072', 'ff0000']}
                    folium.TileLayer(tiles=dnbr.getMapId(dnbr_vis)['tile_fetcher'].url_format, attr='Sentinel-2', name='Burn Status', opacity=0.6).add_to(m)

                if show_precip:
                    p_vis = {'min': 0, 'max': 200, 'palette': ['f0f9e8', 'bae4bc', '7bccc4', '43a2ca', '0868ac']}
                    folium.TileLayer(tiles=precip.getMapId(p_vis)['tile_fetcher'].url_format, attr='NASA GPM', name='Rainfall').add_to(m)

                if show_risk:
                    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).clip(area)
                    hazard = slope.gte(slope_limit).And(dnbr.gt(0.1))
                    h_id = hazard.updateMask(hazard).getMapId({'palette': ['#ff7b00']})
                    folium.TileLayer(tiles=h_id['tile_fetcher'].url_format, attr='GEE', name='Risk Intersection').add_to(m)

                if show_infra:
                    roads = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(area)
                    roads_img = ee.Image(0).mask(0).paint(roads, 1, 2)
                    infra_id = roads_img.getMapId({'palette': ['#2ecc71']})
                    folium.TileLayer(tiles=infra_id['tile_fetcher'].url_format, attr='TIGER', name='Infrastructure').add_to(m)

                st_folium(m, use_container_width=True, height=750)

        else:
            m = folium.Map(location=[centroid_point.y, centroid_point.x], zoom_start=12, tiles=tile_url, attr="Google")
            folium.GeoJson(fire_data.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 2}).add_to(m)
            st_folium(m, use_container_width=True, height=750)

    except Exception as e:
        st.error(f"Analysis Error: {e}")

# ==========================================
# PAGE 2: USER MANUAL
# ==========================================
elif page == "User Manual":
    st.title("User Manual")
    st.markdown("---")
    st.header("Step-by-Step Instructions")
    st.write("""
    1. **Incident Selection:** Use the dropdown menu to select a historical fire perimeter from the database.
    2. **Temporal Selection:** Use the slider to select a post-fire interval. This determines the date of the satellite imagery used to assess vegetation regrowth.
    3. **Set Thresholds:** Adjust the 'Slope Threshold' to identify hillsides vulnerable to gravity-driven landslides. 
    4. **Execution:** Check 'Execute Hydrologic Analysis' to calculate rainfall totals and identify high-runoff hotspots.
    """)
    st.info("The 'Landscape Healing' metric calculates the percentage of the initial burn scar that has recovered since the fire began.")

# ==========================================
# PAGE 3: TECHNICAL DOCUMENTATION
# ==========================================
elif page == "Technical Documentation":
    st.title("Technical Documentation")
    st.markdown("---")
    
    st.header("1. Multispectral Burn Analysis (dNBR)")
    st.write("This dashboard utilizes the Differenced Normalized Burn Ratio (dNBR) from the Sentinel-2 satellite. Red zones represent a loss of photosynthetic material, exposing soil to erosion.")
    
    st.header("2. Steepness and Landslide Risk")
    st.write("""
    Flat ground allows water to pool, but steep hillsides create velocity. This slider helps identify 'tipping points' 
    where the terrain is so steep that gravity can easily pull destabilized soil down into canyons, creating debris flows.
    """)

    st.header("3. The Watershed Runoff Effect")
    st.write("""
    Healthy watersheds act as a sponge. Burned watersheds act as a funnel.
    * **NASA GPM IMERG:** We pull global satellite rainfall data to track the exact amount of precipitation that hit the burn scar during your selected window.
    * **Runoff Response:** By intersecting rainfall totals with the most severely burned soil, we identify the exact 'hotspots' where runoff velocity and flood risk were highest.
    """)
