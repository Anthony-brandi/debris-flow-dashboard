import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import ee
import json
from datetime import datetime, timedelta

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(page_title="Watershed Runoff & Debris Analysis", layout="wide")

@st.cache_data
def load_fire_perimeters():
    path = 'CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view/CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view.shp'
    fires = gpd.read_file(path)
    
    # Flexible Date Searching to fix "ALARM_DATE" Error
    date_options = ['ALARM_DATE', 'ALARM_DAT', 'START_DATE', 'alarm_date', 'alarm_dat']
    found_col = next((col for col in date_options if col in fires.columns), None)
    
    if found_col:
        fires['final_date'] = pd.to_datetime(fires[found_col], errors='coerce')
    else:
        # Fallback if no date column is found
        fires['final_date'] = pd.to_datetime('2021-06-01')
        
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

if page == "Interactive Risk Map":
    try:
        cal_fires = load_fire_perimeters()
        fire_list = sorted(cal_fires['incident_n'].dropna().unique())
        selected_fire = st.sidebar.selectbox("Select Wildfire Incident", fire_list)
        fire_data = cal_fires[cal_fires['incident_n'] == selected_fire]
        
        # Display the found date in the sidebar
        actual_date = fire_data['final_date'].iloc[0]
        st.sidebar.info(f"Analysis Baseline Date: {actual_date.strftime('%B %d, %Y')}")
        
        centroid_point = fire_data.geometry.centroid.iloc[0]
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Environmental Parameters")
        recovery_months = st.sidebar.select_slider("Observation Window (Months Post-Fire)", options=[1, 6, 12, 18, 24], value=1)
        analyze_btn = st.sidebar.checkbox("Execute Hydrologic Analysis", value=True)
        slope_limit = st.sidebar.slider("Slope Threshold (Degrees)", 10, 45, 27)

        # --- RESULTS STATISTICS AT THE TOP ---
        st.title(f"{selected_fire} Runoff Hazard Assessment")
        
        if analyze_btn:
            with st.spinner("Calculating multispectral and rainfall totals..."):
                area = ee.FeatureCollection(fire_data.__geo_interface__)
                
                # Use automated dates
                pre_date = ee.Date(actual_date.strftime('%Y-%m-%d')).advance(-1, 'year')
                target_date = ee.Date(actual_date.strftime('%Y-%m-%d')).advance(recovery_months, 'month')

                def get_nbr_median(date_obj):
                    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area)\
                        .filterDate(date_obj.advance(-1, 'month'), date_obj.advance(1, 'month'))\
                        .median().clip(area).normalizedDifference(['B8', 'B12'])

                dnbr = get_nbr_median(pre_date).subtract(get_nbr_median(target_date))
                precip = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterBounds(area)\
                    .filterDate(target_date.advance(-1, 'month'), target_date)\
                    .select('precipitation').sum().clip(area)

                # CALCULATE METRICS
                total_acres = (fire_data.to_crs(epsg=3310).area.sum()) * 0.000247105
                high_sev_acres = dnbr.gt(0.44).multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), area.geometry(), 30).getInfo().get('nd', 0) * 0.000247105
                avg_precip = precip.reduceRegion(ee.Reducer.mean(), area.geometry(), 1000).getInfo().get('precipitation', 0)
                recovery_pct = max(0, min(100, (100 - ((high_sev_acres / (total_acres * 0.15)) * 100))))

                # METRICS DASHBOARD
                m1, m2, m3 = st.columns(3)
                m1.metric("Unrecovered Burn Scar", f"{high_sev_acres:,.1f} Ac")
                m2.metric("Healing Rate", f"{recovery_pct:.1f}%")
                m3.metric("Rainfall (NASA GPM)", f"{avg_precip:,.1f} mm")
                st.markdown("---")

                # --- MAP ---
                m = folium.Map(location=[centroid_point.y, centroid_point.x], zoom_start=12, tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr="Google")
                
                # Burn Severity mask
                vis = {'min': 0.1, 'max': 0.5, 'palette': ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']}
                dnbr_masked = dnbr.updateMask(dnbr.gt(0.1))
                folium.TileLayer(tiles=dnbr_masked.getMapId(vis)['tile_fetcher'].url_format, attr='S2', name='Burn Status', opacity=0.7).add_to(m)

                folium.GeoJson(fire_data.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 3}).add_to(m)
                st_folium(m, use_container_width=True, height=650)

    except Exception as e:
        st.error(f"Analysis Error: {e}")

# ==========================================
# 4. DOCUMENTATION PAGES
# ==========================================
elif page == "User Manual":
    st.title("User Manual")
    st.info("The Green lines represent Roads (TIGER/Line). These are clipped specifically to the fire boundary to show which transportation corridors are vulnerable.")
    st.write("1. Select an incident.\n2. Adjust your slope and time sliders.\n3. Run analysis to identify hazard zones.")

elif page == "Technical Documentation":
    st.title("Technical Documentation")
    st.write("Clipped Infrastructure: We use the 2016 TIGER/Line dataset and perform a spatial filter against the CAL FIRE incident perimeter.")
