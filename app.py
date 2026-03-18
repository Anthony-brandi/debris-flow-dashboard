import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import ee
import json
from datetime import datetime, timedelta
import altair as alt

# ==========================================
# 1. SYSTEM CONFIGURATION & UI
# ==========================================
st.set_page_config(page_title="Watershed Risk Portal", layout="wide", page_icon="⛰️")

# Professional GIS Dark Mode Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 5px; border: 1px solid #374151; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADERS & GEE INIT
# ==========================================
@st.cache_data
def load_fire_perimeters():
    path = 'CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view/CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view.shp'
    try:
        fires = gpd.read_file(path)
        # Robust date detection
        date_options = ['ALARM_DATE', 'ALARM_DAT', 'START_DATE', 'alarm_date', 'alarm_dat']
        found_col = next((col for col in date_options if col in fires.columns), None)
        fires['final_date'] = pd.to_datetime(fires[found_col], errors='coerce') if found_col else pd.to_datetime('2021-06-01')
        fires = fires.dissolve(by='incident_n').reset_index()
        return fires.to_crs(epsg=4326)
    except Exception as e:
        st.error("Shapefile not found. Please ensure the 'CA_Perimeters' folder is in the same directory as this script.")
        return None

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
        st.error(f"Google Earth Engine Initialization Error: {e}")

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🛡️ Risk Portal")
page = st.sidebar.selectbox("Select View", ["1. Incident Briefing", "2. Interactive Analysis", "3. Statistical Report"])

all_fires = load_fire_perimeters()
if all_fires is not None:
    fire_names = sorted(all_fires['incident_n'].dropna().unique())
    selected_name = st.sidebar.selectbox("Choose Wildfire Incident", fire_names)
    fire_subset = all_fires[all_fires['incident_n'] == selected_name]
    default_alarm_dt = fire_subset['final_date'].iloc[0]

# ==========================================
# PAGE 1: INCIDENT BRIEFING
# ==========================================
if page == "1. Incident Briefing" and all_fires is not None:
    st.header(f"🔥 Incident Brief: {selected_name}")
    st.markdown("---")
    
    # Metadata Row
    m1, m2, m3, m4 = st.columns(4)
    total_acres = (fire_subset.to_crs(epsg=3310).area.sum()) * 0.000247105
    
    m1.metric("Recorded Ignition", default_alarm_dt.strftime('%b %d, %Y'))
    m2.metric("Total Perimeter", f"{total_acres:,.1f} Ac")
    m3.metric("Lead Agency", fire_subset['agency'].iloc[0] if 'agency' in fire_subset.columns else "CAL FIRE")
    m4.metric("Status", "Contained / In Monitoring")

    st.markdown("---")

    # Context & Simple Map
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Perimeter Overview")
        centroid = fire_subset.geometry.centroid.iloc[0]
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles='CartoDB dark_matter')
        folium.GeoJson(fire_subset.geometry, style_function=lambda x: {'color': 'red', 'fillColor': '#bd0026', 'weight': 2, 'fillOpacity': 0.4}).add_to(m)
        st_folium(m, use_container_width=True, height=500)

    with col2:
        st.subheader("Geomorphic Context")
        st.info(f"""
        **The Transition to Funnel:**
        The {selected_name} fire altered the hydrologic baseline of this region. 
        When high-severity canopy loss intersects with steep topography, the landscape loses its ability to act as a biological 'sponge.' 
        
        This portal models the ensuing 'funnel' effect, mapping the specific initiation zones and stream valleys where post-fire debris flows are most likely to originate during heavy precipitation events.
        """)
        st.success("Proceed to '2. Interactive Analysis' to run the spatial intersection models.")

# ==========================================
# PAGE 2: INTERACTIVE ANALYSIS
# ==========================================
elif page == "2. Interactive Analysis" and all_fires is not None:
    st.title("🛠️ Interactive GIS Lab")
    
    # Manual Baseline & Parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Parameters")
    
    # Allow user to change the baseline date
    manual_baseline = st.sidebar.date_input("Pre-Fire Baseline Date", value=default_alarm_dt - timedelta(days=365))
    recovery_months = st.sidebar.select_slider("Successional Window (Months Post-Fire)", options=[1, 6, 12, 18, 24], value=1)
    slope_limit = st.sidebar.slider("Critical Slope Threshold (°)", 10, 45, 27)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Layer Toggles")
    show_hillshade = st.sidebar.checkbox("3D Topographic Hillshade", value=True)
    show_recovery = st.sidebar.checkbox("Burn Severity (dNBR)", value=True)
    show_precip = st.sidebar.checkbox("Precipitation (NASA GPM)", value=False)
    show_risk = st.sidebar.checkbox("Hazard Intersection (Orange)", value=True)
    show_streams = st.sidebar.checkbox("Stream Routing (HydroSHEDS)", value=True)
    show_infra = st.sidebar.checkbox("Road Vulnerability", value=False)
    
    analyze_btn = st.button("Execute Spatial Modeling")

    if analyze_btn:
        with st.spinner("Processing multispectral and topographic data..."):
            area = ee.FeatureCollection(fire_subset.__geo_interface__)
            
            # Dates
            pre_date = ee.Date(manual_baseline.strftime('%Y-%m-%d'))
            fire_start_ee = ee.Date(default_alarm_dt.strftime('%Y-%m-%d'))
            target_date = fire_start_ee.advance(recovery_months, 'month')

            # dNBR Calculation
            def get_nbr_median(date_obj):
                return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area)\
                    .filterDate(date_obj.advance(-1, 'month'), date_obj.advance(1, 'month'))\
                    .median().clip(area).normalizedDifference(['B8', 'B12'])

            dnbr = get_nbr_median(pre_date).subtract(get_nbr_median(target_date))
            
            # Topography & Hillshade (3D Representation)
            dem = ee.Image("USGS/SRTMGL1_003").clip(area)
            slope = ee.Terrain.slope(dem)
            hillshade = ee.Terrain.hillshade(dem)

            # Hazard Logic
            hazard_mask = slope.gte(slope_limit).And(dnbr.gt(0.44))

            # Map Rendering
            centroid = fire_subset.geometry.centroid.iloc[0]
            m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr="Google")
            
            if show_hillshade:
                folium.TileLayer(tiles=hillshade.getMapId({'min': 0, 'max': 255, 'palette': ['000000', 'ffffff']})['tile_fetcher'].url_format, attr='USGS', name='3D Hillshade', opacity=0.6).add_to(m)
            if show_recovery:
                folium.TileLayer(tiles=dnbr.updateMask(dnbr.gt(0.1)).getMapId({'min': 0.1, 'max': 0.5, 'palette': ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']})['tile_fetcher'].url_format, attr='S2', name='Burn Status', opacity=0.6).add_to(m)
            if show_precip:
                precip = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterBounds(area).filterDate(target_date.advance(-1, 'month'), target_date).select('precipitation').sum().clip(area)
                folium.TileLayer(tiles=precip.updateMask(precip.gt(1)).getMapId({'min': 1, 'max': 150, 'palette': ['#f7fbff','#deebf7','#9ecae1','#4292c6','#084594']})['tile_fetcher'].url_format, attr='NASA', name='Rainfall', opacity=0.5).add_to(m)
            if show_risk:
                folium.TileLayer(tiles=hazard_mask.updateMask(hazard_mask).getMapId({'palette':['#ff7b00']})['tile_fetcher'].url_format, attr='GEE', name='Hazard Zones').add_to(m)
            if show_streams:
                streams = ee.Image(0).mask(0).paint(ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(area), '#3498db', 2)
                folium.TileLayer(tiles=streams.getMapId({'palette':['#3498db']})['tile_fetcher'].url_format, attr='HydroSHEDS', name='Streams').add_to(m)
            if show_infra:
                roads = ee.Image(0).mask(0).paint(ee.FeatureCollection("TIGER/2016/Roads").filterBounds(area), '#2ecc71', 1.5)
                folium.TileLayer(tiles=roads.getMapId({'palette':['#2ecc71']})['tile_fetcher'].url_format, attr='TIGER', name='Roads').add_to(m)

            folium.GeoJson(fire_subset.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 3}).add_to(m)
            st_folium(m, use_container_width=True, height=650)
    else:
        st.info("Adjust your parameters in the sidebar and click 'Execute Spatial Modeling' to load the GIS layers.")

# ==========================================
# PAGE 3: STATISTICAL REPORT
# ==========================================
elif page == "3. Statistical Report" and all_fires is not None:
    st.title("📊 Watershed Statistical Analysis")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Report Parameters")
    recovery_months = st.sidebar.select_slider("Successional Window (Months)", options=[1, 6, 12, 18, 24], value=1)
    slope_limit = st.sidebar.slider("Critical Slope Threshold (°)", 10, 45, 27)
    run_stats = st.button("Generate Quantitative Report")

    if run_stats:
        with st.spinner("Reducing spatial data across HUC-12 boundaries..."):
            area = ee.FeatureCollection(fire_subset.__geo_interface__)
            
            pre_date = ee.Date(default_alarm_dt.strftime('%Y-%m-%d')).advance(-1, 'year')
            fire_start_ee = ee.Date(default_alarm_dt.strftime('%Y-%m-%d'))
            target_date = fire_start_ee.advance(recovery_months, 'month')

            # Math Logic
            def get_nbr_median(date_obj):
                return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area).filterDate(date_obj.advance(-1, 'month'), date_obj.advance(1, 'month')).median().clip(area).normalizedDifference(['B8', 'B12'])

            dnbr = get_nbr_median(pre_date).subtract(get_nbr_median(target_date))
            slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).clip(area)
            hazard_mask = slope.gte(slope_limit).And(dnbr.gt(0.44))
            hazard_area_img = hazard_mask.multiply(ee.Image.pixelArea())

            # Watershed Reduction
            huc12 = ee.FeatureCollection("USGS/WBD/2017/HUC12").filterBounds(area)
            
            def calc_hazard(feature):
                stats = hazard_area_img.reduceRegion(reducer=ee.Reducer.sum(), geometry=feature.geometry(), scale=30, maxPixels=1e9)
                return feature.set('hazard_acres', ee.Number(stats.get('slope')).multiply(0.000247105))

            huc12_stats = huc12.map(calc_hazard).getInfo()
            
            ws_data = []
            for f in huc12_stats['features']:
                props = f['properties']
                ws_data.append({
                    "HUC-12 Watershed Name": props.get('name', 'Unknown'), 
                    "Active Hazard Footprint (Acres)": round(props.get('hazard_acres', 0), 2)
                })
            
            df_ws = pd.DataFrame(ws_data).sort_values(by="Active Hazard Footprint (Acres)", ascending=False)
            df_ws = df_ws[df_ws["Active Hazard Footprint (Acres)"] > 0] # Filter out empty watersheds

            # Layout the Report
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Regional Vulnerability Table")
                st.dataframe(df_ws, use_container_width=True, hide_index=True)
                
                # CSV Export
                csv = df_ws.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=csv,
                    file_name=f'{selected_name}_watershed_risk_month_{recovery_months}.csv',
                    mime='text/csv',
                )
                
            with c2:
                st.subheader("Hazard Distribution by Basin")
                chart = alt.Chart(df_ws).mark_bar(color='#ff7b00').encode(
                    x=alt.X('Active Hazard Footprint (Acres):Q', title='Hazard Area (Acres)'),
                    y=alt.Y('HUC-12 Watershed Name:N', sort='-x', title=None),
                    tooltip=['HUC-12 Watershed Name', 'Active Hazard Footprint (Acres)']
                ).properties(height=400)
                st.altair_chart(chart, use_container_width=True)

            st.markdown("---")
            st.info("""
            **Methodological Note:** This report utilizes the `reduceRegion` function to quantify the total acreage of 'The Deadly Combination' (Burn > 0.44 AND Slope > Threshold) within each USGS HUC-12 boundary. This output is designed to align with the sub-watershed targeting logic utilized by the USGS Post-Fire Debris Flow (PFDF) and Wildcat models.
            """)
    else:
        st.info("Click 'Generate Quantitative Report' to calculate the spatial metrics for this specific timeframe.")
