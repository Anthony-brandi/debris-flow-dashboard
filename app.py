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
st.set_page_config(page_title="Watershed Risk Portal", layout="wide")

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
        date_options = ['ALARM_DATE', 'ALARM_DAT', 'START_DATE', 'alarm_date', 'alarm_dat']
        found_col = next((col for col in date_options if col in fires.columns), None)
        fires['final_date'] = pd.to_datetime(fires[found_col], errors='coerce') if found_col else pd.to_datetime('2021-06-01')
        fires = fires.dissolve(by='incident_n').reset_index()
        return fires.to_crs(epsg=4326)
    except Exception as e:
        st.error("Shapefile not found. Please ensure the 'CA_Perimeters' folder is in the same directory.")
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
        st.error(f"GEE Init Error: {e}")

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Risk Portal")
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
    st.header(f"Incident Brief: {selected_name}")
    st.markdown("---")
    
    m1, m2, m3, m4 = st.columns(4)
    total_acres = (fire_subset.to_crs(epsg=3310).area.sum()) * 0.000247105
    
    m1.metric("Recorded Ignition", default_alarm_dt.strftime('%b %d, %Y'))
    m2.metric("Total Perimeter", f"{total_acres:,.1f} Ac")
    m3.metric("Lead Agency", fire_subset['agency'].iloc[0] if 'agency' in fire_subset.columns else "CAL FIRE")
    m4.metric("Status", "Contained / In Monitoring")

    st.markdown("---")

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
        
        This portal models the ensuing 'funnel' effect, mapping the specific initiation zones and stream valleys where post-fire debris flows are most likely to originate.
        """)
        st.success("Proceed to '2. Interactive Analysis' in the sidebar to run the spatial intersection models.")

# ==========================================
# PAGE 2: INTERACTIVE ANALYSIS
# ==========================================
elif page == "2. Interactive Analysis" and all_fires is not None:
    st.title("Interactive GIS Lab")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Parameters")
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
    show_infra = st.sidebar.checkbox("Road Vulnerability", value=True)
    
    st.markdown("---")
    run_analysis = st.toggle("Activate Spatial Modeling Engine", value=False)

    if run_analysis:
        with st.spinner("Processing multispectral and topographic data..."):
            area = ee.FeatureCollection(fire_subset.__geo_interface__)
            
            pre_date = ee.Date(manual_baseline.strftime('%Y-%m-%d'))
            fire_start_ee = ee.Date(default_alarm_dt.strftime('%Y-%m-%d'))
            target_date = fire_start_ee.advance(recovery_months, 'month')

            def get_nbr_median(date_obj):
                return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area).filterDate(date_obj.advance(-1, 'month'), date_obj.advance(1, 'month')).median().clip(area).normalizedDifference(['B8', 'B12'])

            dnbr = get_nbr_median(pre_date).subtract(get_nbr_median(target_date))
            dem = ee.Image("USGS/SRTMGL1_003").clip(area)
            slope = ee.Terrain.slope(dem)
            hillshade = ee.Terrain.hillshade(dem)

            hazard_mask = slope.gte(slope_limit).And(dnbr.gt(0.44))
            
            # --- AUTOMATED PEAK RESULTS ---
            precip = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterBounds(area).filterDate(target_date.advance(-1, 'month'), target_date).select('precipitation').sum().clip(area)
            peak_rain = precip.reduceRegion(ee.Reducer.max(), area.geometry(), 1000).getInfo().get('precipitation', 0)
            hazard_acres = hazard_mask.multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), area.geometry(), 30).getInfo().get('nd', 0) * 0.000247105
            
            st.subheader("Automated Model Insights")
            m1, m2, m3 = st.columns(3)
            m1.metric("Active Hazard Area", f"{hazard_acres:,.1f} Acres", delta="Critical Zones", delta_color="inverse")
            m2.metric("Peak Rainfall Intensity", f"{peak_rain:,.1f} mm", delta="During Window", delta_color="off")
            m3.metric("Geomorphic Threshold", f"{slope_limit}°", delta="User Defined", delta_color="off")
            st.markdown("---")

            # --- MAP RENDERING ---
            centroid = fire_subset.geometry.centroid.iloc[0]
            m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr="Google")
            
            # Refined Professional Legend
            legend_html = f"""
            <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; background-color: white; border:1px solid grey; z-index:9999; font-size:13px; padding: 12px; border-radius: 4px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
            <b style="color:#2c3e50; font-size:14px;">Spatial Layers</b><br><hr style="margin: 4px 0;">
            <i style="background:red; width:12px; height:12px; float:left; margin-right:8px; border:1px solid black;"></i> <span style="color:black;">Fire Perimeter</span><br>
            <i style="background:#bd0026; width:12px; height:12px; float:left; margin-right:8px;"></i> <span style="color:black;">Severe Burn (dNBR)</span><br>
            <i style="background:#ff7b00; width:12px; height:12px; float:left; margin-right:8px;"></i> <span style="color:black;">Hazard Initiation Zone</span><br>
            <i style="background:#3498db; width:12px; height:3px; float:left; margin-right:8px; margin-top:5px;"></i> <span style="color:black;">HydroSHEDS Streams</span><br>
            <i style="background:#2ecc71; width:12px; height:3px; float:left; margin-right:8px; margin-top:5px;"></i> <span style="color:black;">Vulnerable Roads</span>
            </div>"""
            m.get_root().html.add_child(folium.Element(legend_html))
            
            if show_hillshade:
                folium.TileLayer(tiles=hillshade.getMapId({'min': 0, 'max': 255, 'palette': ['000000', 'ffffff']})['tile_fetcher'].url_format, attr='USGS', name='3D Hillshade', opacity=0.6).add_to(m)
            if show_recovery:
                folium.TileLayer(tiles=dnbr.updateMask(dnbr.gt(0.1)).getMapId({'min': 0.1, 'max': 0.5, 'palette': ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']})['tile_fetcher'].url_format, attr='S2', name='Burn Status', opacity=0.6).add_to(m)
            if show_precip:
                folium.TileLayer(tiles=precip.updateMask(precip.gt(1)).getMapId({'min': 1, 'max': 150, 'palette': ['#f7fbff','#deebf7','#9ecae1','#4292c6','#084594']})['tile_fetcher'].url_format, attr='NASA', name='Rainfall', opacity=0.5).add_to(m)
            if show_risk:
                folium.TileLayer(tiles=hazard_mask.updateMask(hazard_mask).getMapId({'palette':['#ff7b00']})['tile_fetcher'].url_format, attr='GEE', name='Hazard Zones').add_to(m)
            if show_streams:
                streams = ee.Image(0).mask(0).paint(ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(area), '#3498db', 2)
                folium.TileLayer(tiles=streams.getMapId({'palette':['#3498db']})['tile_fetcher'].url_format, attr='HydroSHEDS', name='Streams').add_to(m)
            if show_infra:
                roads = ee.Image(0).mask(0).paint(ee.FeatureCollection("TIGER/2016/Roads").filterBounds(area), '#2ecc71', 1.5)
                # Apply mask to only show roads IN hazard zones for better context
                vulnerable_roads = roads.updateMask(hazard_mask)
                folium.TileLayer(tiles=vulnerable_roads.getMapId({'palette':['#2ecc71']})['tile_fetcher'].url_format, attr='TIGER', name='Roads').add_to(m)

            folium.GeoJson(fire_subset.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 3}).add_to(m)
            st_folium(m, use_container_width=True, height=650)
    else:
        st.info("Toggle 'Activate Spatial Modeling Engine' above to calculate and render the active map layers.")

# ==========================================
# PAGE 3: STATISTICAL REPORT
# ==========================================
elif page == "3. Statistical Report" and all_fires is not None:
    st.title("Watershed Statistical Analysis")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Report Parameters")
    recovery_months = st.sidebar.select_slider("Successional Window (Months)", options=[1, 6, 12, 18, 24], value=1)
    slope_limit = st.sidebar.slider("Critical Slope Threshold (°)", 10, 45, 27)
    
    run_stats = st.toggle("Generate Quantitative Report", value=False)

    if run_stats:
        with st.spinner("Extracting multivariate data across HUC-12 boundaries..."):
            area = ee.FeatureCollection(fire_subset.__geo_interface__)
            
            pre_date = ee.Date(default_alarm_dt.strftime('%Y-%m-%d')).advance(-1, 'year')
            fire_start_ee = ee.Date(default_alarm_dt.strftime('%Y-%m-%d'))
            target_date = fire_start_ee.advance(recovery_months, 'month')

            def get_nbr_median(date_obj):
                return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area).filterDate(date_obj.advance(-1, 'month'), date_obj.advance(1, 'month')).median().clip(area).normalizedDifference(['B8', 'B12'])

            dnbr = get_nbr_median(pre_date).subtract(get_nbr_median(target_date))
            slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).clip(area)
            hazard_mask = slope.gte(slope_limit).And(dnbr.gt(0.44))
            
            # Setup Images for Reduction
            hazard_area_img = hazard_mask.multiply(ee.Image.pixelArea()).rename('hazard_area')
            precip_img = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterBounds(area).filterDate(target_date.advance(-1, 'month'), target_date).select('precipitation').sum().rename('rain')
            
            # Road Length Approximation
            roads = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(area)
            road_pixels = ee.Image(0).mask(0).paint(roads, 1).updateMask(hazard_mask).rename('road_risk')

            # Combine bands for a single reduction pass (efficiency)
            combined_img = hazard_area_img.addBands(precip_img).addBands(road_pixels)
            huc12 = ee.FeatureCollection("USGS/WBD/2017/HUC12").filterBounds(area)
            
            def calc_multivariate(feature):
                stats = combined_img.reduceRegion(
                    reducer=ee.Reducer.sum().combine(reducer2=ee.Reducer.mean(), sharedInputs=False),
                    geometry=feature.geometry(), 
                    scale=30, 
                    maxPixels=1e9
                )
                return feature.set('stats', stats)

            huc12_stats = huc12.map(calc_multivariate).getInfo()
            
            ws_data = []
            for f in huc12_stats['features']:
                props = f['properties']
                stats_dict = props.get('stats', {})
                if stats_dict is None: stats_dict = {}
                
                # Parse Stats
                raw_sq_meters = stats_dict.get('hazard_area_sum', 0)
                acres = raw_sq_meters * 0.000247105 if raw_sq_meters else 0
                rain_mm = stats_dict.get('rain_mean', 0)
                
                # Approximate road length (pixels * 30m resolution / 1609 to get miles)
                road_pix_count = stats_dict.get('road_risk_sum', 0)
                road_miles = (road_pix_count * 30) / 1609.34 if road_pix_count else 0
                
                ws_data.append({
                    "HUC-12 Watershed": props.get('name', 'Unknown'), 
                    "Hazard Area (Acres)": round(acres, 2),
                    "Avg Rainfall (mm)": round(rain_mm, 1) if rain_mm else 0,
                    "Roads at Risk (Miles)": round(road_miles, 2)
                })
            
            df_ws = pd.DataFrame(ws_data).sort_values(by="Hazard Area (Acres)", ascending=False)
            df_ws = df_ws[df_ws["Hazard Area (Acres)"] > 0]

            st.subheader("Multivariate Decision Matrix")
            st.dataframe(df_ws, use_container_width=True, hide_index=True)
            
            c1, c2 = st.columns(2)
            with c1:
                csv = df_ws.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Decision Matrix (CSV)",
                    data=csv,
                    file_name=f'{selected_name}_decision_matrix.csv',
                    mime='text/csv',
                )
                
            with c2:
                if not df_ws.empty:
                    chart = alt.Chart(df_ws).mark_bar(color='#bd0026').encode(
                        x=alt.X('Hazard Area (Acres):Q', title='Hazard Area (Acres)'),
                        y=alt.Y('HUC-12 Watershed:N', sort='-x', title=None),
                        tooltip=['HUC-12 Watershed', 'Hazard Area (Acres)', 'Roads at Risk (Miles)']
                    ).properties(height=300)
                    st.altair_chart(chart, use_container_width=True)

            st.markdown("---")
            st.info("""
            **Methodological Note:** This matrix utilizes spatial reduction to quantify 'The Deadly Combination' per sub-watershed. Furthermore, it overlays the TIGER/Line network to approximate the specific mileage of infrastructure trapped within those active hazard zones, providing actionable intelligence for evacuation and resource staging.
            """)
    else:
        st.info("Toggle 'Generate Quantitative Report' to calculate the spatial metrics for this specific timeframe.")
