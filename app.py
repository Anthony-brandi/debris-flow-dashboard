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

@st.cache_data
def fetch_dins_damage(incident_name):
    url = "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/ArcGIS/rest/services/DINS_Public_View/FeatureServer/0/query"
    clean_name = str(incident_name).strip().upper()
    
    params = {
        "where": f"UPPER(INCIDENT_NAME) LIKE '%{clean_name}%' AND DAMAGE IN ('Destroyed', 'Major')",
        "outFields": "*",
        "returnCountOnly": "true",
        "f": "json"
    }
    try:
        response = requests.get(url, params=params, timeout=10).json()
        return response.get('count', 0)
    except:
        return 0

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
# 3. GLOBAL SIDEBAR NAVIGATION & PARAMETERS
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
    st.sidebar.subheader("Global Model Parameters")
    manual_baseline = st.sidebar.date_input("Pre-Fire Baseline Date", value=default_alarm_dt - timedelta(days=365))
    recovery_months = st.sidebar.select_slider("Observation Window (Months Post-Fire)", options=[1, 6, 12, 18, 24], value=1)
    
    dnbr_limit = st.sidebar.slider("Burn Severity Threshold (dNBR)", 0.10, 0.70, 0.25, 0.05)
    slope_limit = st.sidebar.slider("Critical Slope Threshold (Degrees)", 10, 45, 27)

# ==========================================
# PAGE 1: INCIDENT BRIEFING
# ==========================================
if page == "1. Incident Briefing" and all_fires is not None:
    st.header(f"Incident Brief: {selected_name}")
    st.markdown("---")
    
    destroyed_count = fetch_dins_damage(selected_name)
    total_acres = (fire_subset.to_crs(epsg=3310).area.sum()) * 0.000247105
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recorded Ignition", default_alarm_dt.strftime('%b %d, %Y'))
    m2.metric("Total Perimeter", f"{total_acres:,.1f} Ac")
    m3.metric("Lead Agency", fire_subset['agency'].iloc[0] if 'agency' in fire_subset.columns else "CAL FIRE")
    
    if destroyed_count > 0:
        m4.metric("Structures Destroyed", f"{destroyed_count}")
    else:
        m4.metric("Structures Destroyed", "0 (Or Missing DINS Data)")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Perimeter Overview")
        centroid = fire_subset.geometry.centroid.iloc[0]
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles='CartoDB dark_matter')
        folium.GeoJson(fire_subset.geometry, style_function=lambda x: {'color': 'red', 'fillColor': '#bd0026', 'weight': 2, 'fillOpacity': 0.4}).add_to(m)
        st_folium(m, use_container_width=True, height=500)

    with col2:
        st.subheader("Historical Fire Fact Sheet")
        
        garbage_cols = ['objectid', 'globalid', 'shape', 'geometry', 'incident_1', 'poly_datec', 'creationda', 'creator', 'editdate', 'editor', 'shape_leng', 'shape_area', 'irwinid']
        valid_data = {}
        for col in fire_subset.columns:
            if str(col).lower() not in garbage_cols:
                val = fire_subset[col].iloc[0]
                if pd.notna(val) and val != '' and val != 0:
                    clean_name = str(col).replace('_', ' ').title()
                    valid_data[clean_name] = str(val)
        
        if valid_data:
            df_facts = pd.DataFrame(list(valid_data.items()), columns=['Parameter', 'Recorded Data'])
            st.dataframe(df_facts, use_container_width=True, hide_index=True)
        else:
            st.info("No extended metadata found in the shapefile.")
            
        st.markdown("---")
        st.info(f"""
        **Geomorphic Context:**
        The {selected_name} fire altered the hydrologic baseline of this region. 
        When high-severity canopy loss intersects with steep topography, the landscape loses its ability to act as a biological sponge. 
        """)

# ==========================================
# PAGE 2: INTERACTIVE ANALYSIS
# ==========================================
elif page == "2. Interactive Analysis" and all_fires is not None:
    st.title("Interactive GIS Lab")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Layer Toggles")
    show_hillshade = st.sidebar.checkbox("3D Topographic Hillshade", value=True)
    show_recovery = st.sidebar.checkbox("Burn Severity (dNBR)", value=True)
    show_precip = st.sidebar.checkbox("Precipitation (NASA GPM)", value=False)
    show_risk = st.sidebar.checkbox("Hazard Intersection (Orange)", value=True)
    show_streams = st.sidebar.checkbox("Stream Routing (HydroSHEDS)", value=True)
    
    run_analysis = st.toggle("Activate Spatial Modeling Engine", value=False)

    if run_analysis:
        with st.spinner("Processing multispectral and topographic data..."):
            try:
                area = ee.FeatureCollection(fire_subset.__geo_interface__)
                
                pre_date = ee.Date(manual_baseline.strftime('%Y-%m-%d'))
                fire_start_ee = ee.Date(default_alarm_dt.strftime('%Y-%m-%d'))
                target_date = fire_start_ee.advance(recovery_months, 'month')

                def get_nbr_median(date_obj):
                    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area).filterDate(date_obj.advance(-2.5, 'month'), date_obj.advance(2.5, 'month')).median().clip(area).normalizedDifference(['B8', 'B12'])

                dnbr = get_nbr_median(pre_date).subtract(get_nbr_median(target_date))
                dem = ee.Image("USGS/SRTMGL1_003").clip(area)
                slope = ee.Terrain.slope(dem)
                hillshade = ee.Terrain.hillshade(dem)

                hazard_mask = slope.gte(slope_limit).And(dnbr.gt(dnbr_limit))
                
                try:
                    precip = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterBounds(area).filterDate(target_date.advance(-1, 'month'), target_date).select('precipitation').sum().clip(area)
                    peak_rain = precip.reduceRegion(ee.Reducer.max(), area.geometry(), 250).getInfo().get('precipitation', 0)
                    hazard_acres = hazard_mask.multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), area.geometry(), 250).getInfo().get('nd', 0) * 0.000247105
                except Exception:
                    peak_rain, hazard_acres = 0, 0

                st.subheader("Automated Model Insights")
                m1, m2, m3 = st.columns(3)
                m1.metric("Active Hazard Area", f"{hazard_acres:,.1f} Acres", delta=f"> {dnbr_limit} dNBR", delta_color="inverse")
                m2.metric("Peak Rainfall Intensity", f"{peak_rain:,.1f} mm", delta="During Window", delta_color="off")
                m3.metric("Geomorphic Threshold", f"{slope_limit} Degrees", delta="User Defined", delta_color="off")
                st.markdown("---")

                centroid = fire_subset.geometry.centroid.iloc[0]
                m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr="Google")
                
                legend_html = f"""
                <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; background-color: white; border:1px solid grey; z-index:9999; font-size:13px; padding: 12px; border-radius: 4px;">
                <b style="color:#2c3e50; font-size:14px;">Spatial Layers</b><br><hr style="margin: 4px 0;">
                <i style="background:red; width:12px; height:12px; float:left; margin-right:8px; border:1px solid black;"></i> <span style="color:black;">Fire Perimeter</span><br>
                <i style="background:#bd0026; width:12px; height:12px; float:left; margin-right:8px;"></i> <span style="color:black;">Severe Burn (dNBR)</span><br>
                <i style="background:#ff7b00; width:12px; height:12px; float:left; margin-right:8px;"></i> <span style="color:black;">Hazard Initiation Zone</span><br>
                <i style="background:#3498db; width:12px; height:3px; float:left; margin-right:8px; margin-top:5px;"></i> <span style="color:black;">HydroSHEDS Streams</span>
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

                folium.GeoJson(fire_subset.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 3}).add_to(m)
                st_folium(m, use_container_width=True, height=650)
            
            except Exception as e:
                st.error(f"Geospatial calculation failed. Error details: {e}")
    else:
        st.info("Toggle 'Activate Spatial Modeling Engine' to render layers.")

# ==========================================
# PAGE 3: STATISTICAL REPORT
# ==========================================
elif page == "3. Statistical Report" and all_fires is not None:
    st.title("Watershed Statistical Analysis")
    
    run_stats = st.toggle("Generate Regional Vulnerability Map & Report", value=False)

    if run_stats:
        with st.spinner("Extracting atmospheric data and reducing spatial arrays across HUC-12 boundaries..."):
            try:
                area = ee.FeatureCollection(fire_subset.__geo_interface__)
                pre_date = ee.Date(manual_baseline.strftime('%Y-%m-%d'))
                fire_start_ee = ee.Date(default_alarm_dt.strftime('%Y-%m-%d'))
                target_date = fire_start_ee.advance(recovery_months, 'month')

                def get_nbr_median(date_obj):
                    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area).filterDate(date_obj.advance(-2.5, 'month'), date_obj.advance(2.5, 'month')).median().clip(area).normalizedDifference(['B8', 'B12'])

                dnbr = get_nbr_median(pre_date).subtract(get_nbr_median(target_date))
                slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).clip(area)
                hazard_mask = slope.gte(slope_limit).And(dnbr.gt(dnbr_limit))
                
                hazard_area_img = hazard_mask.multiply(ee.Image.pixelArea()).rename('hazard_area')
                precip_img = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterBounds(area).filterDate(target_date.advance(-1, 'month'), target_date).select('precipitation').sum().rename('rainfall')
                
                combined_img = hazard_area_img.addBands(precip_img)
                
                # OPTIMIZATION: Tightly bound the HUC12 search to just the fire geometry to save memory
                huc12 = ee.FeatureCollection("USGS/WBD/2017/HUC12").filterBounds(area.geometry())
                
                # CRITICAL FIX: scale=500 and tileScale=16 prevents the GEE Timeout
                reduced_stats_fc = combined_img.reduceRegions(
                    collection=huc12,
                    reducer=ee.Reducer.sum().combine(reducer2=ee.Reducer.mean(), sharedInputs=False),
                    scale=500,
                    tileScale=16
                )
                
                def remove_geo(feature):
                    return ee.Feature(None, feature.toDictionary())
                
                stats_only = reduced_stats_fc.map(remove_geo)
                reduced_stats = stats_only.getInfo()
                
                ws_data = []
                for f in reduced_stats['features']:
                    props = f['properties']
                    
                    raw_sq_meters = props.get('hazard_area_sum', 0)
                    if raw_sq_meters is None: raw_sq_meters = 0
                    acres = raw_sq_meters * 0.000247105
                    
                    rain_mm = props.get('rainfall_mean', 0)
                    if rain_mm is None: rain_mm = 0
                    
                    if acres > 0:
                        ws_data.append({
                            "HUC-12 Watershed Name": props.get('name', 'Unknown'), 
                            "Active Hazard Footprint (Acres)": round(acres, 2),
                            "Mean Rainfall (mm)": round(rain_mm, 2)
                        })
                
                df_ws = pd.DataFrame(ws_data).sort_values(by="Active Hazard Footprint (Acres)", ascending=False)
                
                if df_ws.empty:
                    st.success("No active hazard areas detected. Try lowering the dNBR threshold in the sidebar.")
                else:
                    st.subheader("Regional Vulnerability Map")
                    
                    centroid = fire_subset.geometry.centroid.iloc[0]
                    m3 = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles='CartoDB dark_matter')
                    
                    w_outline = ee.Image(0).mask(0).paint(huc12, 'purple', 2)
                    folium.TileLayer(tiles=w_outline.getMapId({'palette':['purple']})['tile_fetcher'].url_format, attr='USGS', name='Watersheds').add_to(m3)
                    folium.TileLayer(tiles=hazard_mask.updateMask(hazard_mask).getMapId({'palette':['#ff7b00']})['tile_fetcher'].url_format, attr='GEE', name='Hazard Zones').add_to(m3)

                    folium.GeoJson(fire_subset.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 2}).add_to(m3)
                    st_folium(m3, use_container_width=True, height=500)
                    st.markdown("---")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("Debris Flow Trigger Matrix")
                        st.dataframe(df_ws, use_container_width=True, hide_index=True)
                        
                        csv = df_ws.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Matrix (CSV)", data=csv, file_name=f'{selected_name}_trigger_matrix.csv', mime='text/csv')
                        
                    with c2:
                        st.subheader("Atmospheric Trigger Analysis")
                        scatter = alt.Chart(df_ws).mark_circle(size=200, color='#3498db', opacity=0.8).encode(
                            x=alt.X('Active Hazard Footprint (Acres):Q', title='Hazard Area (Acres)'),
                            y=alt.Y('Mean Rainfall (mm):Q', title='Cumulative Rainfall (mm)'),
                            tooltip=['HUC-12 Watershed Name', 'Active Hazard Footprint (Acres)', 'Mean Rainfall (mm)']
                        ).properties(height=350)
                        
                        text = scatter.mark_text(align='left', baseline='middle', dx=15, color="white").encode(text='HUC-12 Watershed Name:N')
                        st.altair_chart(scatter + text, use_container_width=True)

                st.info("""
                **Analytical Note:** This scatter matrix identifies Trigger Zones. Watersheds positioned in the top-right quadrant exhibit both critical geomorphic instability (high hazard area) and significant atmospheric loading (high rainfall), making them prime candidates for immediate debris flow monitoring.
                """)
            
            except Exception as e:
                st.error(f"Earth Engine Computation Timeout. Error details: {e}")

    else:
        st.info("Toggle 'Generate Regional Vulnerability Map & Report' to calculate spatial metrics.")
