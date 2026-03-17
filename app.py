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
        centroid_point = fire_data.geometry.centroid.iloc[0]
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Environmental Parameters")
        recovery_months = st.sidebar.select_slider("Observation Window (Months Post-Fire)", options=[1, 6, 12, 18, 24], value=1)
        analyze_btn = st.sidebar.checkbox("Execute Hydrologic Analysis", value=False)
        slope_limit = st.sidebar.slider("Slope Threshold (Degrees)", 10, 45, 27)

        with st.sidebar.expander("Map Layer Settings", expanded=True):
            show_recovery = st.checkbox("Burn Severity (dNBR)", value=True)
            show_precip = st.checkbox("Precipitation (NASA GPM)", value=True)
            show_risk = st.checkbox("Hazard Intersection", value=False)
            show_watersheds = st.checkbox("Watershed Outlines", value=False)
            basemap = st.radio("Style", ["Satellite", "Terrain"])
            tile_url = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}' if basemap == "Satellite" else 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}'

        st.title(f"{selected_fire} Runoff Hazard Assessment")
        
        if analyze_btn:
            with st.spinner("Retrieving geospatial and satellite data..."):
                area = ee.FeatureCollection(fire_data.__geo_interface__)
                
                # --- DATE LOGIC ---
                date_col = next((c for c in ['ALARM_DATE', 'START_DATE', 'alarm_date'] if c in fire_data.columns), None)
                fire_start_dt = pd.to_datetime(fire_data[date_col].iloc[0]) if date_col else datetime(2021, 6, 1)
                pre_date = ee.Date(fire_start_dt.strftime('%Y-%m-%d')).advance(-1, 'year')
                target_date = ee.Date(fire_start_dt.strftime('%Y-%m-%d')).advance(recovery_months, 'month')

                def get_nbr_median(date_obj):
                    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area)\
                        .filterDate(date_obj.advance(-1, 'month'), date_obj.advance(1, 'month'))\
                        .median().clip(area).normalizedDifference(['B8', 'B12'])

                dnbr = get_nbr_median(pre_date).subtract(get_nbr_median(target_date))

                # --- PRECIPITATION (NASA GPM V07 WITH FALLBACK) ---
                precip_col = ee.ImageCollection("NASA/GPM_L3/IMERG_V07")\
                    .filterBounds(area)\
                    .filterDate(target_date.advance(-1, 'month'), target_date)\
                    .select('precipitation')
                
                # Check if data exists for this month, if not, alert user
                has_data = precip_col.size().getInfo()
                if has_data > 0:
                    precip = precip_col.sum().clip(area)
                    avg_precip = precip.reduceRegion(ee.Reducer.mean(), area.geometry(), 1000).getInfo().get('precipitation', 0)
                else:
                    st.warning("Precipitation data for this specific month is not yet available in the NASA GPM database. Showing 0mm.")
                    precip = ee.Image(0).clip(area)
                    avg_precip = 0

                # --- STATS ---
                total_acres = (fire_data.to_crs(epsg=3310).area.sum()) * 0.000247105
                high_sev_acres = dnbr.gt(0.44).multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), area.geometry(), 30).getInfo().get('nd', 0) * 0.000247105

                m1, m2, m3 = st.columns(3)
                m1.metric("High Severity Area", f"{high_sev_acres:,.1f} Ac")
                m2.metric("Rainfall Accumulation", f"{avg_precip:,.1f} mm")
                m3.metric("Total Perimeter", f"{total_acres:,.0f} Ac")

                # --- MAP ---
                m = folium.Map(location=[centroid_point.y, centroid_point.x], zoom_start=12, tiles=tile_url, attr="Google")
                
                # LEGEND FIX: High contrast text on white background
                legend_html = f"""
                <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; background-color: white; border:2px solid black; z-index:9999; font-size:12px; padding: 10px; border-radius: 5px;">
                <b style="color:black;">Analysis Legend</b><br>
                <i style="background:red; width:10px; height:10px; float:left; margin-right:5px; border:1px solid black;"></i> <span style="color:black;">Perimeter</span><br>
                <i style="background:#bd0026; width:10px; height:10px; float:left; margin-right:5px;"></i> <span style="color:black;">Burn Scar (High)</span><br>
                <i style="background:#084594; width:10px; height:10px; float:left; margin-right:5px;"></i> <span style="color:black;">Heavy Rainfall</span><br>
                <i style="background:#ff7b00; width:10px; height:10px; float:left; margin-right:5px;"></i> <span style="color:black;">Hazard Zone</span><br>
                <i style="border: 1px solid purple; width:10px; height:2px; float:left; margin-right:5px; margin-top:4px;"></i> <span style="color:black;">Watershed</span>
                </div>"""
                m.get_root().html.add_child(folium.Element(legend_html))

                if show_precip and has_data > 0:
                    # Masking ensures only rain > 5mm is shown to prevent map 'washing out'
                    p_vis = {'min': 5, 'max': 150, 'palette': ['#f7fbff','#deebf7','#9ecae1','#4292c6','#084594']}
                    precip_masked = precip.updateMask(precip.gt(5))
                    folium.TileLayer(tiles=precip_masked.getMapId(p_vis)['tile_fetcher'].url_format, attr='NASA', name='Rainfall', opacity=0.45).add_to(m)

                if show_recovery:
                    vis = {'min': 0.1, 'max': 0.5, 'palette': ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']}
                    dnbr_masked = dnbr.updateMask(dnbr.gt(0.1))
                    folium.TileLayer(tiles=dnbr_masked.getMapId(vis)['tile_fetcher'].url_format, attr='S2', name='Burn Status', opacity=0.7).add_to(m)

                folium.GeoJson(fire_data.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 3}).add_to(m)
                st_folium(m, use_container_width=True, height=750)

        else:
            m = folium.Map(location=[centroid_point.y, centroid_point.x], zoom_start=12, tiles=tile_url, attr="Google")
            folium.GeoJson(fire_data.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 2}).add_to(m)
            st_folium(m, use_container_width=True, height=750)

    except Exception as e:
        st.error(f"Error: {e}")
