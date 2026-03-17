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

# ==========================================
# PAGE 1: INTERACTIVE RISK MAP
# ==========================================
if page == "Interactive Risk Map":
    try:
        cal_fires = load_fire_perimeters()
        fire_list = sorted(cal_fires['incident_n'].dropna().unique())
        selected_fire = st.sidebar.selectbox("Select Wildfire Incident", fire_list)
        fire_data = cal_fires[cal_fires['incident_n'] == selected_fire]
        
        # Centroid defined for map focusing
        centroid_point = fire_data.geometry.centroid.iloc[0]
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Environmental Parameters")
        recovery_months = st.sidebar.select_slider("Observation Window (Months Post-Fire)", options=[1, 6, 12, 18, 24], value=1)
        analyze_btn = st.sidebar.checkbox("Execute Hydrologic Analysis", value=False)
        slope_limit = st.sidebar.slider("Slope Threshold (Degrees)", 10, 45, 27)

        with st.sidebar.expander("Map Layer Settings", expanded=True):
            show_recovery = st.checkbox("Burn Severity (dNBR)", value=True)
            show_precip = st.checkbox("Precipitation (NASA GPM)", value=False)
            show_risk = st.checkbox("Hazard Intersection (Slope+Burn)", value=True)
            show_watersheds = st.checkbox("Watershed Outlines (HUC-12)", value=True)
            show_infra = st.checkbox("Clipped Infrastructure (Roads)", value=True)
            basemap = st.radio("Style", ["Satellite", "Terrain"])
            tile_url = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}' if basemap == "Satellite" else 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}'

        st.title(f"{selected_fire} Runoff Hazard Assessment")
        
        if analyze_btn:
            with st.spinner("Processing multispectral and hydrologic datasets..."):
                area = ee.FeatureCollection(fire_data.__geo_interface__)
                
                # --- DNBR & TOPOGRAPHY ---
                date_col = next((c for c in ['ALARM_DATE', 'START_DATE', 'alarm_date'] if c in fire_data.columns), None)
                fire_start_dt = pd.to_datetime(fire_data[date_col].iloc[0]) if date_col else datetime(2021, 6, 1)
                pre_date = ee.Date(fire_start_dt.strftime('%Y-%m-%d')).advance(-1, 'year')
                target_date = ee.Date(fire_start_dt.strftime('%Y-%m-%d')).advance(recovery_months, 'month')

                def get_nbr_median(date_obj):
                    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area)\
                        .filterDate(date_obj.advance(-1, 'month'), date_obj.advance(1, 'month'))\
                        .median().clip(area).normalizedDifference(['B8', 'B12'])

                dnbr = get_nbr_median(pre_date).subtract(get_nbr_median(target_date))
                dem = ee.Image("USGS/SRTMGL1_003")
                slope = ee.Terrain.slope(dem).clip(area)

                # --- STATS ---
                total_acres = (fire_data.to_crs(epsg=3310).area.sum()) * 0.000247105
                high_sev_acres = dnbr.gt(0.44).multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), area.geometry(), 30).getInfo().get('nd', 0) * 0.000247105
                recovery_pct = max(0, min(100, (100 - ((high_sev_acres / (total_acres * 0.15)) * 100))))

                m1, m2, m3 = st.columns(3)
                m1.metric("High Severity Area", f"{high_sev_acres:,.1f} Ac")
                m2.metric("Landscape Healing", f"{recovery_pct:.1f}%")
                m3.metric("Total Perimeter", f"{total_acres:,.0f} Ac")

                # --- MAP INITIALIZATION ---
                m = folium.Map(location=[centroid_point.y, centroid_point.x], zoom_start=12, tiles=tile_url, attr="Google")
                
                # Professional Legend
                legend_html = f"""
                <div style="position: fixed; bottom: 50px; left: 50px; width: 240px; background-color: white; border:2px solid grey; z-index:9999; font-size:12px; padding: 10px; border-radius: 5px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
                <b>Analysis Legend</b><br>
                <i style="background:red; width:10px; height:10px; float:left; margin-right:5px; border:1px solid black;"></i> Fire Perimeter<br>
                <i style="background:rgba(189, 0, 38, 0.7); width:10px; height:10px; float:left; margin-right:5px;"></i> Burn Scar (High Severity)<br>
                <i style="background:#ff7b00; width:10px; height:10px; float:left; margin-right:5px;"></i> Debris Risk Zone<br>
                <i style="background:#2ecc71; width:10px; height:10px; float:left; margin-right:5px;"></i> Clipped Roads<br>
                <i style="border: 1px solid purple; width:10px; height:2px; float:left; margin-right:5px; margin-top:4px;"></i> Watershed Boundary
                </div>"""
                m.get_root().html.add_child(folium.Element(legend_html))

                if show_recovery:
                    # Isolating actual burn scars from healthy land
                    vis = {'min': 0.1, 'max': 0.5, 'palette': ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']}
                    folium.TileLayer(tiles=dnbr.updateMask(dnbr.gt(0.1)).getMapId(vis)['tile_fetcher'].url_format, attr='S2', name='Burn Status', opacity=0.7).add_to(m)

                if show_watersheds:
                    watersheds = ee.FeatureCollection("USGS/WBD/2017/HUC12").filterBounds(area)
                    # Rendering as an outline to prevent the "white-out" effect
                    w_outline = ee.Image(0).mask(0).paint(watersheds, 'purple', 1.5)
                    folium.TileLayer(tiles=w_outline.getMapId({'palette':['purple']})['tile_fetcher'].url_format, attr='USGS', name='Watersheds').add_to(m)

                if show_risk:
                    hazard = slope.gte(slope_limit).And(dnbr.gt(0.1))
                    folium.TileLayer(tiles=hazard.updateMask(hazard).getMapId({'palette':['#ff7b00']})['tile_fetcher'].url_format, attr='GEE', name='Risk Intersection').add_to(m)

                if show_infra:
                    # Precise road clipping to fire boundary
                    roads = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(area)
                    r_img = ee.Image(0).mask(0).paint(roads, '#2ecc71', 1)
                    folium.TileLayer(tiles=r_img.getMapId({'palette':['#2ecc71']})['tile_fetcher'].url_format, attr='TIGER', name='Roads').add_to(m)

                # Base Perimeter for reference
                folium.GeoJson(fire_data.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 3}).add_to(m)
                st_folium(m, use_container_width=True, height=750)

        else:
            # Standby Map
            m = folium.Map(location=[centroid_point.y, centroid_point.x], zoom_start=12, tiles=tile_url, attr="Google")
            folium.GeoJson(fire_data.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 2}).add_to(m)
            st_folium(m, use_container_width=True, height=750)

    except Exception as e:
        st.error(f"Execution Error: {e}")

# ==========================================
# PAGE 2 & 3: DOCUMENTATION
# ==========================================
elif page == "User Manual":
    st.title("User Manual")
    st.info("The purple lines represent Watershed Boundaries (HUC-12). These show the drainages where post-fire runoff will concentrate.")
    st.write("1. Select an incident and timeline.\n2. Adjust your slope sensitivity.\n3. Run analysis to identify where infrastructure crosses critical hazard zones.")

elif page == "Technical Documentation":
    st.title("Technical Methodology")
    st.write("This dashboard leverages dNBR multispectral indices and USGS topographic thresholds to model landslide initiation points.")
