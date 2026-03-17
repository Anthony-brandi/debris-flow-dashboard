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
    
    # Robust date detection
    date_options = ['ALARM_DATE', 'ALARM_DAT', 'START_DATE', 'alarm_date', 'alarm_dat']
    found_col = next((col for col in date_options if col in fires.columns), None)
    
    if found_col:
        fires['final_date'] = pd.to_datetime(fires[found_col], errors='coerce')
    else:
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
# 3. SIDEBAR CONTROLS (FULL RESTORATION)
# ==========================================
st.sidebar.title("Main Menu")
page = st.sidebar.radio("Navigation", ["Interactive Risk Map", "User Manual", "Technical Documentation"])

if page == "Interactive Risk Map":
    try:
        cal_fires = load_fire_perimeters()
        fire_list = sorted(cal_fires['incident_n'].dropna().unique())
        selected_fire = st.sidebar.selectbox("Select Wildfire Incident", fire_list)
        fire_data = cal_fires[cal_fires['incident_n'] == selected_fire]
        
        actual_date = fire_data['final_date'].iloc[0]
        st.sidebar.info(f"Analysis Baseline: {actual_date.strftime('%B %d, %Y')}")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Environmental Parameters")
        recovery_months = st.sidebar.select_slider("Observation Window (Months Post-Fire)", options=[1, 6, 12, 18, 24], value=1)
        analyze_btn = st.sidebar.checkbox("Execute Hydrologic Analysis", value=True)
        slope_limit = st.sidebar.slider("Slope Threshold (Degrees)", 10, 45, 27)

        # MANDATORY LAYER TABS (Fixed and always visible)
        st.sidebar.markdown("---")
        st.sidebar.subheader("Map Layer Controls")
        show_recovery = st.sidebar.checkbox("Burn Severity (dNBR)", value=True)
        show_precip = st.sidebar.checkbox("Precipitation (NASA GPM)", value=False)
        show_risk = st.sidebar.checkbox("Hazard Zones (Slope+Burn)", value=True)
        show_watersheds = st.sidebar.checkbox("Watershed Outlines", value=True)
        show_infra = st.sidebar.checkbox("Clipped Roads (TIGER)", value=True)
        
        basemap = st.sidebar.radio("Reference Style", ["Satellite", "Terrain"])
        tile_url = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}' if basemap == "Satellite" else 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}'

        # ==========================================
        # 4. ANALYTICS & VISUALIZATION
        # ==========================================
        st.title(f"{selected_fire} Runoff Hazard Assessment")
        
        if analyze_btn:
            with st.spinner("Executing spatial intersection..."):
                area = ee.FeatureCollection(fire_data.__geo_interface__)
                
                # Spectral Analysis
                pre_date = ee.Date(actual_date.strftime('%Y-%m-%d')).advance(-1, 'year')
                target_date = ee.Date(actual_date.strftime('%Y-%m-%d')).advance(recovery_months, 'month')

                def get_nbr_median(date_obj):
                    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area)\
                        .filterDate(date_obj.advance(-1, 'month'), date_obj.advance(1, 'month'))\
                        .median().clip(area).normalizedDifference(['B8', 'B12'])

                dnbr = get_nbr_median(pre_date).subtract(get_nbr_median(target_date))
                
                # Topography
                dem = ee.Image("USGS/SRTMGL1_003")
                slope = ee.Terrain.slope(dem).clip(area)

                # Rainfall (NASA GPM)
                precip = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterBounds(area)\
                    .filterDate(target_date.advance(-1, 'month'), target_date)\
                    .select('precipitation').sum().clip(area)

                # Stats Calculation
                total_acres = (fire_data.to_crs(epsg=3310).area.sum()) * 0.000247105
                high_sev_acres = dnbr.gt(0.44).multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), area.geometry(), 30).getInfo().get('nd', 0) * 0.000247105
                avg_precip = precip.reduceRegion(ee.Reducer.mean(), area.geometry(), 1000).getInfo().get('precipitation', 0)
                recovery_pct = max(0, min(100, (100 - ((high_sev_acres / (total_acres * 0.15)) * 100))))

                # METRICS ROW
                m1, m2, m3 = st.columns(3)
                m1.metric("High Severity Area", f"{high_sev_acres:,.1f} Ac")
                m2.metric("Healing Rate", f"{recovery_pct:.1f}%")
                m3.metric("Rainfall (NASA GPM)", f"{avg_precip:,.1f} mm")
                st.markdown("---")

                # MAP RENDER
                centroid_point = fire_data.geometry.centroid.iloc[0]
                m = folium.Map(location=[centroid_point.y, centroid_point.x], zoom_start=12, tiles=tile_url, attr="Google")
                
                # Legend Fix
                legend_html = f"""
                <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; background-color: white; border:2px solid grey; z-index:9999; font-size:12px; padding: 10px; border-radius: 5px;">
                <b style="color:black;">Analysis Legend</b><br>
                <i style="background:red; width:10px; height:10px; float:left; margin-right:5px; border:1px solid black;"></i> Perimeter<br>
                <i style="background:#bd0026; width:10px; height:10px; float:left; margin-right:5px;"></i> Burn Scar (High)<br>
                <i style="background:#2ecc71; width:10px; height:10px; float:left; margin-right:5px;"></i> Clipped Roads<br>
                <i style="background:#ff7b00; width:10px; height:10px; float:left; margin-right:5px;"></i> Hazard Zone<br>
                <i style="border: 1px solid purple; width:10px; height:2px; float:left; margin-right:5px; margin-top:4px;"></i> Watershed
                </div>"""
                m.get_root().html.add_child(folium.Element(legend_html))

                if show_recovery:
                    vis = {'min': 0.1, 'max': 0.5, 'palette': ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']}
                    folium.TileLayer(tiles=dnbr.updateMask(dnbr.gt(0.1)).getMapId(vis)['tile_fetcher'].url_format, attr='S2', name='Burn Status', opacity=0.7).add_to(m)

                if show_precip:
                    p_vis = {'min': 1, 'max': 150, 'palette': ['#f7fbff','#deebf7','#9ecae1','#4292c6','#084594']}
                    folium.TileLayer(tiles=precip.updateMask(precip.gt(1)).getMapId(p_vis)['tile_fetcher'].url_format, attr='NASA', name='Rainfall', opacity=0.5).add_to(m)

                if show_risk:
                    hazard = slope.gte(slope_limit).And(dnbr.gt(0.1))
                    folium.TileLayer(tiles=hazard.updateMask(hazard).getMapId({'palette':['#ff7b00']})['tile_fetcher'].url_format, attr='GEE', name='Risk').add_to(m)

                if show_watersheds:
                    watersheds = ee.FeatureCollection("USGS/WBD/2017/HUC12").filterBounds(area)
                    w_outline = ee.Image(0).mask(0).paint(watersheds, 'purple', 2)
                    folium.TileLayer(tiles=w_outline.getMapId({'palette':['purple']})['tile_fetcher'].url_format, attr='USGS', name='Watersheds').add_to(m)

                if show_infra:
                    roads = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(area)
                    r_img = ee.Image(0).mask(0).paint(roads, '#2ecc71', 1.5)
                    folium.TileLayer(tiles=r_img.getMapId({'palette':['#2ecc71']})['tile_fetcher'].url_format, attr='TIGER', name='Roads').add_to(m)

                folium.GeoJson(fire_data.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 3}).add_to(m)
                st_folium(m, use_container_width=True, height=750)

    except Exception as e:
        st.error(f"Analysis Error: {e}")

# ==========================================
# 5. DOCUMENTATION (FULL RESTORATION)
# ==========================================
elif page == "User Manual":
    st.title("User Manual")
    st.info("The Green lines represent Roads (TIGER/Line). These are clipped specifically to the fire boundary to show which transportation corridors are vulnerable.")
    st.header("Step-by-Step Instructions")
    st.write("""
    1. **Incident Selection:** Use the dropdown menu to choose a historical wildfire perimeter.
    2. **Timeline Analysis:** Adjust the observation slider to view landscape recovery at different monthly intervals.
    3. **Parameter Tuning:** Set the Slope Threshold to identify high-gradient initiation zones.
    4. **Visualization:** Toggle map layers in the sidebar to visualize the intersection of severe burn scars and critical terrain.
    """)

elif page == "Technical Documentation":
    st.title("Technical Methodology")
    st.markdown("---")
    st.subheader("Clipped Infrastructure")
    st.write("Clipped Infrastructure: We use the 2016 TIGER/Line dataset and perform a spatial filter against the CAL FIRE incident perimeter.")
    st.header("Scientific Framework")
    st.write("""
    * **dNBR:** Differenced Normalized Burn Ratio. Uses SWIR and NIR bands to measure vegetation loss.
    * **Hydrologic Response:** Modeling the transition of fire-impacted soils from a 'Sponge' (infiltration) to a 'Funnel' (runoff).
    * **Topographic Initiation:** Identifying zones above 27 degrees where soil cohesion is compromised.
    """)
