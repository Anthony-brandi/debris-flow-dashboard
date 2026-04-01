import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import ee
import json
from datetime import datetime, timedelta

# ==========================================
# 1. PAGE SETUP & ARCHITECTURE
# ==========================================
st.set_page_config(page_title="PF-WRP | Post-Fire Watershed Risk Portal", layout="wide")

st.sidebar.title("PF-WRP Navigation")
page = st.sidebar.radio("Select Module:", [
    "1. Incident Briefing", 
    "2. Spatial Modeling Lab", 
    "3. Watershed Loading (Phase 2)"
])

# ==========================================
# 2. GEE INITIALIZATION
# ==========================================
if 'ee_initialized' not in st.session_state:
    try:
        if "EARTHENGINE_JSON" in st.secrets:
            creds_dict = json.loads(st.secrets["EARTHENGINE_JSON"])
            credentials = ee.ServiceAccountCredentials(creds_dict['client_email'], key_data=st.secrets["EARTHENGINE_JSON"])
            ee.Initialize(credentials, project='strange-bird-461405-v7')
        else:
            ee.Initialize(project='strange-bird-461405-v7')
        st.session_state['ee_initialized'] = True
    except Exception as e:
        st.error(f"Earth Engine Initialization Error: {e}")

@st.cache_data
def load_and_clean_data():
    # Update this path to match your local environment
    path = 'Master_Fire_Dataset.geojson.zip'
    try:
        fires = gpd.read_file(path)
        fires = fires.dissolve(by='incident_n').reset_index()
        return fires.to_crs(epsg=4326)
    except Exception as e:
        st.error(f"Failed to load perimeter data: {e}")
        return gpd.GeoDataFrame()

# ==========================================
# GLOBAL FIRE SELECTION & DATE PARSING
# ==========================================
cal_fires = load_and_clean_data()

if not cal_fires.empty:
    fire_list = sorted(cal_fires['incident_n'].fillna(cal_fires['mission']).dropna().unique())
    selected_fire = st.sidebar.selectbox("Select Wildfire Perimeter", fire_list)
    fire_data = cal_fires[cal_fires['incident_n'] == selected_fire]
    
    # Dynamic Ignition Date Extractor
    ignition_date = datetime(2021, 1, 1) # Fallback
    for col in ['START_DATE', 'ALARM_DATE', 'alarm_date', 'cont_date']:
        if col in fire_data.columns and not pd.isna(fire_data[col].iloc[0]):
            try:
                ignition_date = pd.to_datetime(fire_data[col].iloc[0]).to_pydatetime()
                break
            except Exception:
                continue

    # Sentinel-2 Time Windows (Pre-fire: 1 year before, Post-fire: 3-6 months after)
    pre_fire_start = (ignition_date - timedelta(days=365)).strftime('%Y-%m-%d')
    pre_fire_end = (ignition_date - timedelta(days=1)).strftime('%Y-%m-%d')
    post_fire_start = (ignition_date + timedelta(days=90)).strftime('%Y-%m-%d')
    post_fire_end = (ignition_date + timedelta(days=180)).strftime('%Y-%m-%d')

    area = ee.FeatureCollection(fire_data.__geo_interface__)
    centroid = fire_data.to_crs(epsg=3310).geometry.centroid.to_crs(epsg=4326).iloc[0]
else:
    st.error("No fire perimeters loaded. Please check the filepath.")
    st.stop()

def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)

# ==========================================
# PAGE 1: INCIDENT BRIEFING
# ==========================================
if page == "1. Incident Briefing":
    st.title(f"Incident Briefing: {selected_fire}")
    st.markdown("### Rapid Assessment Overview")
    
    total_ac = (fire_data.to_crs(epsg=3310).area.sum()) * 0.000247105
    st.metric("Total Acres Burned", f"{total_ac:,.0f} ac")
    st.metric("Estimated Ignition Date", ignition_date.strftime('%B %d, %Y'))

    st.markdown("---")
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles="CartoDB positron")
    folium.GeoJson(fire_data.geometry, style_function=lambda x: {'fillColor': 'red', 'color': 'darkred', 'weight': 2, 'fillOpacity': 0.4}).add_to(m)
    st_folium(m, use_container_width=True, height=500)

# ==========================================
# PAGE 2: SPATIAL MODELING LAB
# ==========================================
elif page == "2. Spatial Modeling Lab":
    st.title("Spatial Modeling Lab (Engineering View)")
    
    # ---------------------------------------------------------
    # THE BLACK BOX: Hardcoded Scientific Thresholds
    # ---------------------------------------------------------
    SLOPE_LIMIT = 25
    DNBR_THRESHOLD = 0.25
    
    st.sidebar.markdown("### Model Parameters")
    st.sidebar.info(f"**Critical Slope:** > {SLOPE_LIMIT} Degrees\n\n**Severity (dNBR):** > {DNBR_THRESHOLD}")
    
    with st.sidebar.expander("Methodology & Reasoning"):
        st.write("""
        **Why 25 Degrees?**
        Academic literature dictates that post-fire debris flows primarily initiate in zero-order basins and channels where geomorphic slopes exceed 25 degrees, providing the necessary gravitational energy to entrain sediment.
        
        **Why dNBR > 0.25?**
        This threshold isolates moderate-to-high severity burn areas. In these zones, the consumption of canopy and root structures, combined with hydrophobic soil sealing, creates the ideal conditions for rapid surface runoff.
        """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Layer Visibility")
    show_risk = st.sidebar.checkbox("Hazard Intersection (Risk)", value=True)
    show_slope = st.sidebar.checkbox("Topographic Velocity (Slope)", value=False)
    show_severity = st.sidebar.checkbox("Burn Severity (dNBR)", value=False)
    show_soils = st.sidebar.checkbox("Soil Erodibility (K-Factor)", value=False)
    show_streams = st.sidebar.checkbox("HydroSHEDS Stream Routing", value=True)
    show_roads = st.sidebar.checkbox("TIGER Roads", value=True)

    with st.spinner("Compiling Spatial Intersection Data via Earth Engine..."):
        
        # 1. TOPOGRAPHIC VELOCITY (SLOPE)
        dem = ee.Image("USGS/SRTMGL1_003")
        slope = ee.Terrain.slope(dem).clip(area)
        slope_mask = slope.gte(SLOPE_LIMIT)

        # 2. BURN SEVERITY (dNBR)
        s2_pre = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(area).filterDate(pre_fire_start, pre_fire_end).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).map(mask_s2_clouds).median().clip(area)
        s2_post = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(area).filterDate(post_fire_start, post_fire_end).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).map(mask_s2_clouds).median().clip(area)
        
        pre_nbr = s2_pre.normalizedDifference(['B8', 'B12'])
        post_nbr = s2_post.normalizedDifference(['B8', 'B12'])
        dnbr = pre_nbr.subtract(post_nbr)
        severity_mask = dnbr.gte(DNBR_THRESHOLD)

        # 3. SOIL ERODIBILITY
        raw_soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0').clip(area)
        erodible_soils = raw_soil.lt(11).selfMask() 

        # HAZARD INTERSECTION
        hazard_intersection = slope_mask.And(severity_mask).And(erodible_soils).selfMask()

        # RASTERIZED VECTORS
        roads = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(area)
        roads_img = ee.Image(0).mask(0).paint(roads, 1, 2)
        
        streams = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(area)
        streams_img = ee.Image(0).mask(0).paint(streams, 1, 1)

        # MAP RENDER
        m2 = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Hybrid')
        folium.GeoJson(fire_data.geometry, style_function=lambda x: {'fillColor': 'transparent', 'color': 'white', 'weight': 2, 'dashArray': '5, 5'}).add_to(m2)

        if show_slope:
            s_vis = slope_mask.selfMask().getMapId({'palette': ['yellow'], 'opacity': 0.4})
            folium.TileLayer(tiles=s_vis['tile_fetcher'].url_format, attr='USGS', name='Slope').add_to(m2)
        if show_severity:
            sev_vis = severity_mask.selfMask().getMapId({'palette': ['red'], 'opacity': 0.4})
            folium.TileLayer(tiles=sev_vis['tile_fetcher'].url_format, attr='ESA', name='Severity').add_to(m2)
        if show_soils:
            soil_vis = erodible_soils.getMapId({'palette': ['#800026'], 'opacity': 0.4})
            folium.TileLayer(tiles=soil_vis['tile_fetcher'].url_format, attr='OpenLandMap', name='Soils').add_to(m2)
        if show_streams:
            stream_vis = streams_img.getMapId({'palette': ['#3498db']})
            folium.TileLayer(tiles=stream_vis['tile_fetcher'].url_format, attr='WWF', name='Streams').add_to(m2)
        if show_roads:
            road_vis = roads_img.getMapId({'palette': ['#2ecc71']})
            folium.TileLayer(tiles=road_vis['tile_fetcher'].url_format, attr='TIGER', name='Roads').add_to(m2)
        if show_risk:
            risk_vis = hazard_intersection.getMapId({'palette': ['#ff7b00'], 'opacity': 0.9})
            folium.TileLayer(tiles=risk_vis['tile_fetcher'].url_format, attr='GEE', name='Risk Intersection').add_to(m2)

        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; background-color: white; border:2px solid grey; z-index:9999; font-size:12px; padding: 10px;">
        <b>PF-WRP Legend</b><br>
        <i style="background:#ff7b00; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Hazard Intersection<br>
        <i style="background:yellow; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Critical Slope<br>
        <i style="background:red; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Severe dNBR<br>
        <i style="background:#800026; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Erodible Soils<br>
        <i style="background:#3498db; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Stream Routing<br>
        <i style="background:#2ecc71; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Infrastructure<br>
        </div>"""
        m2.get_root().html.add_child(folium.Element(legend_html))
        
        # The map will now redraw smoothly just by toggling visibility layers
        toggle_key = f"lab_{selected_fire}_v{show_risk}{show_slope}{show_severity}{show_soils}{show_streams}{show_roads}"
        st_folium(m2, use_container_width=True, height=700, key=toggle_key)

# ==========================================
# PAGE 3: WATERSHED LOADING (PHASE 2)
# ==========================================
elif page == "3. Watershed Loading (Phase 2)":
    st.warning("Phase 2 Module: Vulnerability Matrix and Sediment Yield Calculations are currently under development.")
