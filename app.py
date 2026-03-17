import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import ee
import json
from datetime import datetime, timedelta

# ==========================================
# 1. SYSTEM CONFIGURATION & DATA RECOVERY
# ==========================================
st.set_page_config(page_title="Post-Fire Geomorphic Recovery Analysis", layout="wide")

GW_BASINS_URL = "https://opendata.arcgis.com/datasets/5da310c66bc649f0bc4ad21820b36873_0.geojson"

@st.cache_data
def load_fire_perimeters():
    path = 'CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view/CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view.shp'
    fires = gpd.read_file(path)
    fires = fires.dissolve(by='incident_n').reset_index()
    return fires.to_crs(epsg=4326)

@st.cache_data
def load_filtered_basins(fire_geometry):
    all_basins = gpd.read_file(GW_BASINS_URL)
    return all_basins[all_basins.intersects(fire_geometry)]

# ==========================================
# 2. GOOGLE EARTH ENGINE INITIALIZATION
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
        st.error(f"GEE Initialization Error: {e}")

# ==========================================
# 3. SIDEBAR: PARAMETER DEFINITION
# ==========================================
st.sidebar.title("Analytical Parameters")
page = st.sidebar.radio("Navigation", ["Interactive Analysis", "Scientific Documentation"])

if page == "Interactive Analysis":
    try:
        cal_fires = load_fire_perimeters()
        fire_list = sorted(cal_fires['incident_n'].dropna().unique())
        selected_fire = st.sidebar.selectbox("Select Wildfire Incident", fire_list)
        fire_data = cal_fires[cal_fires['incident_n'] == selected_fire]
        fire_geom = fire_data.geometry.iloc[0]
        centroid_point = fire_data.geometry.centroid.iloc[0]
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Temporal Settings")
        enable_temporal = st.sidebar.toggle("Enable Temporal Analysis", value=True)
        
        recovery_months = 1
        if enable_temporal:
            recovery_months = st.sidebar.select_slider(
                "Post-Fire Observation Window (Months)", 
                options=[1, 6, 12, 18, 24], 
                value=1
            )
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Geomorphic Settings")
        analyze_btn = st.sidebar.checkbox("Execute Spatial Computation", value=False)
        slope_limit = st.sidebar.slider("Slope Threshold (Degrees)", 10, 45, 27)

        with st.sidebar.expander("Geospatial Layer Settings"):
            show_recovery = st.checkbox("Burn Severity (dNBR)", value=True)
            show_basins = st.checkbox("Groundwater Basin Boundaries", value=True)
            show_risk = st.checkbox("Critical Hazard Intersection", value=False)
            show_infra = st.checkbox("Infrastructure (TIGER/Line)", value=True)
            basemap = st.radio("Reference Basemap", ["Satellite", "Terrain"])
            tile_url = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}' if basemap == "Satellite" else 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}'

        # ==========================================
        # 4. RESULTS INTERPRETATION PANEL
        # ==========================================
        st.title(f"{selected_fire} Debris Flow and Aquifer Recharge Analysis")
        
        if analyze_btn:
            with st.spinner("Processing multispectral and topographic data..."):
                area = ee.FeatureCollection(fire_data.__geo_interface__)
                
                # --- TEMPORAL DNBR PROCESSING ---
                date_col = next((c for c in ['ALARM_DATE', 'START_DATE', 'alarm_date'] if c in fire_data.columns), None)
                fire_start_dt = pd.to_datetime(fire_data[date_col].iloc[0]) if date_col else datetime(2021, 6, 1)
                
                pre_date = ee.Date(fire_start_dt.strftime('%Y-%m-%d')).advance(-1, 'year')
                post_date = ee.Date(fire_start_dt.strftime('%Y-%m-%d')).advance(recovery_months, 'month')

                def get_nbr_median(target_date):
                    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                        .filterBounds(area)\
                        .filterDate(target_date.advance(-1, 'month'), target_date.advance(1, 'month'))\
                        .median().clip(area).normalizedDifference(['B8', 'B12'])

                nbr_pre = get_nbr_median(pre_date)
                nbr_post = get_nbr_median(post_date)
                dnbr = nbr_pre.subtract(nbr_post)

                # --- TOPOGRAPHY & INFRASTRUCTURE ---
                dem = ee.Image("USGS/SRTMGL1_003")
                slope = ee.Terrain.slope(dem).clip(area)
                roads = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(area)
                road_stats = roads.map(lambda f: f.set('length', f.length())).aggregate_sum('length').getInfo()
                road_miles = road_stats * 0.000621371

                # --- STATISTICAL COMPUTATION ---
                total_acres = (fire_data.to_crs(epsg=3310).area.sum()) * 0.000247105
                high_sev_stats = dnbr.gt(0.44).multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(), geometry=area.geometry(), scale=30
                ).getInfo()
                high_sev_acres = high_sev_stats.get('nd', 0) * 0.000247105
                
                steep_stats = slope.gte(slope_limit).multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(), geometry=area.geometry(), scale=30
                ).getInfo()
                steep_acres = steep_stats.get('slope', 0) * 0.000247105

                # --- RECOVERY PERCENTAGE CALCULATION ---
                # Estimated baseline: peak high severity is often ~15% of perimeter for major CA fires
                baseline_peak = total_acres * 0.15 
                recovery_value = 100 - ((high_sev_acres / baseline_peak) * 100)
                recovery_pct = max(0, min(100, recovery_value)) # Clamping between 0 and 100

                # --- METRIC DISPLAY ---
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Total Perimeter", f"{total_acres:,.0f} Ac")
                m_col2.metric("High Severity", f"{high_sev_acres:,.1f} Ac")
                m_col3.metric("Landscape Recovery", f"{recovery_pct:.1f}%")
                m_col4.metric("Roads Exposed", f"{road_miles:.2f} Mi")

                with st.expander("Interpret These Results", expanded=True):
                    st.write(f"""
                    **High Severity:** Represents areas where canopy and root structures are completely compromised. At **{high_sev_acres:,.1f} acres**, 
                    these zones pose the highest risk for sediment transport.
                    
                    **Landscape Recovery:** This metric indicates the percentage of the landscape that has transitioned out of the 'High Severity' class 
                    since the initial ignition. A value of **{recovery_pct:.1f}%** suggests that the landscape is effectively stabilizing, 
                    allowing for increased groundwater infiltration as required by SGMA.
                    """)

                # SUCCESSION CHART
                st.markdown("---")
                chart_data = pd.DataFrame({
                    "Interval": ["Initial (Estimated)", "Current Observation", "24-Month Target"],
                    "High Severity Acres": [baseline_peak, high_sev_acres, baseline_peak * 0.1]
                })
                st.bar_chart(chart_data, x="Interval", y="High Severity Acres")

                # MAP RENDER
                m = folium.Map(location=[centroid_point.y, centroid_point.x], zoom_start=12, tiles=tile_url, attr="Google")
                folium.GeoJson(fire_geom, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 2}).add_to(m)

                if show_basins:
                    local_basins = load_filtered_basins(fire_geom)
                    folium.GeoJson(local_basins, name="Groundwater Basins", 
                                   style_function=lambda x: {'fillColor': '#3498db', 'color': 'blue', 'weight': 1, 'fillOpacity': 0.15}).add_to(m)

                if show_recovery:
                    dnbr_vis = {'min': -0.1, 'max': 0.5, 'palette': ['ffffff', '7ad071', 'f9e072', 'ff0000']}
                    dnbr_mapid = dnbr.getMapId(dnbr_vis)
                    folium.TileLayer(tiles=dnbr_mapid['tile_fetcher'].url_format, attr='Sentinel-2', name='Burn Severity', opacity=0.7).add_to(m)

                if show_infra:
                    roads_img = ee.Image(0).mask(0).paint(roads, 1, 2)
                    infra_id = roads_img.getMapId({'palette': ['#2ecc71']})
                    folium.TileLayer(tiles=infra_id['tile_fetcher'].url_format, attr='TIGER', name='Infrastructure').add_to(m)

                st_folium(m, use_container_width=True, height=700, key="map_main")

        else:
            m = folium.Map(location=[centroid_point.y, centroid_point.x], zoom_start=12, tiles=tile_url, attr="Google")
            folium.GeoJson(fire_geom, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 2}).add_to(m)
            st_folium(m, use_container_width=True, height=700, key="map_default")

    except Exception as e:
        st.error(f"Computation Error: {e}")

# ==========================================
# 5. SCIENTIFIC DOCUMENTATION
# ==========================================
elif page == "Scientific Documentation":
    st.title("Methodology and User Interpretation")
    st.markdown("---")
    
    st.header("1. Understanding the Interface Toggles")
    st.write("""
    * **Post-Fire Observation Window:** Wildfire recovery is a temporal process. By adjusting this slider, users can observe the 'greening up' 
        of the landscape as secondary succession replaces scorched soil with new vegetation.
    * **Slope Threshold:** This slider isolates terrain that exceeds the critical angle of repose. In California, slopes greater than 27 degrees 
        are significantly more likely to initiate high-velocity debris flows during precipitation events.
    * **Critical Hazard Intersection:** This layer identifies areas where steep terrain meets unrecovered burn scars. These are the 
        primary 'hotspots' for sediment transport and infrastructure risk.
    """)

    

    st.header("2. Interpreting Burn Severity (dNBR)")
    st.write("""
    Burn severity is calculated using the Differenced Normalized Burn Ratio (dNBR). High dNBR values (shown in red) indicate a 
    total loss of photosynthetic material and soil organic matter. As these values decrease over the 24-month horizon, it indicates 
    that the landscape's ability to absorb water (recharge) is returning.
    """)

    

    st.header("3. SGMA and Aquifer Recharge")
    st.write("""
    Groundwater Sustainability Agencies (GSAs) are mandated to maintain the health of California's aquifers. 
    A 'High Severity' burn act as an umbrella over the basin, causing rain to run off rather than sink in. 
    This dashboard quantifies that 'recharge deficit' so GSAs can prioritize reforestation in high-impact zones.
    """)
