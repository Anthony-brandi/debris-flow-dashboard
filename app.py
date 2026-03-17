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
st.set_page_config(page_title="Advanced Watershed Risk Modeler", layout="wide")

@st.cache_data
def load_fire_perimeters():
    path = 'CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view/CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view.shp'
    fires = gpd.read_file(path)
    
    # Robust multi-column date detection (First-Person Design)
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
# 3. SIDEBAR NAVIGATION & PARAMETERS
# ==========================================
st.sidebar.title("Analytical Controls")
page = st.sidebar.radio("Navigation", ["Interactive Risk Map", "User Manual", "Technical Documentation"])

if page == "Interactive Risk Map":
    try:
        cal_fires = load_fire_perimeters()
        fire_list = sorted(cal_fires['incident_n'].dropna().unique())
        selected_fire = st.sidebar.selectbox("Select Wildfire Incident", fire_list)
        fire_data = cal_fires[cal_fires['incident_n'] == selected_fire]
        
        actual_date = fire_data['final_date'].iloc[0]
        st.sidebar.info(f"My Analysis Baseline: {actual_date.strftime('%B %d, %Y')}")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Model Parameters")
        recovery_months = st.sidebar.select_slider("Successional Window (Months Post-Fire)", options=[1, 6, 12, 18, 24], value=1)
        slope_limit = st.sidebar.slider("Critical Slope Threshold (Degrees)", 10, 45, 27)
        analyze_btn = st.sidebar.checkbox("Execute Spatial Metrics", value=True)

        st.sidebar.markdown("---")
        st.sidebar.subheader("Map Layer Toggles")
        show_recovery = st.sidebar.checkbox("Burn Severity (dNBR)", value=True)
        show_precip = st.sidebar.checkbox("Precipitation (NASA GPM)", value=False)
        show_risk = st.sidebar.checkbox("Hazard Intersection (Orange)", value=True)
        show_streams = st.sidebar.checkbox("Stream Valleys (HydroSHEDS)", value=True)
        show_watersheds = st.sidebar.checkbox("Watershed Boundaries (HUC-12)", value=True)
        show_infra = st.sidebar.checkbox("Road Vulnerability (TIGER)", value=True)
        
        basemap = st.sidebar.radio("Reference Style", ["Satellite", "Terrain"])
        tile_url = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}' if basemap == "Satellite" else 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}'

        # ==========================================
        # 4. QUANTITATIVE ANALYTICS (Addressing Professor's Request)
        # ==========================================
        st.title(f"Hydrologic Risk Assessment: {selected_fire}")
        
        if analyze_btn:
            with st.spinner("Calculating Watershed-Level Hazard Area..."):
                area = ee.FeatureCollection(fire_data.__geo_interface__)
                
                # --- SATELLITE & TOPOGRAPHY DATA ---
                pre_date = ee.Date(actual_date.strftime('%Y-%m-%d')).advance(-1, 'year')
                target_date = ee.Date(actual_date.strftime('%Y-%m-%d')).advance(recovery_months, 'month')

                def get_nbr_median(date_obj):
                    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area)\
                        .filterDate(date_obj.advance(-1, 'month'), date_obj.advance(1, 'month'))\
                        .median().clip(area).normalizedDifference(['B8', 'B12'])

                dnbr = get_nbr_median(pre_date).subtract(get_nbr_median(target_date))
                dem = ee.Image("USGS/SRTMGL1_003")
                slope = ee.Terrain.slope(dem).clip(area)
                precip = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterBounds(area)\
                    .filterDate(target_date.advance(-1, 'month'), target_date)\
                    .select('precipitation').sum().clip(area)

                # --- DEFINE HAZARD ZONE & CALCULATE AREA PER WATERSHED ---
                # Hazard = Burn > 0.44 AND Slope > User Threshold
                hazard_mask = slope.gte(slope_limit).And(dnbr.gt(0.44))
                hazard_area_img = hazard_mask.multiply(ee.Image.pixelArea())

                # Get USGS Watersheds
                huc12 = ee.FeatureCollection("USGS/WBD/2017/HUC12").filterBounds(area)
                
                # Spatial Reduction to get acreage per watershed
                def calc_hazard_per_watershed(feature):
                    stats = hazard_area_img.reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=feature.geometry(),
                        scale=30,
                        maxPixels=1e9
                    )
                    return feature.set('hazard_acres', ee.Number(stats.get('slope')).multiply(0.000247105))

                huc12_stats = huc12.map(calc_hazard_per_watershed).getInfo()
                
                # Create the Summary Table
                ws_data = []
                for f in huc12_stats['features']:
                    props = f['properties']
                    ws_data.append({
                        "Watershed Name": props.get('name'), 
                        "Hazard Footprint (Acres)": round(props.get('hazard_acres', 0), 2)
                    })
                
                df_ws = pd.DataFrame(ws_data).sort_values(by="Hazard Footprint (Acres)", ascending=False)

                # --- DISPLAY METRICS ---
                m1, m2, m3 = st.columns(3)
                total_hazard = df_ws["Hazard Footprint (Acres)"].sum()
                m1.metric("Total Perimeter Area", f"{(fire_data.to_crs(epsg=3310).area.sum()) * 0.000247105:,.1f} Ac")
                m2.metric("Active Hazard Footprint", f"{total_hazard:,.1f} Ac")
                m3.metric("Rainfall (Window)", f"{precip.reduceRegion(ee.Reducer.mean(), area.geometry(), 1000).getInfo().get('precipitation', 0):,.1f} mm")

                # Sub-Watershed Table (The "Follow-on" Request)
                st.subheader("Regional Vulnerability: Hazard Area per Watershed")
                st.dataframe(df_ws, use_container_width=True)
                st.markdown("---")

                # --- MAP RENDERING ---
                centroid_point = fire_data.geometry.centroid.iloc[0]
                m = folium.Map(location=[centroid_point.y, centroid_point.x], zoom_start=12, tiles=tile_url, attr="Google")
                
                # Professional Legend
                legend_html = f"""
                <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; background-color: white; border:2px solid black; z-index:9999; font-size:12px; padding: 10px; border-radius: 5px;">
                <b style="color:black;">Analysis Legend</b><br>
                <i style="background:red; width:10px; height:10px; float:left; margin-right:5px; border:1px solid black;"></i> <span style="color:black;">Perimeter</span><br>
                <i style="background:#bd0026; width:10px; height:10px; float:left; margin-right:5px;"></i> <span style="color:black;">Burn Scar (High)</span><br>
                <i style="background:#ff7b00; width:10px; height:10px; float:left; margin-right:5px;"></i> <span style="color:black;">Hazard Hotspot</span><br>
                <i style="background:#3498db; width:10px; height:2px; float:left; margin-right:5px; margin-top:4px;"></i> <span style="color:black;">Stream Channel</span><br>
                <i style="border: 1px solid purple; width:10px; height:2px; float:left; margin-right:5px; margin-top:4px;"></i> <span style="color:black;">Watershed Boundary</span>
                </div>"""
                m.get_root().html.add_child(folium.Element(legend_html))

                if show_recovery:
                    vis = {'min': 0.1, 'max': 0.5, 'palette': ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']}
                    folium.TileLayer(tiles=dnbr.updateMask(dnbr.gt(0.1)).getMapId(vis)['tile_fetcher'].url_format, attr='S2', name='Burn Status', opacity=0.7).add_to(m)

                if show_precip:
                    p_vis = {'min': 1, 'max': 150, 'palette': ['#f7fbff','#deebf7','#9ecae1','#4292c6','#084594']}
                    folium.TileLayer(tiles=precip.updateMask(precip.gt(1)).getMapId(p_vis)['tile_fetcher'].url_format, attr='NASA', name='Rainfall', opacity=0.5).add_to(m)

                if show_risk:
                    folium.TileLayer(tiles=hazard_mask.updateMask(hazard_mask).getMapId({'palette':['#ff7b00']})['tile_fetcher'].url_format, attr='GEE', name='Risk').add_to(m)

                if show_watersheds:
                    w_outline = ee.Image(0).mask(0).paint(huc12, 'purple', 2)
                    folium.TileLayer(tiles=w_outline.getMapId({'palette':['purple']})['tile_fetcher'].url_format, attr='USGS', name='Watersheds').add_to(m)

                if show_streams:
                    # HYDROGRAPHIC ROUTING (Addressing Stream Valleys)
                    streams = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(area)
                    s_img = ee.Image(0).mask(0).paint(streams, '#3498db', 1.5)
                    folium.TileLayer(tiles=s_img.getMapId({'palette':['#3498db']})['tile_fetcher'].url_format, attr='HydroSHEDS', name='Streams').add_to(m)

                if show_infra:
                    roads = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(area)
                    r_img = ee.Image(0).mask(0).paint(roads, '#2ecc71', 1.5)
                    folium.TileLayer(tiles=r_img.getMapId({'palette':['#2ecc71']})['tile_fetcher'].url_format, attr='TIGER', name='Roads').add_to(m)

                folium.GeoJson(fire_data.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 3}).add_to(m)
                st_folium(m, use_container_width=True, height=750)

        else:
            m = folium.Map(location=[fire_data.geometry.centroid.iloc[0].y, fire_data.geometry.centroid.iloc[0].x], zoom_start=12, tiles=tile_url, attr="Google")
            folium.GeoJson(fire_data.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 2}).add_to(m)
            st_folium(m, use_container_width=True, height=750)

    except Exception as e:
        st.error(f"Modeling Error: {e}")

# ==========================================
# 5. USER MANUAL (MY PERSPECTIVE)
# ==========================================
elif page == "User Manual":
    st.title("User Manual: My Operational Guide")
    st.markdown("---")
    st.info("Regional Statistics: I have added a data table that calculates the exact acreage of hazard zones for every watershed (HUC-12) unit.")
    st.header("My Operational Workflow")
    st.write("""
    1.  **Selecting the Event:** I start by choosing a fire perimeter. My code automatically anchors the 'Pre-Fire Baseline' to exactly 365 days prior to ignition.
    2.  **Tracking Recovery:** I use the **Successional Window** slider to observe how vegetation regrowth reduces the hydrophobic scar over time.
    3.  **Regional Metrics:** When you click 'Execute', my model calculates the specific hazard footprint for every individual watershed. I did this to move beyond visual maps and provide hard data on which canyons are the most dangerous.
    4.  **Channel Analysis:** I have integrated **Hydrographic Stream Routing** to show where debris flows will likely channelize after leaving the steep initiation zones.
    """)

# ==========================================
# 6. TECHNICAL DOCUMENTATION (MY PERSPECTIVE)
# ==========================================
elif page == "Technical Documentation":
    st.title("Technical Documentation: The Science of the 'Funnel'")
    st.markdown("---")
    st.subheader("I. Quantitative Watershed Metrics")
    st.write("To move beyond simple visualization, I utilize the `reduceRegion` function to quantify the total acreage of high-severity burn scars intersecting with steep topography within each USGS HUC-12 boundary. This allows for a ranked risk assessment of the drainage basins.")
    
    st.subheader("II. Hydrographic Routing & Stream Valleys")
    st.write("Debris flows are historically channelized events. I have integrated the HydroSHEDS river network to identify the primary 'valleys' that water and debris will follow once they leave the high-gradient initiation zones. This helps in identifying the specific canyons at risk of flooding.")
    
    st.subheader("III. The Sponge vs. The Funnel")
    st.write("I pull in NASA GPM Rainfall data to see exactly how much water hit the watershed 'funnel.' When rain hits bare, charred soil on a steep slope, it doesn't soak in; it gains speed. This gain in velocity is what triggers the mass movement of debris seen in the Gifford fire study.")
