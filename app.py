import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import ee
import json
from datetime import datetime, timedelta
import zipfile
import os

# ==========================================
# 1. PAGE SETUP & ARCHITECTURE
# ==========================================
st.set_page_config(page_title="PF-WRP | Post-Fire Watershed Risk Portal", layout="wide")

st.sidebar.title("PF-WRP Navigation")
page = st.sidebar.radio("Select Module:", [
    "1. Incident Briefing", 
    "2. Spatial Modeling Lab", 
    "3. Watershed Loading (Phase 2 & 3)"
])

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
        st.error(f"Earth Engine Initialization Error: {e}")

# ==========================================
# 3. ROBUST CLOUD DATA LOADER
# ==========================================
@st.cache_data
def fetch_and_extract_fire_data():
    zip_path = 'Master_Fire_Dataset.geojson.zip'
    extract_dir = 'temp_fire_data_v4'
    
    try:
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    if not member.startswith('__MACOSX') and member.endswith('.geojson'):
                        zip_ref.extract(member, extract_dir)
        
        for file in os.listdir(extract_dir):
            if file.endswith('.geojson'):
                geojson_path = os.path.join(extract_dir, file)
                fires = gpd.read_file(geojson_path)
                fires = fires.dissolve(by='incident_n').reset_index()
                return fires.to_crs(epsg=4326)
        raise FileNotFoundError("No .geojson file found in archive.")
    except Exception as e:
        st.error(f"Failed to load perimeter data: {e}")
        return gpd.GeoDataFrame()

# ==========================================
# GLOBAL FIRE SELECTION & DATE PARSING
# ==========================================
cal_fires = fetch_and_extract_fire_data()

if not cal_fires.empty:
    name_col = 'incident_n' if 'incident_n' in cal_fires.columns else cal_fires.columns[0]
    fire_series = cal_fires[name_col]
    if 'mission' in cal_fires.columns:
        fire_series = fire_series.fillna(cal_fires['mission'])
    fire_list = sorted(fire_series.dropna().astype(str).unique())
    selected_fire = st.sidebar.selectbox("Select Wildfire Perimeter", fire_list)
    fire_data = cal_fires[cal_fires[name_col] == selected_fire]
    
    ignition_date = datetime(2021, 1, 1)
    for col in ['START_DATE', 'ALARM_DATE', 'alarm_date', 'cont_date']:
        if col in fire_data.columns and not pd.isna(fire_data[col].iloc[0]):
            try:
                ignition_date = pd.to_datetime(fire_data[col].iloc[0]).to_pydatetime()
                break
            except: continue

    # ADJUSTED DATES: 10 to 90 days post-fire to avoid winter cloud cover masking out the burn scar
    pre_fire_start = (ignition_date - timedelta(days=365)).strftime('%Y-%m-%d')
    pre_fire_end = (ignition_date - timedelta(days=1)).strftime('%Y-%m-%d')
    post_fire_start = (ignition_date + timedelta(days=10)).strftime('%Y-%m-%d')
    post_fire_end = (ignition_date + timedelta(days=90)).strftime('%Y-%m-%d')

    area = ee.FeatureCollection(fire_data.__geo_interface__)
    centroid = fire_data.to_crs(epsg=3310).geometry.centroid.to_crs(epsg=4326).iloc[0]
else:
    st.error("No fire perimeters loaded.")
    st.stop()

def mask_s2_clouds(image):
    qa = image.select('QA60')
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask).divide(10000)

# ==========================================
# PAGE 1: INCIDENT BRIEFING
# ==========================================
if page == "1. Incident Briefing":
    st.title(f"Incident Briefing: {selected_fire}")
    total_ac = (fire_data.to_crs(epsg=3310).area.sum()) * 0.000247105
    st.metric("Total Acres Burned", f"{total_ac:,.0f} ac")
    st.metric("Estimated Ignition Date", ignition_date.strftime('%B %d, %Y'))
    
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles="CartoDB positron")
    folium.GeoJson(fire_data.geometry, style_function=lambda x: {'fillColor': 'red', 'color': 'darkred', 'weight': 2, 'fillOpacity': 0.4}).add_to(m)
    st_folium(m, use_container_width=True, height=500)

# ==========================================
# PAGE 2: SPATIAL MODELING LAB
# ==========================================
elif page == "2. Spatial Modeling Lab":
    st.title("Spatial Modeling Lab (Engineering View)")
    
    SLOPE_LIMIT = 25
    DNBR_THRESHOLD = 0.15  # Adjusted back to your original baseline to ensure detection
    
    st.sidebar.info(f"**Critical Slope:** > {SLOPE_LIMIT} Degrees\n\n**Severity (dNBR):** > {DNBR_THRESHOLD}\n\n**Concavity:** Zero-Order Basins")
    with st.sidebar.expander("Methodology & Reasoning"):
        st.write("""
        **Gravitational Energy:** Slopes > 25 Degrees provide necessary velocity.
        **Burn Severity:** dNBR > 0.15 isolates hydrophobic soil sealing.
        **Topographic Concavity:** Debris flows do not initiate on flat ridges. The concavity kernel isolates ravines and hollows that actively funnel water inward, serving as the critical geomorphic trigger.
        """)

    show_risk = st.sidebar.checkbox("Hazard Intersection (Risk)", value=True)
    show_slope = st.sidebar.checkbox("Topographic Velocity (Slope)", value=False)
    show_concavity = st.sidebar.checkbox("Topographic Concavity (Hollows)", value=False) 
    show_severity = st.sidebar.checkbox("Burn Severity (dNBR)", value=False)
    show_soils = st.sidebar.checkbox("Soil Erodibility (K-Factor)", value=False)
    show_streams = st.sidebar.checkbox("HydroSHEDS Stream Routing", value=True)
    show_roads = st.sidebar.checkbox("TIGER Roads", value=True)

    with st.spinner("Compiling Spatial Intersection Data..."):
        dem = ee.Image("USGS/SRTMGL1_003")
        slope = ee.Terrain.slope(dem).clip(area)
        slope_mask = slope.gte(SLOPE_LIMIT)

        # FIXED CONCAVITY: Pixel-level focal mean prevents dynamic zoom errors. Target is -3 meters depth.
        local_mean = dem.focal_mean(radius=50, units='meters').clip(area)
        concavity_mask = dem.subtract(local_mean).lt(-3) 

        s2_pre = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(area).filterDate(pre_fire_start, pre_fire_end).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)).map(mask_s2_clouds).median().clip(area)
        s2_post = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(area).filterDate(post_fire_start, post_fire_end).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)).map(mask_s2_clouds).median().clip(area)
        dnbr = s2_pre.normalizedDifference(['B8', 'B12']).subtract(s2_post.normalizedDifference(['B8', 'B12']))
        severity_mask = dnbr.gte(DNBR_THRESHOLD)

        erodible_soils = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0').clip(area).lt(11).selfMask()
        
        hazard_intersection = slope_mask.And(concavity_mask).And(severity_mask).And(erodible_soils).selfMask()

        roads_img = ee.Image(0).mask(0).paint(ee.FeatureCollection("TIGER/2016/Roads").filterBounds(area), 1, 2)
        streams_img = ee.Image(0).mask(0).paint(ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(area), 1, 1)

        m2 = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Hybrid')
        if show_slope: folium.TileLayer(tiles=slope_mask.selfMask().getMapId({'palette':['yellow'],'opacity':0.4})['tile_fetcher'].url_format, attr='USGS', name='Slope').add_to(m2)
        if show_concavity: folium.TileLayer(tiles=concavity_mask.selfMask().getMapId({'palette':['#8e44ad'],'opacity':0.6})['tile_fetcher'].url_format, attr='USGS', name='Concavity').add_to(m2)
        if show_severity: folium.TileLayer(tiles=severity_mask.selfMask().getMapId({'palette':['red'],'opacity':0.4})['tile_fetcher'].url_format, attr='ESA', name='Severity').add_to(m2)
        if show_soils: folium.TileLayer(tiles=erodible_soils.getMapId({'palette':['#800026'],'opacity':0.4})['tile_fetcher'].url_format, attr='Soil', name='Soils').add_to(m2)
        if show_streams: folium.TileLayer(tiles=streams_img.getMapId({'palette':['#3498db']})['tile_fetcher'].url_format, attr='WWF', name='Streams').add_to(m2)
        if show_roads: folium.TileLayer(tiles=roads_img.getMapId({'palette':['#2ecc71']})['tile_fetcher'].url_format, attr='TIGER', name='Roads').add_to(m2)
        if show_risk: folium.TileLayer(tiles=hazard_intersection.getMapId({'palette':['#ff7b00'],'opacity':0.9})['tile_fetcher'].url_format, attr='GEE', name='Risk').add_to(m2)

        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; background-color: white; border:2px solid grey; z-index:9999; font-size:12px; padding: 10px;">
        <b>PF-WRP Legend</b><br>
        <i style="background:#ff7b00; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Hazard Intersection<br>
        <i style="background:yellow; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Critical Slope<br>
        <i style="background:#8e44ad; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Topographic Concavity<br>
        <i style="background:red; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Severe dNBR<br>
        <i style="background:#800026; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Erodible Soils<br>
        <i style="background:#3498db; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Stream Routing<br>
        <i style="background:#2ecc71; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Infrastructure<br>
        </div>"""
        m2.get_root().html.add_child(folium.Element(legend_html))

        toggle_key = f"lab_{selected_fire}_v{show_risk}{show_slope}{show_concavity}{show_severity}{show_soils}{show_streams}{show_roads}"
        st_folium(m2, use_container_width=True, height=700, key=toggle_key)

# ==========================================
# PAGE 3: WATERSHED LOADING (PHASE 2 & 3)
# ==========================================
elif page == "3. Watershed Loading (Phase 2 & 3)":
    st.title("Watershed Loading (Vulnerability Matrix)")
    
    with st.spinner("Executing zonal statistics and sediment math across HUC-12 basins via Earth Engine..."):
        SLOPE_LIMIT = 25
        DNBR_THRESHOLD = 0.15

        dem = ee.Image("USGS/SRTMGL1_003")
        slope_mask = ee.Terrain.slope(dem).clip(area).gte(SLOPE_LIMIT)
        
        local_mean = dem.focal_mean(radius=50, units='meters').clip(area)
        concavity_mask = dem.subtract(local_mean).lt(-3)

        s2_pre = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(area).filterDate(pre_fire_start, pre_fire_end).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)).map(mask_s2_clouds).median().clip(area)
        s2_post = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(area).filterDate(post_fire_start, post_fire_end).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)).map(mask_s2_clouds).median().clip(area)
        severity_mask = s2_pre.normalizedDifference(['B8', 'B12']).subtract(s2_post.normalizedDifference(['B8', 'B12'])).gte(DNBR_THRESHOLD)

        erodible_soils = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0').clip(area).lt(11).selfMask()
        
        hazard_intersection = slope_mask.And(concavity_mask).And(severity_mask).And(erodible_soils).selfMask()
        hazard_area_img = hazard_intersection.multiply(ee.Image.pixelArea())

        precip = ee.ImageCollection("NASA/GPM_L3/IMERG_V06").filterDate(post_fire_start, post_fire_end).select('precipitationCal').max().clip(area)

        huc12 = ee.FeatureCollection("USGS/WBD/2017/HUC12").filterBounds(area)

        def process_basin(f):
            geom = f.geometry()
            h_area = hazard_area_img.reduceRegion(reducer=ee.Reducer.sum(), geometry=geom, scale=30, maxPixels=1e9).get('slope')
            p_mean = precip.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=100, maxPixels=1e9).get('precipitationCal')
            return f.set('hazard_area_m2', h_area).set('peak_rain_mm', p_mean)

        huc12_processed = huc12.map(process_basin)
        huc_data = huc12_processed.getInfo()

        basin_results = []
        for feature in huc_data['features']:
            props = feature['properties']
            name = props.get('name', 'Unknown Basin')
            huc12_id = props.get('huc12', 'Unknown ID')

            raw_area = props.get('hazard_area_m2')
            raw_rain = props.get('peak_rain_mm')

            h_area_m2 = float(raw_area) if raw_area is not None else 0.0
            p_rain_mm = float(raw_rain) if raw_rain is not None else 0.0

            rain_depth_m = (p_rain_mm * 24) / 1000.0
            k_factor = 0.35
            sediment_yield_m3 = h_area_m2 * rain_depth_m * k_factor

            basin_results.append({
                'HUC12_ID': huc12_id,
                'Basin Name': name,
                'Hazard Area (Acres)': h_area_m2 * 0.000247105,
                'Peak Rain (mm/hr)': p_rain_mm,
                'Sediment Yield (m³)': sediment_yield_m3
            })

        df_results = pd.DataFrame(basin_results).sort_values(by='Sediment Yield (m³)', ascending=False)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Watershed Matrix")
            st.dataframe(df_results[['Basin Name', 'Sediment Yield (m³)', 'Hazard Area (Acres)']].style.format({"Sediment Yield (m³)": "{:,.0f}", "Hazard Area (Acres)": "{:,.1f}"}), use_container_width=True)
            st.info("**Sediment Math Engine:**\nCalculated using the spatial intersection area ($m^2$) multiplied by the modeled 24-hour storm depth ($m$) and a K-Factor proxy ($0.35$) for erodible soils. *Note: Area calculation utilizes the Topographic Concavity filter to isolate zero-order basins.*")
            
            st.markdown("---")
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            clean_fire_name = selected_fire.replace(" ", "_")
            st.download_button(
                label="Download Executive Report (CSV)",
                data=csv_data,
                file_name=f"{clean_fire_name}_Watershed_Vulnerability_Report.csv",
                mime="text/csv",
                key=f"export_csv_{selected_fire}", 
                use_container_width=True
            )

            st.markdown("---")
            st.success("**Stream Transport Dynamics:**\nLine thickness and color represent the **Average Long-Term Discharge** (Flow Accumulation proxy). \n* **Thin / Cyan:** Headwater streams (low discharge).\n* **Thick / Navy:** Major transport arteries (massive discharge). \n\n*Rivers cutting through high-yield (dark red) basins act as the primary drainage funnel and are at extreme risk of debris flow inundation.*")

        with col2:
            st.markdown("### Basin Choropleth & Stream Transport")
            
            gdf = gpd.GeoDataFrame.from_features(huc_data['features'])
            gdf.set_crs(epsg=4326, inplace=True)
            gdf = gdf.merge(df_results, left_on='huc12', right_on='HUC12_ID')

            m3 = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles='CartoDB positron')

            folium.Choropleth(
                geo_data=gdf,
                name='Sediment Yield',
                data=df_results,
                columns=['HUC12_ID', 'Sediment Yield (m³)', 'Basin Name'],
                key_on='feature.properties.huc12',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.3,
                legend_name='Estimated Sediment Yield (Cubic Meters)'
            ).add_to(m3)

            streams = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(area)
            
            def style_streams(f):
                discharge = ee.Number(f.get('DIS_AV_CMS')).add(1) 
                log_dis = discharge.log10()
                line_width = log_dis.multiply(1.5).add(0.5)
                return f.set('acc_width', line_width).set('acc_color', log_dis)

            styled_streams = streams.map(style_streams)
            stream_img = ee.Image(0).mask(0).paint(styled_streams, 'acc_color', 'acc_width')
            
            stream_vis = stream_img.getMapId({
                'min': 0, 
                'max': 2.5, 
                'palette': ['#00b4d8', '#0077b6', '#03045e'] 
            })
            
            folium.TileLayer(
                tiles=stream_vis['tile_fetcher'].url_format, 
                attr='WWF', 
                name='Stream Transport (Discharge)', 
                overlay=True
            ).add_to(m3)

            tooltip = folium.GeoJsonTooltip(
                fields=['name', 'Sediment Yield (m³)', 'Hazard Area (Acres)'],
                aliases=['Basin:', 'Est. Yield (m³):', 'Hazard Area (Acres):'],
                localize=True
            )
            folium.GeoJson(
                gdf,
                style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent'},
                tooltip=tooltip
            ).add_to(m3)

            st_folium(m3, use_container_width=True, height=600, key=f"huc12_{selected_fire}")
