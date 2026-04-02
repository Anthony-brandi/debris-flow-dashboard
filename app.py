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
import math

# ==========================================
# 1. PAGE SETUP & ARCHITECTURE
# ==========================================
st.set_page_config(page_title="PF-WRP | Post-Fire Watershed Risk Portal", layout="wide")

st.sidebar.title("PF-WRP Navigation")
page = st.sidebar.radio("Select Module:", [
    "1. Incident Briefing", 
    "2. Spatial Modeling Lab", 
    "3. Watershed Loading (Phase 2 & 3)",
    "4. Documentation & Methodology"
])

# ==========================================
# 2. ISOLATED DEBRIS FLOW MATH ENGINE
# ==========================================
def calculate_gartner_volume(b23_m2, hm_m2, r15_mmhr):
    b23_km2 = (b23_m2 / 1_000_000) if b23_m2 else 0.0
    hm_km2 = (hm_m2 / 1_000_000) if hm_m2 else 0.0
    r15 = float(r15_mmhr)

    if b23_km2 <= 0.001 or r15 <= 0:
        return 0.0

    try:
        ln_v = 4.22 + (0.13 * math.log(b23_km2)) + (0.36 * math.log(r15)) + (0.39 * math.sqrt(hm_km2))
        return math.exp(ln_v) 
    except ValueError:
        return 0.0

# ==========================================
# 3. GEE INITIALIZATION
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
# 4. ROBUST CLOUD DATA LOADER (UPGRADED FOR DYNAMIC COLUMNS)
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
        
        gdfs = []
        for file in os.listdir(extract_dir):
            if file.endswith('.geojson'):
                geojson_path = os.path.join(extract_dir, file)
                fires = gpd.read_file(geojson_path)
                
                # DYNAMIC COLUMN DISCOVERY: Find whatever this specific file calls the fire name
                possible_names = ['incident_n', 'FIRE_NAME', 'Fire_Name', 'Name', 'name', 'mission']
                actual_name_col = next((col for col in possible_names if col in fires.columns), fires.columns[0])
                
                # Standardize the column name to 'incident_n' so the app logic stays perfectly intact
                fires = fires.rename(columns={actual_name_col: 'incident_n'})
                
                fires = fires.dissolve(by='incident_n').reset_index()
                gdfs.append(fires.to_crs(epsg=4326))
                
        if gdfs:
            return pd.concat(gdfs, ignore_index=True)
            
        raise FileNotFoundError("No .geojson file found in archive.")
    except Exception as e:
        st.error(f"Failed to load perimeter data: {e}")
        return gpd.GeoDataFrame()

# ==========================================
# GLOBAL FIRE SELECTION & DATA CLEANING
# ==========================================
cal_fires = fetch_and_extract_fire_data()

if not cal_fires.empty:
    name_col = 'incident_n' if 'incident_n' in cal_fires.columns else cal_fires.columns[0]
    fire_series = cal_fires[name_col]
    if 'mission' in cal_fires.columns:
        fire_series = fire_series.fillna(cal_fires['mission'])
        
    raw_fire_list = sorted(fire_series.dropna().astype(str).unique())
    clean_fire_list = [f for f in raw_fire_list if not f.replace('-', '').replace(' ', '').isnumeric() and len(f) > 3]
    
    selected_fire = st.sidebar.selectbox("Select Wildfire Perimeter", clean_fire_list)
    fire_data = cal_fires[cal_fires[name_col] == selected_fire]
    
    ignition_date = datetime(2021, 1, 1)
    for col in ['START_DATE', 'ALARM_DATE', 'alarm_date', 'cont_date']:
        if col in fire_data.columns and not pd.isna(fire_data[col].iloc[0]):
            try:
                ignition_date = pd.to_datetime(fire_data[col].iloc[0]).to_pydatetime()
                break
            except: continue

    pre_fire_start = (ignition_date - timedelta(days=365)).strftime('%Y-%m-%d')
    pre_fire_end = (ignition_date - timedelta(days=1)).strftime('%Y-%m-%d')
    post_fire_start = (ignition_date + timedelta(days=1)).strftime('%Y-%m-%d')
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
    
    SLOPE_LIMIT = 23 
    DNBR_THRESHOLD = 0.15
    
    st.sidebar.info(f"**Critical Slope:** >= {SLOPE_LIMIT} Degrees\n\n**Severity (dNBR):** > {DNBR_THRESHOLD}")
    
    st.sidebar.markdown("### Map Controls")
    basemap_choice = st.sidebar.radio("Reference Basemap:", ["Satellite", "Terrain", "Minimal"])
    
    st.sidebar.markdown("### Layer Visibility")
    show_risk = st.sidebar.checkbox("Composite Hazard Score", value=True)
    show_slope = st.sidebar.checkbox("Topographic Velocity (Slope)", value=False)
    show_concavity = st.sidebar.checkbox("Initiation Points (Hollows)", value=False) 
    show_severity = st.sidebar.checkbox("Burn Severity (dNBR)", value=False)
    show_soils = st.sidebar.checkbox("Soil Erodibility (Sand %)", value=False)
    show_streams = st.sidebar.checkbox("HydroSHEDS Stream Routing", value=True)
    show_roads = st.sidebar.checkbox("TIGER Roads", value=True)

    with st.spinner("Compiling Spatial Intersection Data..."):
        dem = ee.Image("USGS/SRTMGL1_003")
        slope = ee.Terrain.slope(dem).clip(area)
        slope_mask = slope.gte(SLOPE_LIMIT)

        local_mean = dem.focal_mean(radius=50, units='meters').clip(area)
        concavity_mask = dem.subtract(local_mean).lt(-3) 

        dummy_s2 = ee.Image.constant([0.0001, 0.0001, 0]).rename(['B8', 'B12', 'QA60'])
        
        s2_pre_col = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(area).filterDate(pre_fire_start, pre_fire_end).merge(ee.ImageCollection([dummy_s2]))
        s2_post_col = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(area).filterDate(post_fire_start, post_fire_end).merge(ee.ImageCollection([dummy_s2]))
        
        s2_pre = s2_pre_col.map(mask_s2_clouds).median().clip(area)
        s2_post = s2_post_col.map(mask_s2_clouds).median().clip(area)
        
        dnbr = s2_pre.normalizedDifference(['B8', 'B12']).subtract(s2_post.normalizedDifference(['B8', 'B12']))
        severity_mask = dnbr.gte(DNBR_THRESHOLD)

        erodible_soils = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select('b0').clip(area)
        soil_risk_mask = erodible_soils.gte(40) 
        
        slope_safe = ee.Image(slope_mask).unmask(0).select(0).rename('val').toInt()
        sev_safe = ee.Image(severity_mask).unmask(0).select(0).rename('val').toInt()
        soil_safe = ee.Image(soil_risk_mask).unmask(0).select(0).rename('val').toInt()

        risk_score = slope_safe.add(sev_safe).add(soil_safe)
        hazard_intersection = risk_score.gte(2).selfMask() 

        roads_img = ee.Image(0).mask(0).paint(ee.FeatureCollection("TIGER/2016/Roads").filterBounds(area), 1, 2)
        streams_img = ee.Image(0).mask(0).paint(ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(area), 1, 1)

        if basemap_choice == "Satellite":
            m2 = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Hybrid')
            perimeter_color = 'white'
        elif basemap_choice == "Terrain":
            m2 = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles='https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}', attr='Google Terrain')
            perimeter_color = 'black'
        else:
            m2 = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles='CartoDB positron')
            perimeter_color = 'black'

        folium.GeoJson(
            fire_data.geometry, 
            style_function=lambda x: {'fillColor': 'transparent', 'color': perimeter_color, 'weight': 2.5, 'dashArray': '5, 5'}
        ).add_to(m2)

        C_RISK = '#FF5733'
        C_SLOPE = 'yellow'
        C_CONCAVITY = '#8e44ad'
        C_SEVERITY = 'red'
        C_STREAMS = '#3498db'
        C_ROADS = '#2ecc71'

        if show_slope: folium.TileLayer(tiles=slope_mask.selfMask().getMapId({'palette':[C_SLOPE]})['tile_fetcher'].url_format, attr='USGS', name='Slope', opacity=0.4).add_to(m2)
        if show_concavity: folium.TileLayer(tiles=concavity_mask.selfMask().getMapId({'palette':[C_CONCAVITY]})['tile_fetcher'].url_format, attr='USGS', name='Concavity', opacity=0.6).add_to(m2)
        if show_severity: folium.TileLayer(tiles=severity_mask.selfMask().getMapId({'palette':[C_SEVERITY]})['tile_fetcher'].url_format, attr='ESA', name='Severity', opacity=0.4).add_to(m2)
        if show_soils: folium.TileLayer(tiles=erodible_soils.getMapId({'min': 10, 'max': 80, 'palette': ['#f4a460', '#d2691e', '#8b4513']})['tile_fetcher'].url_format, attr='Soil', name='Soils', opacity=0.7).add_to(m2)
        if show_streams: folium.TileLayer(tiles=streams_img.getMapId({'palette':[C_STREAMS]})['tile_fetcher'].url_format, attr='WWF', name='Streams').add_to(m2)
        if show_roads: folium.TileLayer(tiles=roads_img.getMapId({'palette':[C_ROADS]})['tile_fetcher'].url_format, attr='TIGER', name='Roads').add_to(m2)
        if show_risk: folium.TileLayer(tiles=hazard_intersection.getMapId({'min': 1, 'max': 1, 'palette':[C_RISK]})['tile_fetcher'].url_format, attr='GEE', name='Risk', opacity=0.8).add_to(m2)

        legend_items = []
        if show_risk: legend_items.append(f'<i style="background:{C_RISK}; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Hazard Intersection<br>')
        if show_slope: legend_items.append(f'<i style="background:{C_SLOPE}; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Critical Slope<br>')
        if show_concavity: legend_items.append(f'<i style="background:{C_CONCAVITY}; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Initiation Hollows<br>')
        if show_severity: legend_items.append(f'<i style="background:{C_SEVERITY}; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Severe dNBR<br>')
        if show_soils: legend_items.append(f'<i style="background:linear-gradient(to right, #f4a460, #8b4513); width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Erodible Soils (Sand %)<br>')
        if show_streams: legend_items.append(f'<i style="background:{C_STREAMS}; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Stream Routing<br>')
        if show_roads: legend_items.append(f'<i style="background:{C_ROADS}; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Infrastructure<br>')
        
        legend_items.append(f'<i style="background:transparent; border: 2px dashed {perimeter_color}; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i> Fire Perimeter<br>')

        legend_html = f"""
        <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; background-color: white; border:2px solid grey; z-index:9999; font-size:12px; padding: 10px;">
        <b>PF-WRP Legend</b><br>
        {''.join(legend_items)}
        </div>"""
        
        m2.get_root().html.add_child(folium.Element(legend_html))

        toggle_key = f"lab_{selected_fire}_v{show_risk}{show_slope}{show_concavity}{show_severity}{show_soils}{show_streams}{show_roads}_{basemap_choice}"
        st_folium(m2, use_container_width=True, height=700, key=toggle_key)

# ==========================================
# PAGE 3: WATERSHED LOADING (PHASE 2 & 3)
# ==========================================
elif page == "3. Watershed Loading (Phase 2 & 3)":
    st.title("Watershed Loading (Predictive Vulnerability Matrix)")
    
    st.sidebar.markdown("### Operational Weather Inputs")
    design_storm_mmhr = st.sidebar.slider(
        "Design Storm (Peak 15-min Rainfall Intensity in mm/hr)", 
        min_value=10.0, max_value=60.0, value=24.0, step=2.0,
        help="CAL FIRE often evaluates baselines at 24mm/hr. Higher values simulate intense atmospheric rivers."
    )
    
    with st.spinner(f"Executing zonal statistics via Earth Engine & running Gartner 2014 Regression for {design_storm_mmhr} mm/hr..."):
        SLOPE_LIMIT = 23
        DNBR_THRESHOLD = 0.15

        dem = ee.Image("USGS/SRTMGL1_003")
        slope_mask = ee.Terrain.slope(dem).clip(area).gte(SLOPE_LIMIT)

        dummy_s2 = ee.Image.constant([0.0001, 0.0001, 0]).rename(['B8', 'B12', 'QA60'])
        
        s2_pre_col = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(area).filterDate(pre_fire_start, pre_fire_end).merge(ee.ImageCollection([dummy_s2]))
        s2_post_col = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(area).filterDate(post_fire_start, post_fire_end).merge(ee.ImageCollection([dummy_s2]))
        
        s2_pre = s2_pre_col.map(mask_s2_clouds).median().clip(area)
        s2_post = s2_post_col.map(mask_s2_clouds).median().clip(area)
        
        dnbr = s2_pre.normalizedDifference(['B8', 'B12']).subtract(s2_post.normalizedDifference(['B8', 'B12']))
        severity_mask = dnbr.gte(DNBR_THRESHOLD)

        b23_area_img = slope_mask.unmask(0).multiply(ee.Image.pixelArea()).rename('b23_m2')
        hm_area_img = severity_mask.unmask(0).multiply(ee.Image.pixelArea()).rename('hm_m2')
        combined_reducer_img = ee.Image.cat([b23_area_img, hm_area_img])

        huc12 = ee.FeatureCollection("USGS/WBD/2017/HUC12").filterBounds(area)

        huc12_processed = combined_reducer_img.reduceRegions(
            collection=huc12,
            reducer=ee.Reducer.sum(),
            scale=30,
            tileScale=4 
        )

        huc_data = huc12_processed.getInfo()

        basin_results = []
        for feature in huc_data['features']:
            props = feature['properties']
            name = props.get('name', 'Unknown Basin')
            huc12_id = props.get('huc12', 'Unknown ID')

            raw_b23 = props.get('b23_m2')
            raw_hm = props.get('hm_m2')

            b23_m2 = float(raw_b23) if raw_b23 is not None else 0.0
            hm_m2 = float(raw_hm) if raw_hm is not None else 0.0

            sediment_yield_m3 = calculate_gartner_volume(b23_m2, hm_m2, design_storm_mmhr)

            basin_results.append({
                'HUC12_ID': huc12_id,
                'Basin Name': name,
                'Critical Slope Area (Acres)': b23_m2 * 0.000247105,
                'Severe Burn Area (Acres)': hm_m2 * 0.000247105,
                'Simulated Storm (mm/hr)': design_storm_mmhr,
                'Sediment Yield (m³)': sediment_yield_m3
            })

        df_results = pd.DataFrame(basin_results).sort_values(by='Sediment Yield (m³)', ascending=False)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Watershed Matrix")
            st.dataframe(df_results[['Basin Name', 'Sediment Yield (m³)', 'Critical Slope Area (Acres)']].style.format({"Sediment Yield (m³)": "{:,.0f}", "Critical Slope Area (Acres)": "{:,.1f}"}), use_container_width=True)
            st.info("Sediment Math Engine:\nVolumes calculated using the USGS Gartner et al. (2014) empirical logistic regression model. Utilizing localized B23 (slope >= 23 degrees) and HM (Moderate/High Severity) areas against the predictive user-defined storm intensity.")
            
            st.markdown("---")
            clean_fire_name = selected_fire.replace(" ", "_")
            
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Executive Report (CSV)",
                data=csv_data,
                file_name=f"{clean_fire_name}_Watershed_Vulnerability_Report.csv",
                mime="text/csv",
                use_container_width=True
            )

            gdf_export = gpd.GeoDataFrame.from_features(huc_data['features'])
            gdf_export = gdf_export.merge(df_results, left_on='huc12', right_on='HUC12_ID')
            geojson_data = gdf_export.to_json()
            st.download_button(
                label="Download Operational Polygons (GeoJSON)",
                data=geojson_data,
                file_name=f"{clean_fire_name}_Debris_Flow_Hazards.geojson",
                mime="application/geo+json",
                use_container_width=True
            )

            st.markdown("---")
            st.success("Stream Transport Dynamics:\nLine thickness represents Average Long-Term Discharge. Rivers cutting through high-yield (dark red) basins act as the primary drainage funnel and are at extreme risk of inundation.")

        with col2:
            st.markdown("### Predictive Basin Choropleth")
            
            gdf = gpd.GeoDataFrame.from_features(huc_data['features'])
            gdf.set_crs(epsg=4326, inplace=True)
            gdf = gdf.merge(df_results, left_on='huc12', right_on='HUC12_ID')

            m3 = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles='CartoDB positron')

            folium.GeoJson(
                fire_data.geometry, 
                style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 2, 'dashArray': '5, 5'}
            ).add_to(m3)

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
                fields=['name', 'Sediment Yield (m³)', 'Critical Slope Area (Acres)'],
                aliases=['Basin:', 'Est. Yield (m³):', 'B23 Area (Acres):'],
                localize=True
            )
            folium.GeoJson(
                gdf,
                style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent'},
                tooltip=tooltip
            ).add_to(m3)

            st_folium(m3, use_container_width=True, height=600, key=f"huc12_{selected_fire}")

# ==========================================
# PAGE 4: DOCUMENTATION & METHODOLOGY
# ==========================================
elif page == "4. Documentation & Methodology":
    st.title("System Documentation & Scientific Methodology")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Operational User Guide", "Scientific Methodology"])
    
    with tab1:
        st.markdown("### Incident Command Workflow")
        
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Debris_flow.jpg/800px-Debris_flow.jpg", caption="Debris flow inundation zone.", use_container_width=True)
        
        st.info("Overview: The Post-Fire Watershed Risk Portal (PF-WRP) is a decision support system designed to rapidly assess debris flow and sediment loading risks following wildfire events.")

        st.markdown("""
        #### Step 1: Select the Incident
        Navigate to the **Incident Briefing** module using the sidebar to select a specific fire perimeter from the master dataset. Ensure the fire perimeter and ignition dates align with current incident records.

        #### Step 2: Interrogate Spatial Drivers
        Navigate to the **Spatial Modeling Lab**. Use the map controls and layer visibility toggles to isolate specific geomorphic risk factors (e.g., Burn Severity, Critical Slope, Initiation Hollows). This step visualizes the composite hazard score and identifies spatial initiation zones prior to calculating downstream watershed impacts.

        #### Step 3: Simulate Predictive Rainfall
        Navigate to **Phase 3: Watershed Loading**. Utilize the Operational Weather Inputs sidebar to input the anticipated **Peak 15-minute Rainfall Intensity (mm/hr)** for upcoming storm systems. 
        """)
        
        st.warning("Note: CAL FIRE baseline evaluations typically begin at 24mm/hr. Higher values should be used to simulate intense atmospheric river events.")

        st.markdown("""
        #### Step 4: Export Operational Data
        Once the Earth Engine completes the HUC-12 basin calculations, utilize the export buttons to download:
        * **Executive Report (CSV):** For quantitative review and rapid triage by the WERT team.
        * **Operational Polygons (GeoJSON):** For direct integration into local offline GIS systems or evacuation routing software.
        """)

    with tab2:
        st.markdown("### Understanding the Hazard Parameters")
        st.markdown("To predict a debris flow, the system maps the landscape to find where four specific conditions overlap. A basin is considered high-risk when these factors combine during a severe rainstorm.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **1. Topographic Velocity (Critical Slope)**
            * **The Threshold:** Slopes greater than or equal to 23 Degrees.
            * **Why it matters:** Debris flows require steep terrain to gain momentum. If a hillside is too flat, water just creates puddles or slow-moving mud. At 23 degrees, gravity becomes stronger than the friction holding the soil to the mountain, allowing massive walls of mud to rapidly accelerate downhill.
            * **Scientific Source:** [Staley, D. M., et al. (2017)](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1002/2017GL074243).
            
            **2. Topographic Concavity (Initiation Points)**
            * **The Threshold:** Local elevation less than -3m relative to the surrounding area.
            * **Why it matters:** Water sheds off the top of ridges and gathers in valleys, hollows, and ravines (concave features). Debris flows rarely start on flat cliff faces; they initiate in these hollows where surface water naturally funnels together and concentrates its erosive energy.
            * **Scientific Source:** [Rengers, F. K., et al. (2018)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018JF004837).
            """)
            
        with col2:
            st.markdown("""
            **3. Burn Severity (dNBR)**
            * **The Threshold:** dNBR > 0.15 (Moderate to High Severity).
            * **Why it matters:** When a fire burns incredibly hot, it doesn't just destroy the tree roots holding the dirt together. It actually vaporizes organic matter in the soil, which cools and forms a waxy, water-repellent (hydrophobic) crust on the ground. When rain hits this baked crust, it cannot soak in. Almost 100 percent of the water rushes downhill instantly.
            * **Scientific Source:** [Key, C. H., & Benson, N. C. (2006)](https://research.fs.usda.gov/treesearch/24042).

            **4. Soil Erodibility**
            * **The Threshold:** Measured by the percentage of sand in the soil profile.
            * **Why it matters:** Not all dirt is created equal. Clay soils are sticky and cohesive, meaning they hold together well even when wet. Sandy, loose soils (high sand mass fraction) have no binding agents and are easily detached and swept away by fast-moving water.
            * **Scientific Source:** [Hengl, T., et al. (2023)](https://essd.copernicus.org/articles/18/989/2026/).
            """)

        st.markdown("---")
        st.markdown("### The Sediment Math Engine")
        st.success("Identifying the danger zones is only the first step. The system then uses historical data to predict exactly how much mud and rock will come out of those mountains.")

        st.markdown("""
        **The Gartner Model**
        After evaluating hundreds of real-world debris flows in Southern California, USGS scientists discovered a pattern. They created a formula that looks at the steepness of a basin, how badly it burned, and the intensity of a rainstorm to predict the total volume of the resulting debris flow. This system automates that exact formula.
        
        **The Calculation Variables:**
        * **V (Volume):** The final estimated size of the debris flow in cubic meters.
        * **B23 (Basin Area):** The total area of the watershed that is steeper than 23 degrees.
        * **HM (High/Moderate Burn):** The total area of the watershed that suffered severe fire damage.
        * **R15 (Rainfall):** The user-defined peak 15-minute rainfall intensity (e.g., 24 mm/hr).

        **Source Documentation:** [USGS Empirical Logistic Regression - Gartner et al., 2014](https://pubs.usgs.gov/publication/70188574).
        """)
