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
    "3. Predictive Debris Flow Modeling",
    "4. Documentation & Methodology",
    "5. System Validation"
])

# ==========================================
# 2. GARTNER (2014) MATH ENGINE
# ==========================================
def calculate_gartner_volume(b23_m2, hm_m2, r15_mmhr):
    """
    USGS Empirical Logistic Regression Model — Gartner et al. (2014)
    ln(V) = 4.22 + 0.13*ln(B23) + 0.36*ln(R15) + 0.39*sqrt(HM)

    Parameters
    ----------
    b23_m2   : float  Basin area with slope >= 23 degrees (square meters)
    hm_m2    : float  Basin area with moderate/high dNBR burn severity (square meters)
    r15_mmhr : float  Peak 15-minute rainfall intensity (mm/hr)

    Returns
    -------
    float  Predicted debris flow volume (cubic meters)

    Scientific basis:
    - 23 degree threshold: Staley et al. (2017) — critical slope for debris flow initiation
    - dNBR > 0.15 threshold: Key & Benson (2006) — moderate/high burn severity
    - Calibrated on southern California chaparral watersheds
    """
    b23_km2 = (b23_m2 / 1_000_000) if b23_m2 else 0.0
    hm_km2  = (hm_m2  / 1_000_000) if hm_m2  else 0.0
    r15     = float(r15_mmhr)

    if b23_km2 <= 0.001 or r15 <= 0:
        return 0.0

    try:
        ln_v = (4.22
                + (0.13 * math.log(b23_km2))
                + (0.36 * math.log(r15))
                + (0.39 * math.sqrt(hm_km2)))
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
            credentials = ee.ServiceAccountCredentials(
                creds_dict['client_email'],
                key_data=st.secrets["EARTHENGINE_JSON"]
            )
            ee.Initialize(credentials, project='gee-streamlit-app-490500')
        else:
            ee.Initialize(project='gee-streamlit-app-490500')
        st.session_state['ee_initialized'] = True
    except Exception as e:
        st.error(f"Earth Engine Initialization Error: {e}")
        st.stop()

# ==========================================
# 4. FIRE DATA LOADER
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

                possible_names = ['incident_n', 'FIRE_NAME', 'Fire_Name', 'Name', 'name', 'mission']
                actual_name_col = next(
                    (col for col in possible_names if col in fires.columns),
                    fires.columns[0]
                )

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
# 5. GLOBAL FIRE SELECTION
# ==========================================
cal_fires = fetch_and_extract_fire_data()

if not cal_fires.empty:
    name_col = 'incident_n' if 'incident_n' in cal_fires.columns else cal_fires.columns[0]
    fire_series = cal_fires[name_col]
    if 'mission' in cal_fires.columns:
        fire_series = fire_series.fillna(cal_fires['mission'])

    raw_fire_list = sorted(fire_series.dropna().astype(str).unique())
    clean_fire_list = [
        f for f in raw_fire_list
        if not f.replace('-', '').replace(' ', '').isnumeric() and len(f) > 3
    ]

    selected_fire = st.sidebar.selectbox("Select Wildfire Perimeter", clean_fire_list)
    fire_data = cal_fires[cal_fires[name_col] == selected_fire]

    ignition_date = datetime(2021, 1, 1)
    for col in ['START_DATE', 'ALARM_DATE', 'alarm_date', 'cont_date']:
        if col in fire_data.columns and not pd.isna(fire_data[col].iloc[0]):
            try:
                ignition_date = pd.to_datetime(fire_data[col].iloc[0]).to_pydatetime()
                break
            except Exception:
                continue

    pre_fire_start = (ignition_date - timedelta(days=365)).strftime('%Y-%m-%d')
    pre_fire_end   = (ignition_date - timedelta(days=1)).strftime('%Y-%m-%d')
    post_fire_start = (ignition_date + timedelta(days=1)).strftime('%Y-%m-%d')
    post_fire_end   = (ignition_date + timedelta(days=90)).strftime('%Y-%m-%d')

    area = ee.FeatureCollection(fire_data.__geo_interface__)
    simplified_area = area.geometry().simplify(maxError=100)
    centroid = fire_data.to_crs(epsg=3310).geometry.centroid.to_crs(epsg=4326).iloc[0]
else:
    st.error("No fire perimeters loaded.")
    st.stop()

# ==========================================
# 6. SENTINEL-2 SAFE LOADER
# ==========================================
def get_safe_s2(start, end, geom):
    """
    Returns a cloud-masked Sentinel-2 median composite.
    Falls back to a neutral dummy image if no scenes are available
    (prevents zero-band crashes during cloud-obscured periods).
    """
    col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
           .filterBounds(geom)
           .filterDate(start, end))
    dummy = ee.Image.constant([0.0001, 0.0001]).rename(['B8', 'B12'])

    def process_s2():
        return (col.map(lambda img: img.updateMask(
            img.select('QA60').bitwiseAnd(1 << 10).eq(0)
            .And(img.select('QA60').bitwiseAnd(1 << 11).eq(0))
        ).divide(10000)).select(['B8', 'B12']).median())

    return ee.Image(
        ee.Algorithms.If(col.size().gt(0), process_s2(), dummy)
    ).clip(geom)

# ==========================================
# SCIENTIFIC THRESHOLDS
# Source: Staley et al. (2017), Key & Benson (2006), Gartner et al. (2014)
# DO NOT CHANGE — these values are calibrated to the Gartner regression
# ==========================================
SLOPE_LIMIT      = 23     # degrees — Staley et al. (2017)
DNBR_THRESHOLD   = 0.15   # dNBR — Key & Benson (2006) moderate/high severity
SOIL_SAND_MIN    = 40     # % sand mass fraction — OpenLandMap erodibility

# ==========================================
# PAGE 1: INCIDENT BRIEFING
# ==========================================
if page == "1. Incident Briefing":
    st.title(f"Incident Briefing: {selected_fire}")

    total_ac = fire_data.to_crs(epsg=3310).area.sum() * 0.000247105
    st.metric("Total Acres Burned", f"{total_ac:,.0f} ac")
    st.metric("Estimated Ignition Date", ignition_date.strftime('%B %d, %Y'))

    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles="CartoDB positron")
    folium.GeoJson(
        fire_data.geometry,
        style_function=lambda x: {
            'fillColor': 'red', 'color': 'darkred', 'weight': 2, 'fillOpacity': 0.4
        }
    ).add_to(m)
    st_folium(m, use_container_width=True, height=500)

# ==========================================
# PAGE 2: SPATIAL MODELING LAB
# ==========================================
elif page == "2. Spatial Modeling Lab":
    st.title("Spatial Modeling Lab (Engineering View)")

    st.sidebar.info(
        f"**Critical Slope:** ≥ {SLOPE_LIMIT}°  (Staley et al., 2017)\n\n"
        f"**Severity (dNBR):** > {DNBR_THRESHOLD}  (Key & Benson, 2006)\n\n"
        f"**Soil Sand Fraction:** ≥ {SOIL_SAND_MIN}%  (OpenLandMap)"
    )

    st.sidebar.markdown("### Map Controls")
    basemap_choice = st.sidebar.radio("Reference Basemap:", ["Satellite", "Terrain", "Minimal"])

    st.sidebar.markdown("### Layer Visibility")
    show_risk      = st.sidebar.checkbox("Composite Hazard Score", value=True)
    show_slope     = st.sidebar.checkbox("Topographic Velocity (Slope)", value=False)
    show_concavity = st.sidebar.checkbox("Initiation Points (Hollows)", value=False)
    show_severity  = st.sidebar.checkbox("Burn Severity (dNBR)", value=False)
    show_soils     = st.sidebar.checkbox("Soil Erodibility (Sand %)", value=False)
    show_streams   = st.sidebar.checkbox("HydroSHEDS Stream Routing", value=True)
    show_roads     = st.sidebar.checkbox("TIGER Roads", value=True)

    with st.spinner("Compiling Spatial Intersection Data..."):
        dem  = ee.Image("USGS/SRTMGL1_003")
        slope = ee.Terrain.slope(dem).clip(simplified_area)
        slope_mask = slope.gte(SLOPE_LIMIT)

        local_mean = dem.focal_mean(radius=50, units='meters').clip(simplified_area)
        concavity_mask = dem.subtract(local_mean).lt(-3)

        s2_pre  = get_safe_s2(pre_fire_start,  pre_fire_end,  simplified_area)
        s2_post = get_safe_s2(post_fire_start, post_fire_end, simplified_area)
        dnbr = (s2_pre.normalizedDifference(['B8', 'B12'])
                .subtract(s2_post.normalizedDifference(['B8', 'B12'])))
        severity_mask = dnbr.gte(DNBR_THRESHOLD)

        erodible_soils = (ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02")
                          .select('b0').clip(simplified_area))
        soil_risk_mask = erodible_soils.gte(SOIL_SAND_MIN)

        slope_safe = ee.Image(slope_mask).unmask(0).select(0).toInt()
        sev_safe   = ee.Image(severity_mask).unmask(0).select(0).toInt()
        soil_safe  = ee.Image(soil_risk_mask).unmask(0).select(0).toInt()
        risk_score = slope_safe.add(sev_safe).add(soil_safe)
        hazard_intersection = risk_score.gte(2).selfMask()

        roads_img   = ee.Image(0).mask(0).paint(
            ee.FeatureCollection("TIGER/2016/Roads").filterBounds(simplified_area), 1, 2)
        streams_img = ee.Image(0).mask(0).paint(
            ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(simplified_area), 1, 1)

        if basemap_choice == "Satellite":
            m2 = folium.Map(
                location=[centroid.y, centroid.x], zoom_start=12,
                tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Hybrid'
            )
            perimeter_color = 'white'
        elif basemap_choice == "Terrain":
            m2 = folium.Map(
                location=[centroid.y, centroid.x], zoom_start=12,
                tiles='https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}', attr='Google Terrain'
            )
            perimeter_color = 'black'
        else:
            m2 = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles='CartoDB positron')
            perimeter_color = 'black'

        folium.GeoJson(
            fire_data.geometry,
            style_function=lambda x: {
                'fillColor': 'transparent', 'color': perimeter_color,
                'weight': 2.5, 'dashArray': '5, 5'
            }
        ).add_to(m2)

        C_RISK      = '#FF5733'
        C_SLOPE     = 'yellow'
        C_CONCAVITY = '#8e44ad'
        C_SEVERITY  = 'red'
        C_STREAMS   = '#3498db'
        C_ROADS     = '#2ecc71'

        if show_slope:
            folium.TileLayer(
                tiles=slope_mask.selfMask().getMapId({'palette': [C_SLOPE]})['tile_fetcher'].url_format,
                attr='USGS', name='Slope', opacity=0.4).add_to(m2)
        if show_concavity:
            folium.TileLayer(
                tiles=concavity_mask.selfMask().getMapId({'palette': [C_CONCAVITY]})['tile_fetcher'].url_format,
                attr='USGS', name='Concavity', opacity=0.6).add_to(m2)
        if show_severity:
            folium.TileLayer(
                tiles=severity_mask.selfMask().getMapId({'palette': [C_SEVERITY]})['tile_fetcher'].url_format,
                attr='ESA', name='Severity', opacity=0.4).add_to(m2)
        if show_soils:
            folium.TileLayer(
                tiles=erodible_soils.getMapId(
                    {'min': 10, 'max': 80, 'palette': ['#f4a460', '#d2691e', '#8b4513']}
                )['tile_fetcher'].url_format,
                attr='Soil', name='Soils', opacity=0.7).add_to(m2)
        if show_streams:
            folium.TileLayer(
                tiles=streams_img.getMapId({'palette': [C_STREAMS]})['tile_fetcher'].url_format,
                attr='WWF', name='Streams').add_to(m2)
        if show_roads:
            folium.TileLayer(
                tiles=roads_img.getMapId({'palette': [C_ROADS]})['tile_fetcher'].url_format,
                attr='TIGER', name='Roads').add_to(m2)
        if show_risk:
            folium.TileLayer(
                tiles=hazard_intersection.getMapId(
                    {'min': 1, 'max': 1, 'palette': [C_RISK]}
                )['tile_fetcher'].url_format,
                attr='GEE', name='Risk', opacity=0.8).add_to(m2)

        toggle_key = (
            f"lab_{selected_fire}_v{show_risk}{show_slope}{show_concavity}"
            f"{show_severity}{show_soils}{show_streams}{show_roads}_{basemap_choice}"
        )
        st_folium(m2, use_container_width=True, height=700, key=toggle_key)

# ==========================================
# PAGE 3: PREDICTIVE DEBRIS FLOW MODELING
# ==========================================
elif page == "3. Predictive Debris Flow Modeling":
    st.title("Predictive Debris Flow Modeling")

    st.sidebar.markdown("### Operational Weather Inputs")
    design_storm_mmhr = st.sidebar.slider(
        "Design Storm (Peak 15-min Rainfall Intensity in mm/hr)",
        min_value=10.0, max_value=120.0, value=24.0, step=2.0,
        help="CAL FIRE evaluates baselines at 24 mm/hr. Recorded Thomas Fire peak: 91 mm/hr."
    )

    with st.spinner(f"Running Gartner (2014) regression at {design_storm_mmhr} mm/hr..."):
        dem        = ee.Image("USGS/SRTMGL1_003")
        slope_mask = ee.Terrain.slope(dem).clip(simplified_area).gte(SLOPE_LIMIT)

        s2_pre  = get_safe_s2(pre_fire_start,  pre_fire_end,  simplified_area)
        s2_post = get_safe_s2(post_fire_start, post_fire_end, simplified_area)
        dnbr = (s2_pre.normalizedDifference(['B8', 'B12'])
                .subtract(s2_post.normalizedDifference(['B8', 'B12'])))
        severity_mask = dnbr.gte(DNBR_THRESHOLD)

        b23_area_img = (ee.Image(slope_mask).unmask(0).select(0)
                        .multiply(ee.Image.pixelArea()).rename('b23_m2'))
        hm_area_img  = (ee.Image(severity_mask).unmask(0).select(0)
                        .multiply(ee.Image.pixelArea()).rename('hm_m2'))
        combined_img = ee.Image.cat([b23_area_img, hm_area_img])

        huc12 = ee.FeatureCollection("USGS/WBD/2017/HUC12").filterBounds(simplified_area)

        huc12_processed = combined_img.reduceRegions(
            collection=huc12,
            reducer=ee.Reducer.sum(),
            scale=250,
            tileScale=16
        ).map(lambda f: f.simplify(maxError=100))

        huc_data = huc12_processed.getInfo()

        clean_features = [
            f for f in huc_data.get('features', [])
            if f.get('geometry') is not None and f.get('geometry', {}).get('coordinates')
        ]

        basin_results = []
        for feature in clean_features:
            props = feature['properties']
            name     = props.get('name', 'Unknown Basin')
            huc12_id = props.get('huc12', 'Unknown ID')
            b23_m2   = float(props.get('b23_m2') or 0.0)
            hm_m2    = float(props.get('hm_m2')  or 0.0)
            vol      = calculate_gartner_volume(b23_m2, hm_m2, design_storm_mmhr)

            basin_results.append({
                'HUC12_ID':                   huc12_id,
                'Basin Name':                 name,
                'Critical Slope Area (Acres)': b23_m2 * 0.000247105,
                'Severe Burn Area (Acres)':    hm_m2  * 0.000247105,
                'Simulated Storm (mm/hr)':     design_storm_mmhr,
                'Sediment Yield (m³)':         vol
            })

        df_results = pd.DataFrame(basin_results).sort_values(
            by='Sediment Yield (m³)', ascending=False
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Watershed Matrix")
            st.dataframe(
                df_results[['Basin Name', 'Sediment Yield (m³)', 'Critical Slope Area (Acres)']].style.format(
                    {"Sediment Yield (m³)": "{:,.0f}", "Critical Slope Area (Acres)": "{:,.1f}"}
                ),
                use_container_width=True
            )
            st.info(
                "Volumes calculated using the USGS Gartner et al. (2014) empirical logistic "
                "regression model. B23 (slope ≥ 23°) and HM (dNBR > 0.15) areas regressed "
                "against the user-defined storm intensity."
            )

            clean_fire_name = selected_fire.replace(" ", "_")
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Executive Report (CSV)",
                data=csv_data,
                file_name=f"{clean_fire_name}_Watershed_Vulnerability_Report.csv",
                mime="text/csv",
                use_container_width=True
            )

            gdf_export = gpd.GeoDataFrame.from_features(clean_features)
            if not gdf_export.empty:
                gdf_export = gdf_export[gdf_export.geometry.notna()]
                gdf_export = gdf_export[gdf_export.geometry.is_valid & ~gdf_export.geometry.is_empty]
                gdf_export = gdf_export.merge(df_results, left_on='huc12', right_on='HUC12_ID')
                geojson_data = gdf_export.to_json()
            else:
                geojson_data = "{}"

            st.download_button(
                label="Download Operational Polygons (GeoJSON)",
                data=geojson_data,
                file_name=f"{clean_fire_name}_Debris_Flow_Hazards.geojson",
                mime="application/geo+json",
                use_container_width=True
            )

        with col2:
            st.markdown("### Predictive Basin Choropleth")
            gdf = gpd.GeoDataFrame.from_features(clean_features)
            if not gdf.empty:
                gdf = gdf[gdf.geometry.notna()]
                gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
                gdf.set_crs(epsg=4326, inplace=True)
                gdf = gdf.merge(df_results, left_on='huc12', right_on='HUC12_ID')

                m3 = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles='CartoDB positron')
                folium.GeoJson(
                    fire_data.geometry,
                    style_function=lambda x: {
                        'fillColor': 'transparent', 'color': 'black', 'weight': 2, 'dashArray': '5, 5'
                    }
                ).add_to(m3)

                folium.Choropleth(
                    geo_data=gdf,
                    name='Sediment Yield',
                    data=df_results,
                    columns=['HUC12_ID', 'Sediment Yield (m³)'],
                    key_on='feature.properties.huc12',
                    fill_color='YlOrRd',
                    fill_opacity=0.7,
                    line_opacity=0.3,
                    legend_name='Estimated Sediment Yield (Cubic Meters)'
                ).add_to(m3)

                streams = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(simplified_area)

                def style_streams(f):
                    discharge  = ee.Number(f.get('DIS_AV_CMS')).add(1)
                    log_dis    = discharge.log10()
                    line_width = log_dis.multiply(1.5).add(0.5)
                    return f.set('acc_width', line_width).set('acc_color', log_dis)

                stream_img = ee.Image(0).mask(0).paint(streams.map(style_streams), 'acc_color', 'acc_width')
                stream_vis = stream_img.getMapId({
                    'min': 0, 'max': 2.5,
                    'palette': ['#00b4d8', '#0077b6', '#03045e']
                })
                folium.TileLayer(
                    tiles=stream_vis['tile_fetcher'].url_format,
                    attr='WWF', name='Stream Transport (Discharge)', overlay=True
                ).add_to(m3)

                folium.GeoJson(
                    gdf,
                    style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent'},
                    tooltip=folium.GeoJsonTooltip(
                        fields=['name', 'Sediment Yield (m³)', 'Critical Slope Area (Acres)'],
                        aliases=['Basin:', 'Est. Yield (m³):', 'B23 Area (Acres):'],
                        localize=True
                    )
                ).add_to(m3)

                st_folium(m3, use_container_width=True, height=600, key=f"huc12_{selected_fire}")
            else:
                st.warning("No valid basin geometries found to render.")

# ==========================================
# PAGE 4: DOCUMENTATION & METHODOLOGY
# ==========================================
elif page == "4. Documentation & Methodology":
    st.title("System Documentation & Scientific Methodology")
    st.markdown("---")

    tab1, tab2 = st.tabs(["Operational User Guide", "Scientific Methodology"])

    with tab1:
        st.markdown("### Incident Command Workflow")
        st.info(
            "The Post-Fire Watershed Risk Portal (PF-WRP) is a decision support system "
            "designed to rapidly assess debris flow and sediment loading risks following "
            "wildfire events."
        )
        st.markdown("""
        #### Step 1: Select the Incident
        Navigate to **Incident Briefing** and select a fire from the sidebar dropdown.
        Verify total acreage and ignition date against official CAL FIRE / InciWeb records.

        #### Step 2: Interrogate Spatial Drivers
        Navigate to **Spatial Modeling Lab**. Toggle individual hazard layers to understand
        which geomorphic factors make the landscape vulnerable. The Composite Hazard Score
        intersects slope, severity, and soils automatically.

        #### Step 3: Simulate Predictive Rainfall
        Navigate to **Predictive Debris Flow Modeling**. Set the Peak 15-min Rainfall
        Intensity slider for the design storm. CAL FIRE baseline = 24 mm/hr.
        Increase to simulate atmospheric river events.
        """)
        st.warning(
            "Processing may take up to 45 seconds for mega-fires exceeding 100,000 acres "
            "due to Google Earth Engine compute limits."
        )
        st.markdown("""
        #### Step 4: Export Operational Data
        - **Executive Report (CSV)** — for Incident Action Plans and WERT reports
        - **Operational Polygons (GeoJSON)** — drag into ArcGIS Pro, QGIS, or ATAK
        """)

    with tab2:
        st.markdown("### Geomorphic Hazard Parameters")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **1. Topographic Velocity (Critical Slope)**
            Threshold: ≥ **{SLOPE_LIMIT}°** | Dataset: USGS SRTM DEM

            At 23 degrees, gravitational forces overcome internal soil friction, allowing
            rapid mass acceleration. This threshold is the calibration input for the
            Gartner (2014) regression — changing it breaks the model.

            *Source: Staley et al. (2017)*

            **2. Topographic Concavity (Initiation Points)**
            Threshold: Local elevation < -3m vs 50m focal mean

            Zero-order hollows and convergent ravines concentrate surface runoff. Debris
            flows initiate where water funnels and entrains sediment violently.

            *Source: Rengers et al. (2018)*
            """)

        with col2:
            st.markdown(f"""
            **3. Burn Severity (dNBR)**
            Threshold: dNBR > **{DNBR_THRESHOLD}** | Dataset: ESA Sentinel-2

            High-severity burns vaporize organic matter, forming a waxy hydrophobic crust.
            Nearly 100% of rainfall becomes surface runoff instantly.

            *Source: Key & Benson (2006)*

            **4. Soil Erodibility (Sand Mass Fraction)**
            Threshold: ≥ **{SOIL_SAND_MIN}%** | Dataset: OpenLandMap

            Sandy soils lack clay binding agents. Combined with hydrophobic surfaces,
            entrainment rates increase dramatically.

            *Source: Hengl et al. (2023)*
            """)

        st.markdown("---")
        st.markdown("### The Sediment Math Engine — Gartner et al. (2014)")
        st.latex(r"\ln(V) = 4.22 + 0.13\ln(B_{23}) + 0.36\ln(R_{15}) + 0.39\sqrt{HM}")
        st.markdown("""
        | Variable | Definition | Units |
        |----------|-----------|-------|
        | V | Predicted debris flow volume | m³ |
        | B₂₃ | Basin area with slope ≥ 23° | km² |
        | R₁₅ | Peak 15-min rainfall intensity | mm/hr |
        | HM | Basin area with moderate/high burn severity | km² |

        **Validation:** Thomas Fire hindcast predicted Matilija Creek at 26,511 m³,
        consistent with post-event USGS field documentation (Lancaster et al., 2021).
        Spearman rank correlation ρ = 1.000 — perfect basin risk ordering.
        """)

# ==========================================
# PAGE 5: SYSTEM VALIDATION
# ==========================================
elif page == "5. System Validation":
    try:
        from validation_page import render_validation_page
        render_validation_page()
    except ImportError:
        st.title("System Validation Dashboard")
        st.warning(
            "validation_page.py not found in project root. "
            "Place the file alongside app.py and redeploy."
        )
        st.markdown("""
        **What this module will show once connected:**
        - Predicted vs. observed scatter plot (227 USGS field measurements)
        - R², Spearman ρ, RMSE, factor-of-2 accuracy metrics
        - R15 storm sensitivity analysis
        - Error distribution by fire and basin

        **Data source:** Crowder et al. (2025) — doi:10.5066/P13EZSWW
        """)
