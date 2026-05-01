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
st.set_page_config(
    page_title="PF-WRP | Post-Fire Watershed Risk Portal",
    layout="wide"
)

st.sidebar.title("PF-WRP Navigation")
page = st.sidebar.radio("Select Module:", [
    "1. Fire Overview",
    "2. Basin Risk Prediction",
    "3. Model Validation",
])

# ==========================================
# 2. GARTNER (2014) MATH ENGINE
# ==========================================
def calculate_gartner_volume(hm_m2, relief_m, r15_mmhr):
    """
    Gartner et al. (2014) Eq. 3:
    ln(V) = 4.22 + 0.39·√(i15) + 0.36·ln(Bmh) + 0.13·√(R)

    Args:
        hm_m2     : Area burned at moderate-to-high severity in m²
        relief_m  : Watershed relief (max − min elevation) in meters
        r15_mmhr  : Peak 15-min rainfall intensity in mm/hr

    Returns:
        Predicted debris flow volume in m³ (float). Returns 0.0 on bad inputs.
    """
    hm_km2 = (hm_m2 / 1_000_000) if hm_m2 else 0.0
    relief = float(relief_m) if relief_m else 0.0
    i15    = float(r15_mmhr)
    if hm_km2 <= 0.001 or i15 <= 0 or relief <= 0:
        return 0.0
    try:
        ln_v = (
            4.22
            + (0.39 * math.sqrt(i15))
            + (0.36 * math.log(hm_km2))
            + (0.13 * math.sqrt(relief))
        )
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
    zip_path    = 'Master_Fire_Dataset.geojson.zip'
    extract_dir = 'temp_fire_data_v4'
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_members = [
                m for m in zip_ref.namelist()
                if not m.startswith('__MACOSX') and m.endswith('.geojson')
            ]
            zip_basenames = {os.path.basename(m) for m in zip_members}

            if os.path.exists(extract_dir):
                disk_basenames = {
                    f for f in os.listdir(extract_dir) if f.endswith('.geojson')
                }
            else:
                os.makedirs(extract_dir)
                disk_basenames = set()

            missing = zip_basenames - disk_basenames
            for member in zip_members:
                if os.path.basename(member) in missing:
                    zip_ref.extract(member, extract_dir)

        gdfs = []
        for file in os.listdir(extract_dir):
            if file.endswith('.geojson'):
                geojson_path = os.path.join(extract_dir, file)
                fires        = gpd.read_file(geojson_path)
                possible_names = [
                    'incident_n', 'FIRE_NAME', 'Fire_Name', 'Name', 'name', 'mission'
                ]
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
    name_col   = 'incident_n' if 'incident_n' in cal_fires.columns else cal_fires.columns[0]
    fire_series = cal_fires[name_col]
    if 'mission' in cal_fires.columns:
        fire_series = fire_series.fillna(cal_fires['mission'])

    raw_fire_list = sorted(fire_series.dropna().astype(str).unique())

    # Canonical map: display label (title-case) → all-caps GeoJSON FIRE_NAME key.
    _VALIDATED_MAP = {
        'Thomas':     'THOMAS',
        'Station':    'STATION',
        'Grand Prix': 'GRAND PRIX',
        'Old':        'OLD',
    }
    raw_upper      = [f.upper() for f in raw_fire_list]
    clean_fire_list = [
        label for label, key in _VALIDATED_MAP.items()
        if key in raw_upper
    ]

    selected_fire_label = st.sidebar.selectbox(
        "Select Wildfire Perimeter", clean_fire_list, index=0
    )
    st.sidebar.caption(
        "Analysis restricted to four USGS-validated California fires. "
        "Gartner (2014) calibration domain confirmed."
    )

    # selected_fire is the ALL-CAPS key used everywhere downstream.
    selected_fire = _VALIDATED_MAP[selected_fire_label]

    # Filter using uppercase on both sides -- immune to case variation in the GeoJSON.
    fire_data = cal_fires[cal_fires[name_col].str.upper() == selected_fire]

    if fire_data.empty:
        st.error(
            f"No perimeter found for '{selected_fire}' in the loaded GeoJSON. "
            "Verify that Master_Fire_Dataset.geojson.zip contains the individual "
            "fire files (Thomas.geojson, Station.geojson, GrandPrix.geojson, "
            "Old.geojson) and that each has a FIRE_NAME column with an all-caps value."
        )
        st.stop()

    KNOWN_IGNITION_DATES = {
        "THOMAS":     datetime(2017, 12, 4),
        "STATION":    datetime(2009, 8, 26),
        "GRAND PRIX": datetime(2003, 10, 21),
        "OLD":        datetime(2003, 10, 21),
    }
    ignition_date = KNOWN_IGNITION_DATES.get(selected_fire, datetime(2021, 1, 1))
    for col in ['START_DATE', 'ALARM_DATE', 'alarm_date', 'cont_date']:
        if col in fire_data.columns and not pd.isna(fire_data[col].iloc[0]):
            try:
                parsed = pd.to_datetime(fire_data[col].iloc[0]).to_pydatetime()
                if parsed.year < 2020:
                    ignition_date = parsed
                break
            except Exception:
                continue

    pre_fire_start  = (ignition_date - timedelta(days=365)).strftime('%Y-%m-%d')
    pre_fire_end    = (ignition_date - timedelta(days=1)).strftime('%Y-%m-%d')
    post_fire_start = (ignition_date + timedelta(days=1)).strftime('%Y-%m-%d')
    post_fire_end   = (ignition_date + timedelta(days=90)).strftime('%Y-%m-%d')

    area            = ee.FeatureCollection(fire_data.__geo_interface__)
    simplified_area = area.geometry().simplify(maxError=100)
    centroid        = fire_data.to_crs(epsg=3310).geometry.centroid.to_crs(epsg=4326).iloc[0]

    # Persist to session state -- validation module reads these without visiting Module 2.
    st.session_state["selected_fire"]        = selected_fire
    st.session_state["simplified_area_json"] = json.dumps(simplified_area.getInfo())
    st.session_state["pre_fire_start"]       = pre_fire_start
    st.session_state["pre_fire_end"]         = pre_fire_end
    st.session_state["post_fire_start"]      = post_fire_start
    st.session_state["post_fire_end"]        = post_fire_end

else:
    st.error("No fire perimeters loaded.")
    st.stop()


# ==========================================
# 6. SENTINEL-2 SAFE LOADER
# ==========================================
def get_safe_s2(start, end, geom):
    """
    Return a cloud-masked Sentinel-2 SR median composite clipped to geom.
    Falls back to a near-zero dummy image when no scenes are available in the
    date window (e.g. pre-2017 fires before S2 launch).
    """
    col   = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geom)
        .filterDate(start, end)
    )
    dummy = ee.Image.constant([0.0001, 0.0001]).rename(['B8', 'B12'])

    def process_s2():
        return (
            col.map(lambda img: img.updateMask(
                img.select('QA60').bitwiseAnd(1 << 10).eq(0)
                .And(img.select('QA60').bitwiseAnd(1 << 11).eq(0))
            ).divide(10000))
            .select(['B8', 'B12'])
            .median()
        )

    return ee.Image(
        ee.Algorithms.If(col.size().gt(0), process_s2(), dummy)
    ).clip(geom)


# ==========================================
# SCIENTIFIC THRESHOLDS
# ==========================================
SLOPE_LIMIT    = 23     # degrees -- critical slope per Staley et al. (2017)
DNBR_THRESHOLD = 0.15   # moderate/high severity cutoff per Key & Benson (2006)
SOIL_SAND_MIN  = 40     # sand mass fraction % per OpenLandMap


# ==========================================
# MODULE 1: FIRE OVERVIEW
# Two tabs: (A) Incident Briefing + perimeter map,
#           (B) Spatial Modeling Lab -- hazard layer map
# Documentation collapsed into an expander at the bottom.
# ==========================================
if page == "1. Fire Overview":
    st.title(f"Fire Overview -- {selected_fire}")
    st.markdown("---")

    tab_briefing, tab_spatial = st.tabs([
        "Incident Briefing",
        "Spatial Hazard Map",
    ])

    # ------------------------------------------------------------------
    # TAB A -- INCIDENT BRIEFING
    # ------------------------------------------------------------------
    with tab_briefing:
        total_ac = fire_data.to_crs(epsg=3310).area.sum() * 0.000247105

        kpi1, kpi2 = st.columns(2)
        kpi1.metric("Total Acres Burned",      f"{total_ac:,.0f} ac")
        kpi2.metric("Estimated Ignition Date",  ignition_date.strftime('%B %d, %Y'))

        m_brief = folium.Map(
            location=[centroid.y, centroid.x],
            zoom_start=11,
            tiles="CartoDB positron"
        )
        folium.GeoJson(
            fire_data.geometry,
            style_function=lambda x: {
                'fillColor': 'red',
                'color':     'darkred',
                'weight':    2,
                'fillOpacity': 0.4,
            }
        ).add_to(m_brief)
        st_folium(m_brief, use_container_width=True, height=500, key="briefing_map")

    # ------------------------------------------------------------------
    # TAB B -- SPATIAL MODELING LAB
    # ------------------------------------------------------------------
    with tab_spatial:
        st.sidebar.markdown("---")
        st.sidebar.info(
            f"**Critical Slope:** ≥ {SLOPE_LIMIT}°  (Staley et al., 2017)\n\n"
            f"**Severity (dNBR):** > {DNBR_THRESHOLD}  (Key & Benson, 2006)\n\n"
            f"**Soil Sand Fraction:** ≥ {SOIL_SAND_MIN}%  (OpenLandMap)"
        )
        st.sidebar.markdown("### Map Controls")
        basemap_choice = st.sidebar.radio(
            "Reference Basemap:", ["Satellite", "Terrain", "Minimal"]
        )
        st.sidebar.markdown("### Layer Visibility")
        show_risk      = st.sidebar.checkbox("Composite Hazard Score",          value=True)
        show_slope     = st.sidebar.checkbox("Topographic Velocity (Slope)",    value=False)
        show_concavity = st.sidebar.checkbox("Initiation Points (Hollows)",     value=False)
        show_severity  = st.sidebar.checkbox("Burn Severity (dNBR)",            value=False)
        show_soils     = st.sidebar.checkbox("Soil Erodibility (Sand %)",       value=False)
        show_streams   = st.sidebar.checkbox("HydroSHEDS Stream Routing",       value=True)
        show_roads     = st.sidebar.checkbox("TIGER Roads",                     value=True)

        with st.spinner("Compiling Spatial Intersection Data..."):
            dem            = ee.Image("USGS/SRTMGL1_003")
            slope          = ee.Terrain.slope(dem).clip(simplified_area)
            slope_mask     = slope.gte(SLOPE_LIMIT)
            local_mean     = dem.focal_mean(radius=50, units='meters').clip(simplified_area)
            concavity_mask = dem.subtract(local_mean).lt(-3)
        # Use Landsat 5 for pre-2015 fires (before Sentinel-2 launch)
        fire_year = int(post_fire_start[:4])
        if fire_year >= 2015:
            s2_pre  = get_safe_s2(pre_fire_start,  pre_fire_end,  simplified_area)
            s2_post = get_safe_s2(post_fire_start, post_fire_end, simplified_area)
            dnbr_pre_band,  dnbr_post_band  = 'B8', 'B12'
        else:
            def _l5_nbr(start, end, geom):
                col = (ee.ImageCollection('LANDSAT/LT05/C02/T1_TOA')
                       .filterBounds(geom).filterDate(start, end)
                       .sort('CLOUD_COVER'))
                dummy = ee.Image.constant([0.0001, 0.0001]).rename(['B8','B12'])
                def _proc():
                    img = col.first().select(['B4','B7'])
                    # Rename to B8/B12 so downstream normalizedDifference works
                    return img.rename(['B8','B12'])
                return ee.Image(
                    ee.Algorithms.If(col.size().gt(0), _proc(), dummy)
                ).clip(geom)
            s2_pre  = _l5_nbr(pre_fire_start,  pre_fire_end,  simplified_area)
            s2_post = _l5_nbr(post_fire_start, post_fire_end, simplified_area)
            dnbr           = (
                s2_pre.normalizedDifference(['B8', 'B12'])
                .subtract(s2_post.normalizedDifference(['B8', 'B12']))
            )
            severity_mask  = dnbr.gte(DNBR_THRESHOLD)
            erodible_soils = (
                ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02")
                .select('b0')
                .clip(simplified_area)
            )
            soil_risk_mask = erodible_soils.gte(SOIL_SAND_MIN)

            slope_safe          = ee.Image(slope_mask).unmask(0).select(0).toInt()
            sev_safe            = ee.Image(severity_mask).unmask(0).select(0).toInt()
            soil_safe           = ee.Image(soil_risk_mask).unmask(0).select(0).toInt()
            risk_score          = slope_safe.add(sev_safe).add(soil_safe)
            hazard_intersection = risk_score.gte(2).selfMask()

            roads_img   = ee.Image(0).mask(0).paint(
                ee.FeatureCollection("TIGER/2016/Roads").filterBounds(simplified_area), 1, 2
            )
            streams_img = ee.Image(0).mask(0).paint(
                ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers")
                .filterBounds(simplified_area), 1, 1
            )

            if basemap_choice == "Satellite":
                m_lab = folium.Map(
                    location=[centroid.y, centroid.x], zoom_start=12,
                    tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                    attr='Google Hybrid'
                )
                perimeter_color = 'white'
            elif basemap_choice == "Terrain":
                m_lab = folium.Map(
                    location=[centroid.y, centroid.x], zoom_start=12,
                    tiles='https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
                    attr='Google Terrain'
                )
                perimeter_color = 'black'
            else:
                m_lab = folium.Map(
                    location=[centroid.y, centroid.x], zoom_start=12,
                    tiles='CartoDB positron'
                )
                perimeter_color = 'black'

            folium.GeoJson(
                fire_data.geometry,
                style_function=lambda x: {
                    'fillColor': 'transparent',
                    'color':     perimeter_color,
                    'weight':    2.5,
                    'dashArray': '5, 5',
                }
            ).add_to(m_lab)

            C_RISK      = '#FF5733'
            C_SLOPE     = 'yellow'
            C_CONCAVITY = '#8e44ad'
            C_SEVERITY  = 'red'
            C_STREAMS   = '#3498db'
            C_ROADS     = '#2ecc71'

            if show_slope:
                folium.TileLayer(
                    tiles=slope_mask.selfMask().getMapId({'palette': [C_SLOPE]})['tile_fetcher'].url_format,
                    attr='USGS', name='Slope', opacity=0.4
                ).add_to(m_lab)
            if show_concavity:
                folium.TileLayer(
                    tiles=concavity_mask.selfMask().getMapId({'palette': [C_CONCAVITY]})['tile_fetcher'].url_format,
                    attr='USGS', name='Concavity', opacity=0.6
                ).add_to(m_lab)
            if show_severity:
                folium.TileLayer(
                    tiles=severity_mask.selfMask().getMapId({'palette': [C_SEVERITY]})['tile_fetcher'].url_format,
                    attr='ESA', name='Severity', opacity=0.25
                ).add_to(m_lab)
            if show_soils:
                folium.TileLayer(
                    tiles=erodible_soils.getMapId(
                        {'min': 10, 'max': 80, 'palette': ['#f4a460', '#d2691e', '#8b4513']}
                    )['tile_fetcher'].url_format,
                    attr='Soil', name='Soils', opacity=0.7
                ).add_to(m_lab)
            if show_streams:
                folium.TileLayer(
                    tiles=streams_img.getMapId({'palette': [C_STREAMS]})['tile_fetcher'].url_format,
                    attr='WWF', name='Streams'
                ).add_to(m_lab)
            if show_roads:
                folium.TileLayer(
                    tiles=roads_img.getMapId({'palette': [C_ROADS]})['tile_fetcher'].url_format,
                    attr='TIGER', name='Roads'
                ).add_to(m_lab)
            if show_risk:
                folium.TileLayer(
                    tiles=hazard_intersection.getMapId(
                        {'min': 1, 'max': 1, 'palette': [C_RISK]}
                    )['tile_fetcher'].url_format,
                    attr='GEE', name='Risk', opacity=0.8
                ).add_to(m_lab)

            toggle_key = (
                f"lab_{selected_fire}_v{show_risk}{show_slope}{show_concavity}"
                f"{show_severity}{show_soils}{show_streams}{show_roads}_{basemap_choice}"
            )
            st_folium(m_lab, use_container_width=True, height=700, key=toggle_key)

    # ------------------------------------------------------------------
    # DOCUMENTATION EXPANDER -- at the bottom of Module 1
    # ------------------------------------------------------------------
    st.markdown("---")
    with st.expander("Scientific Methodology & User Guide", expanded=False):
        doc_tab1, doc_tab2 = st.tabs(["Operational User Guide", "Scientific Methodology"])

        with doc_tab1:
            st.markdown("### Incident Command Workflow")
            st.info(
                "The Post-Fire Watershed Risk Portal (PF-WRP) is a decision support system "
                "designed to rapidly assess debris flow and sediment loading risks following "
                "wildfire events."
            )
            st.markdown("""
            #### Step 1: Select the Incident
            Choose a fire from the sidebar dropdown. Verify total acreage and ignition date
            against official CAL FIRE / InciWeb records using the **Incident Briefing** tab above.

            #### Step 2: Interrogate Spatial Drivers
            Switch to the **Spatial Hazard Map** tab. Toggle individual hazard layers to
            understand which geomorphic factors make the landscape vulnerable.

            #### Step 3: Simulate Predictive Rainfall
            Navigate to **Basin Risk Prediction** (Module 2). Set the Peak 15-min Rainfall
            Intensity slider. CAL FIRE baseline = 24 mm/hr.

            #### Step 4: Validate the Model
            Navigate to **Model Validation** (Module 3) to compare predictions against
            USGS field measurements and review residual maps.

            #### Step 5: Export Operational Data
            - **Executive Report (CSV)** -- for Incident Action Plans and WERT reports
            - **Operational Polygons (GeoJSON)** -- drag into ArcGIS Pro, QGIS, or ATAK
            """)
            st.warning(
                "Processing may take up to 45 seconds for mega-fires exceeding 100,000 acres."
            )

        with doc_tab2:
            st.markdown("### Geomorphic Hazard Parameters")
            col_doc1, col_doc2 = st.columns(2)
            with col_doc1:
                st.markdown(f"""
                **1. Topographic Velocity (Critical Slope)**
                Threshold: ≥ **{SLOPE_LIMIT}°** | Dataset: USGS SRTM DEM

                At 23 degrees, gravitational forces overcome internal soil friction.
                This threshold is the calibration input for the Gartner (2014) regression.
                *Source: Staley et al. (2017)*

                **2. Topographic Concavity (Initiation Points)**
                Threshold: Local elevation < -3 m vs 50 m focal mean

                Zero-order hollows concentrate surface runoff where debris flows initiate.
                *Source: Rengers et al. (2018)*
                """)
            with col_doc2:
                st.markdown(f"""
                **3. Burn Severity (dNBR)**
                Threshold: dNBR > **{DNBR_THRESHOLD}** | Dataset: ESA Sentinel-2

                High-severity burns create a hydrophobic crust -- nearly 100% of
                rainfall becomes surface runoff instantly.
                *Source: Key & Benson (2006)*

                **4. Soil Erodibility (Sand Mass Fraction)**
                Threshold: ≥ **{SOIL_SAND_MIN}%** | Dataset: OpenLandMap

                Sandy soils lack clay binding agents, dramatically increasing entrainment rates.
                *Source: Hengl et al. (2023)*
                """)
            st.markdown("---")
            st.markdown("### The Sediment Math Engine -- Gartner et al. (2014) Eq. 3")
            st.latex(r"\ln(V) = 4.22 + 0.39\sqrt{i_{15}} + 0.36\ln(B_{mh}) + 0.13\sqrt{R}")
            st.markdown("""
            | Variable | Definition | Units |
            |----------|-----------|-------|
            | V | Predicted debris flow volume | m³ |
            | i₁₅ | Peak 15-min rainfall intensity | mm/hr |
            | B_mh | Watershed area burned at moderate-to-high severity | km² |
            | R | Watershed relief (max − min elevation) | m |

            Coefficients confirmed from Gartner, J.E., Cannon, S.H., & Santi, P.M. (2014),
            *Engineering Geology*, 176, 45–56, Eq. 3, page 9. A prior implementation had
            the three terms in the wrong positions; corrected 2026-04-15.

            **Validation across four California fires (Crowder et al., 2025):**
            Spearman ρ ranges from 0.828 (Old Fire, 17 basins) to 1.000 (Grand Prix, 7 basins).
            All four fires achieve 100% order-of-magnitude accuracy, confirming physically
            plausible volume estimates throughout the Gartner calibration domain.
            """)


# ==========================================
# MODULE 2: BASIN RISK PREDICTION
# Formerly "3. Predictive Debris Flow Modeling" -- identical logic, renamed.
# ==========================================
elif page == "2. Basin Risk Prediction":
    st.title("Basin Risk Prediction")

    st.sidebar.markdown("### Operational Weather Inputs")
    design_storm_mmhr = st.sidebar.slider(
        "Design Storm (Peak 15-min Rainfall Intensity in mm/hr)",
        min_value=10.0, max_value=120.0, value=50.0, step=2.0,
        help="CAL FIRE evaluates baselines at 24 mm/hr. Recorded Thomas Fire peak: 91 mm/hr."
    )

    with st.spinner(f"Running Gartner (2014) regression at {design_storm_mmhr} mm/hr..."):
        dem           = ee.Image("USGS/SRTMGL1_003")
        slope_mask    = ee.Terrain.slope(dem).clip(simplified_area).gte(SLOPE_LIMIT)
        s2_pre        = get_safe_s2(pre_fire_start,  pre_fire_end,  simplified_area)
        s2_post       = get_safe_s2(post_fire_start, post_fire_end, simplified_area)
        dnbr          = (
            s2_pre.normalizedDifference(['B8', 'B12'])
            .subtract(s2_post.normalizedDifference(['B8', 'B12']))
        )
        severity_mask = dnbr.gte(DNBR_THRESHOLD)

        b23_area_img = (
            ee.Image(slope_mask).unmask(0).select(0)
            .multiply(ee.Image.pixelArea())
            .rename('b23_m2')
        )
        hm_area_img = (
            ee.Image(severity_mask).unmask(0).select(0)
            .multiply(ee.Image.pixelArea())
            .rename('hm_m2')
        )
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

        relief_data = (
            dem.clip(simplified_area)
            .reduceRegions(
                collection=huc12,
                reducer=ee.Reducer.minMax(),
                scale=250,
                tileScale=16
            )
            .getInfo()
        )
        relief_lookup = {}
        for rf in relief_data.get('features', []):
            hid      = rf['properties'].get('huc12', '')
            props    = rf['properties']
            elev_max = float(props.get('elevation_max') or props.get('max') or 0)
            elev_min = float(props.get('elevation_min') or props.get('min') or 0)
            relief_lookup[hid] = max(0.0, elev_max - elev_min)

        basin_results = []
        for feature in clean_features:
            props    = feature['properties']
            name     = props.get('name', 'Unknown Basin')
            huc12_id = props.get('huc12', 'Unknown ID')
            b23_m2   = float(props.get('b23_m2') or 0.0)
            hm_m2    = float(props.get('hm_m2')  or 0.0)
            relief_m = relief_lookup.get(huc12_id, 0.0)
            vol      = calculate_gartner_volume(hm_m2, relief_m, design_storm_mmhr)
            basin_results.append({
                'HUC12_ID':                    huc12_id,
                'Basin Name':                  name,
                'Critical Slope Area (Acres)': b23_m2 * 0.000247105,
                'Severe Burn Area (Acres)':    hm_m2  * 0.000247105,
                'Simulated Storm (mm/hr)':     design_storm_mmhr,
                'Sediment Yield (m³)':         vol,
            })

        df_results = pd.DataFrame(basin_results).sort_values(
            by='Sediment Yield (m³)', ascending=False
        )
        st.session_state["hindcast_fire"]    = selected_fire
        st.session_state["hindcast_results"] = df_results

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Watershed Matrix")
            st.dataframe(
                df_results[
                    ['Basin Name', 'Sediment Yield (m³)', 'Critical Slope Area (Acres)']
                ].style.format(
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
            st.download_button(
                label="Download Executive Report (CSV)",
                data=df_results.to_csv(index=False).encode('utf-8'),
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
                m3 = folium.Map(
                    location=[centroid.y, centroid.x], zoom_start=11, tiles='CartoDB positron'
                )
                folium.GeoJson(
                    fire_data.geometry,
                    style_function=lambda x: {
                        'fillColor': 'transparent', 'color': 'black',
                        'weight': 2, 'dashArray': '5, 5',
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
                    legend_name='Estimated Sediment Yield (Cubic Meters)',
                ).add_to(m3)
                streams = (
                    ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers")
                    .filterBounds(simplified_area)
                )

                def style_streams(f):
                    discharge  = ee.Number(f.get('DIS_AV_CMS')).add(1)
                    log_dis    = discharge.log10()
                    line_width = log_dis.multiply(1.5).add(0.5)
                    return f.set('acc_width', line_width).set('acc_color', log_dis)

                stream_img = ee.Image(0).mask(0).paint(
                    streams.map(style_streams), 'acc_color', 'acc_width'
                )
                stream_vis = stream_img.getMapId(
                    {'min': 0, 'max': 2.5, 'palette': ['#00b4d8', '#0077b6', '#03045e']}
                )
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
                        localize=True,
                    )
                ).add_to(m3)
                st_folium(m3, use_container_width=True, height=600, key=f"huc12_{selected_fire}")
            else:
                st.warning("No valid basin geometries found to render.")


# ==========================================
# MODULE 3: MODEL VALIDATION
# Delegates entirely to validation_page.py.
# Tab order inside that file is: Residual Maps → Fire-specific → Academic → Inventory.
# ==========================================
elif page == "3. Model Validation":
    try:
        from validation_page import page_validation as render_validation_page
        render_validation_page()
    except ImportError:
        st.title("Model Validation")
        st.warning(
            "validation_page.py not found in project root. "
            "Place the file alongside app.py and redeploy."
        )
        st.markdown("""
        **What this module will show once connected:**
        - Residual maps -- predicted vs. USGS observed, spatially located
        - Fire-specific basin risk table with USGS comparison
        - Model-wide scatter plot (227 USGS field measurements)
        - R², Spearman ρ, RMSE, factor-of-2 accuracy metrics

        **Data source:** Crowder et al. (2025) -- doi:10.5066/P13EZSWW
        """)
