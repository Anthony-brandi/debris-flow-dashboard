import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import ee
import json
from datetime import datetime, timedelta

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(page_title="Wildfire Debris Flow Analysis", layout="wide", page_icon="⛰️")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Interactive Risk Map", "Technical Documentation"])

# ==========================================
# 2. GEE INITIALIZATION
# ==========================================
if 'ee_initialized' not in st.session_state:
    try:
        try:
            if "EARTHENGINE_JSON" in st.secrets:
                creds_dict = json.loads(st.secrets["EARTHENGINE_JSON"])
                credentials = ee.ServiceAccountCredentials(creds_dict['client_email'], key_data=st.secrets["EARTHENGINE_JSON"])
                ee.Initialize(credentials, project='strange-bird-461405-v7')
            else:
                ee.Initialize(project='strange-bird-461405-v7')
        except FileNotFoundError:
            ee.Initialize(project='strange-bird-461405-v7')
            
        st.session_state['ee_initialized'] = True
    except Exception as e:
        st.error(f"Initialization Error: {e}")

@st.cache_data
def load_and_clean_data():
    path = '/Users/anthonybrandi/Desktop/All Da Folders/QGIS/Senior Project/CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view/CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view.shp'
    fires = gpd.read_file(path)
    fires = fires.dissolve(by='incident_n').reset_index()
    return fires.to_crs(epsg=4326)

# ==========================================
# PAGE 1: INTERACTIVE RISK MAP
# ==========================================
if page == "Interactive Risk Map":
    try:
        cal_fires = load_and_clean_data()
        
        st.sidebar.title("Analysis Control")
        fire_list = sorted(cal_fires['incident_n'].fillna(cal_fires['mission']).dropna().unique())
        selected_fire = st.sidebar.selectbox("Select Wildfire Perimeter", fire_list)
        fire_data = cal_fires[cal_fires['incident_n'] == selected_fire]
        
        # --- DYNAMIC TIME PARSER ---
        default_storm_date = datetime(2021, 12, 14)
        for col in ['CONT_DATE', 'ALARM_DATE', 'alarm_date', 'cont_date', 'START_DATE']:
            if col in fire_data.columns and not pd.isna(fire_data[col].iloc[0]):
                try:
                    parsed_date = pd.to_datetime(fire_data[col].iloc[0])
                    default_storm_date = parsed_date.to_pydatetime() + timedelta(days=90)
                    break
                except Exception:
                    continue

        st.title(f"{selected_fire} Post-Fire Debris Flow Risk Dashboard")
        
        st.sidebar.markdown("---")
        analyze_btn = st.sidebar.checkbox("Run Hazard Analysis", value=False)
        slope_limit = st.sidebar.slider("Slope Threshold (Degrees)", 10, 45, 27)
        
        # --- STORM SIMULATOR ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Hydrological Trigger Simulator")
        
        storm_date = st.sidebar.date_input(f"Simulated Storm Date (+90 days)", value=default_storm_date)
        start_date = (storm_date - timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = (storm_date + timedelta(days=1)).strftime('%Y-%m-%d')

        # --- LAYER CONTROLS ---
        st.sidebar.markdown("---")
        with st.sidebar.expander("⚙️ Map Layer Visibility", expanded=True):
            show_risk = st.checkbox("Critical Risk (Orange)", value=True)
            show_slope = st.checkbox(f"Highlighted Slopes (≥ {slope_limit}°)", value=True)
            show_soil = st.checkbox("Moderate/High Erodibility Soils", value=False)
            show_rain = st.checkbox(f"Storm Precipitation ({storm_date.strftime('%Y-%m-%d')})", value=False)
            show_water = st.checkbox("Watersheds (HUC-12)", value=False)
            show_infra = st.checkbox("Infrastructure (Roads)", value=True)
            
            basemap_opt = st.radio("Basemap Style", ["Google Terrain", "Google Satellite"])
            basemap_url = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}' if basemap_opt == "Google Terrain" else 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'

        # --- MAP INITIALIZATION ---
        centroid = fire_data.to_crs(epsg=3310).geometry.centroid.to_crs(epsg=4326).iloc[0]
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles=basemap_url, attr=basemap_opt)
        folium.GeoJson(fire_data.geometry, style_function=lambda x: {'fillColor': 'transparent', 'color': 'red', 'weight': 3}).add_to(m)

        if analyze_btn:
            # --- DYNAMIC LEGEND ---
            legend_items = ['<i style="background:red; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px; border:1px solid black;"></i> Fire Perimeter']
            if show_risk: legend_items.append('<i style="background:#ff7b00; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px; border:1px solid black;"></i> Critical Risk (Burned + Steep)')
            if show_slope: legend_items.append('<i style="background:yellow; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px; border:1px solid black;"></i> Slope Target Met')
            if show_soil: legend_items.append('<i style="background:#800026; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px; border:1px solid black;"></i> Erodible Soils (Loams/Silts)')
            if show_rain: legend_items.append('<i style="background:blue; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px; border:1px solid black;"></i> Peak Storm Rainfall (10km Res)')
            if show_water: legend_items.append('<i style="background:purple; width:10px; height:2px; float:left; margin-right:5px; margin-top:7px; border:1px solid purple;"></i> Watersheds')
            if show_infra: legend_items.append('<i style="background:#2ecc71; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px; border:1px solid black;"></i> Roads')

            legend_html = f"""
                 <div style="position: fixed; bottom: 50px; left: 50px; width: 310px; background-color: white; color: black; border:2px solid grey; z-index:9999; font-size:13px; padding: 10px; border-radius: 5px; line-height: 1.5;">
                 <b>Active Layers</b><br>{"<br>".join(legend_items)}
                 </div>"""

            with st.spinner(f"Processing Data & Generating Export Links..."):
                area = ee.FeatureCollection(fire_data.__geo_interface__)
                
                # 1. HAZARD MATH
                dem = ee.Image("USGS/SRTMGL1_003")
                slope = ee.Terrain.slope(dem).clip(area)
                slope_mask = slope.gte(slope_limit) 
                
                s2_img = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area).filterDate('2025-06-01', '2026-03-16').median().clip(area)
                nbr = s2_img.normalizedDifference(['B8', 'B12'])
                hazard_mask = slope_mask.And(nbr.lt(0.1))

                # 2. SOIL ERODIBILITY
                raw_soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0').clip(area)
                erodible_soils = raw_soil.lt(11).selfMask()

                # 3. INFRA & WATER
                roads = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(area)
                roads_img = ee.Image(0).mask(0).paint(roads, 1, 2)
                
                watersheds = ee.FeatureCollection("USGS/WBD/2017/HUC12").filterBounds(area)
                water_img = ee.Image(0).mask(0).paint(watersheds, 1, 1)
                
                # 4. PRECIPITATION
                precip = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterDate(start_date, end_date).select('precipitation').max().clip(area)
                precip = precip.updateMask(precip.gt(0.5))

                precip_stat = precip.reduceRegion(reducer=ee.Reducer.max(), geometry=area.geometry(), scale=1000).getInfo().get('precipitation')
                max_rain_hr = precip_stat if precip_stat is not None else 0

                if max_rain_hr >= 12: trigger_status = "CRITICAL 🔴"
                elif max_rain_hr >= 5: trigger_status = "WARNING 🟡"
                elif max_rain_hr > 0.5: trigger_status = "LOW 🟢"
                else: trigger_status = "DORMANT ⚪"

                # --- STATISTICS ---
                total_ac = (fire_data.to_crs(epsg=3310).area.sum()) * 0.000247105
                stats = hazard_mask.multiply(ee.Image.pixelArea()).reduceRegion(reducer=ee.Reducer.sum(), geometry=area.geometry(), scale=30, maxPixels=1e9)
                haz_ac = (stats.getInfo().get('slope', 0)) * 0.000247105
                pct = (haz_ac / total_ac) * 100 if total_ac > 0 else 0

                st.sidebar.markdown("---")
                st.sidebar.subheader("Quantitative Statistics")
                st.sidebar.metric("Critical Risk Area", f"{haz_ac:,.0f} acres")
                st.sidebar.metric(f"Peak Intensity ({storm_date.strftime('%b %d, %Y')})", f"{max_rain_hr:.2f} mm/hr")
                
                # --- NEW: FULL RESOLUTION GOOGLE DRIVE EXPORTER ---
                st.sidebar.markdown("---")
                st.sidebar.subheader("📥 Export GIS Data (Full Res)")
                st.sidebar.caption("Push uncompressed 30m GeoTIFFs directly to your Google Drive to bypass browser limits.")
                
                if st.sidebar.button("🚀 Send Active Layers to Google Drive"):
                    with st.spinner("Dispatching tasks to Google servers..."):
                        try:
                            clean_fire_name = str(selected_fire).replace(" ", "_")
                            if show_risk:
                                task_risk = ee.batch.Export.image.toDrive(
                                    image=hazard_mask.updateMask(hazard_mask),
                                    description=f"{clean_fire_name}_CriticalRisk_30m",
                                    folder="DebrisFlow_Exports",
                                    scale=30,
                                    region=area.geometry(),
                                    maxPixels=1e13
                                )
                                task_risk.start()
                            if show_soil:
                                task_soil = ee.batch.Export.image.toDrive(
                                    image=erodible_soils,
                                    description=f"{clean_fire_name}_ErodibleSoils_250m",
                                    folder="DebrisFlow_Exports",
                                    scale=250,
                                    region=area.geometry(),
                                    maxPixels=1e13
                                )
                                task_soil.start()
                            if show_rain:
                                task_rain = ee.batch.Export.image.toDrive(
                                    image=precip,
                                    description=f"{clean_fire_name}_Precipitation_{storm_date.strftime('%Y%m%d')}",
                                    folder="DebrisFlow_Exports",
                                    scale=1000,
                                    region=area.geometry(),
                                    maxPixels=1e13
                                )
                                task_rain.start()
                            st.sidebar.success("✅ Success! Check the 'DebrisFlow_Exports' folder in your Google Drive in a few minutes.")
                        except Exception as e:
                            st.sidebar.error(f"Export failed: {e}")

                # Cleaned-up Assessment Report
                report_text = f"""WILDFIRE DEBRIS FLOW RISK ASSESSMENT
Fire Perimeter: {selected_fire}
Analysis Date: {datetime.now().strftime('%Y-%m-%d')}

--- STATIC HAZARD STATISTICS ---
Total Perimeter Area: {total_ac:,.0f} acres
Slope Hazard Threshold: >= {slope_limit} degrees
Critical Risk Area (Steep & Burned): {haz_ac:,.0f} acres
Percentage of Area at Risk: {pct:.1f}%

--- ACTIVE TRIGGER INFERENCE (SIMULATION) ---
Storm Date Analyzed: {storm_date.strftime('%Y-%m-%d')}
Peak Rainfall Intensity Recorded: {max_rain_hr:.2f} mm/hr
Hydrological Trigger Inference: {trigger_status}
"""
                st.sidebar.download_button(label="Download Assessment Report", data=report_text, file_name=f"{selected_fire}_Assessment.txt", mime="text/plain")


                # --- RENDER ACTIVE LAYERS ---
                if show_water:
                    w_id = water_img.getMapId({'min': 1, 'max': 1, 'palette': ['purple']})
                    folium.TileLayer(tiles=w_id['tile_fetcher'].url_format, attr='USGS HUC12', name='Watersheds', overlay=True).add_to(m)

                if show_soil:
                    s_id = erodible_soils.getMapId({'palette': ['#800026']}) 
                    folium.TileLayer(tiles=s_id['tile_fetcher'].url_format, attr='OpenLandMap', name='Erodible Soils', overlay=True).add_to(m)
                
                if show_slope:
                    slope_vis = slope_mask.updateMask(slope_mask).getMapId({'palette': ['yellow'], 'opacity': 0.5})
                    folium.TileLayer(tiles=slope_vis['tile_fetcher'].url_format, attr='USGS', name='Highlighted Slopes', overlay=True).add_to(m)

                if show_rain:
                    r_id = precip.getMapId({'min': 0, 'max': 15, 'palette': ['lightblue', 'blue', 'purple', 'black']})
                    folium.TileLayer(tiles=r_id['tile_fetcher'].url_format, attr='NASA GPM', name=f'Peak Storm Intensity', overlay=True, opacity=0.5).add_to(m)

                if show_risk:
                    h_id = hazard_mask.updateMask(hazard_mask).getMapId({'palette': ['#ff7b00'], 'opacity': 0.9})
                    folium.TileLayer(tiles=h_id['tile_fetcher'].url_format, attr='GEE', name='Critical Risk Intersection', overlay=True).add_to(m)

                if show_infra:
                    infra_id = roads_img.getMapId({'min': 1, 'max': 1, 'palette': ['#2ecc71']})
                    folium.TileLayer(tiles=infra_id['tile_fetcher'].url_format, attr='TIGER/Line', name='Evacuation/Infrastructure', overlay=True).add_to(m)
                
                m.get_root().html.add_child(folium.Element(legend_html))

        # --- FINAL MAP RENDER ---
        st_folium(m, use_container_width=True, height=750, key=f"map_{selected_fire}")

    except Exception as e:
        st.error(f"Application Runtime Error: {e}")

# ==========================================
# PAGE 2: TECHNICAL DOCUMENTATION
# ==========================================
elif page == "Technical Documentation":
    st.title("Scientific Framework & Methodology")
    st.markdown("---")
    
    st.header("1. Intersection of Spatial Hazards")
    st.write("The core engine of this dashboard identifies geographic intersections where physical slope thresholds meet severe spectral vegetation loss, creating high-velocity runoff corridors.")
    st.latex(r"Risk = (Slope > \theta) \cap (dNBR < 0.1)")
    st.write("> **Note on Slope Threshold:** Academic literature (USGS) identifies slopes greater than 27 degrees as the primary initiation zones for post-fire debris flows. The interactive slider allows for sensitivity testing around this threshold.")

    st.header("2. Pedological Vulnerability & Geomorphology")
    st.write("""
    **Data Source:** OpenLandMap USDA Soil Texture Class  
    **Geomorphological Context:** This analysis intentionally isolates soil profiles that lack rapid drainage capacity or deep structural anchoring—specifically filtering out highly stable Sands and Loamy Sands. The algorithm identifies highly vulnerable profiles (Loams, Silts, and Clays). When these specific soils are exposed to intense precipitation triggers following severe canopy and root loss, they provide the primary sediment volume necessary to create viscous, destructive debris flows.
    """)

    st.header("3. Active Trigger Inference (NASA GPM IMERG)")
    st.write("""
    **Data Source:** NASA Global Precipitation Measurement (GPM) IMERG V07  
    **Resolution:** 0.1 Degrees (~10 km)
    
    **Geomorphological Context:** Debris flows are historically triggered by short-duration, high-intensity rainfall events during the first winter (typically 1 to 3 months) following the summer fire season. 
    
    This application integrates a real-time historical storm simulator that dynamically adjusts to a 90-day post-ignition window for any selected incident. It continuously calculates the peak rainfall rate (mm/hr) over the burn scar. A peak intensity exceeding the 5–10 mm/hr range is considered the critical geomorphological threshold for initiating a mass failure in freshly burned, hydrophobic landscapes.
    """)

    st.header("4. Core Data Architecture")
    st.write("""
    * **Sentinel-2 (ESA):** Multispectral imagery (20m resolution) utilized to calculate the Differenced Normalized Burn Ratio (dNBR).
    * **SRTM DEM (USGS):** 30-meter topographic gradients isolating gravitational energy exceeding the critical slope threshold.
    * **OpenLandMap USDA Soil Texture:** 250m resolution spatial predictions mathematically masked to isolate highly erodible soils.
    * **TIGER/Line (US Census Bureau):** National road networks integrated to calculate infrastructure exposure and prioritize emergency evacuation routes.
    """)