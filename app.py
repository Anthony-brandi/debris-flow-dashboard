# ==========================================
# PAGE 3: STATISTICAL REPORT
# ==========================================
elif page == "3. Statistical Report" and all_fires is not None:
    st.title("Watershed Statistical Analysis")
    
    run_stats = st.toggle("Generate Regional Vulnerability Map & Report", value=False)

    if run_stats:
        with st.spinner("Extracting atmospheric data and reducing spatial arrays across HUC-12 boundaries..."):
            try:
                area = ee.FeatureCollection(fire_subset.__geo_interface__)
                pre_date = ee.Date(manual_baseline.strftime('%Y-%m-%d'))
                fire_start_ee = ee.Date(default_alarm_dt.strftime('%Y-%m-%d'))
                target_date = fire_start_ee.advance(recovery_months, 'month')

                def get_nbr_median(date_obj):
                    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(area).filterDate(date_obj.advance(-2.5, 'month'), date_obj.advance(2.5, 'month')).median().clip(area).normalizedDifference(['B8', 'B12'])

                dnbr = get_nbr_median(pre_date).subtract(get_nbr_median(target_date))
                slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).clip(area)
                hazard_mask = slope.gte(slope_limit).And(dnbr.gt(dnbr_limit))
                
                hazard_area_img = hazard_mask.multiply(ee.Image.pixelArea()).rename('hazard_area')
                precip_img = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterBounds(area).filterDate(target_date.advance(-1, 'month'), target_date).select('precipitation').sum().rename('rainfall')
                
                combined_img = hazard_area_img.addBands(precip_img)
                
                # OPTIMIZATION: Tightly bound the HUC12 search to just the fire geometry to save memory
                huc12 = ee.FeatureCollection("USGS/WBD/2017/HUC12").filterBounds(area.geometry())
                
                # CRITICAL FIX: scale=500 and tileScale=16 prevents the GEE Timeout
                reduced_stats_fc = combined_img.reduceRegions(
                    collection=huc12,
                    reducer=ee.Reducer.sum().combine(reducer2=ee.Reducer.mean(), sharedInputs=False),
                    scale=500,
                    tileScale=16
                )
                
                def remove_geo(feature):
                    return ee.Feature(None, feature.toDictionary())
                
                stats_only = reduced_stats_fc.map(remove_geo)
                reduced_stats = stats_only.getInfo()
                
                ws_data = []
                for f in reduced_stats['features']:
                    props = f['properties']
                    
                    raw_sq_meters = props.get('hazard_area_sum', 0)
                    if raw_sq_meters is None: raw_sq_meters = 0
                    acres = raw_sq_meters * 0.000247105
                    
                    rain_mm = props.get('rainfall_mean', 0)
                    if rain_mm is None: rain_mm = 0
                    
                    if acres > 0:
                        ws_data.append({
                            "HUC-12 Watershed Name": props.get('name', 'Unknown'), 
                            "Active Hazard Footprint (Acres)": round(acres, 2),
                            "Mean Rainfall (mm)": round(rain_mm, 2)
                        })
                
                df_ws = pd.DataFrame(ws_data).sort_values(by="Active Hazard Footprint (Acres)", ascending=False)
                
                if df_ws.empty:
                    st.success("No active hazard areas detected. Try lowering the dNBR threshold in the sidebar.")
                else:
                    st.subheader("Regional Vulnerability Map")
                    
                    centroid = fire_subset.geometry.centroid.iloc[0]
                    m3 = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles='CartoDB dark_matter')
                    
                    w_outline = ee.Image(0).mask(0).paint(huc12, 'purple', 2)
                    folium.TileLayer(tiles=w_outline.getMapId({'palette':['purple']})['tile_fetcher'].url_format, attr='USGS', name='Watersheds').add_to(m3)
                    folium.TileLayer(tiles=hazard_mask.updateMask(hazard_mask).getMapId({'palette':['#ff7b00']})['tile_fetcher'].url_format, attr='GEE', name='Hazard Zones').add_to(m3)

                    folium.GeoJson(fire_subset.geometry, style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 2}).add_to(m3)
                    st_folium(m3, use_container_width=True, height=500)
                    st.markdown("---")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("Debris Flow Trigger Matrix")
                        st.dataframe(df_ws, use_container_width=True, hide_index=True)
                        
                        csv = df_ws.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Matrix (CSV)", data=csv, file_name=f'{selected_name}_trigger_matrix.csv', mime='text/csv')
                        
                    with c2:
                        st.subheader("Atmospheric Trigger Analysis")
                        scatter = alt.Chart(df_ws).mark_circle(size=200, color='#3498db', opacity=0.8).encode(
                            x=alt.X('Active Hazard Footprint (Acres):Q', title='Hazard Area (Acres)'),
                            y=alt.Y('Mean Rainfall (mm):Q', title='Cumulative Rainfall (mm)'),
                            tooltip=['HUC-12 Watershed Name', 'Active Hazard Footprint (Acres)', 'Mean Rainfall (mm)']
                        ).properties(height=350)
                        
                        text = scatter.mark_text(align='left', baseline='middle', dx=15, color="white").encode(text='HUC-12 Watershed Name:N')
                        st.altair_chart(scatter + text, use_container_width=True)

                st.info("""
                **Analytical Note:** This scatter matrix identifies Trigger Zones. Watersheds positioned in the top-right quadrant exhibit both critical geomorphic instability (high hazard area) and significant atmospheric loading (high rainfall), making them prime candidates for immediate debris flow monitoring.
                """)
            
            except Exception as e:
                st.error(f"Earth Engine Computation Timeout. Error details: {e}")

    else:
        st.info("Toggle 'Generate Regional Vulnerability Map & Report' to calculate spatial metrics.")
