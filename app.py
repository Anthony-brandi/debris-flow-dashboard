# Check if the fires are hiding under weird names
import geopandas as gpd

fires = gpd.read_file('CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view/CA_Perimeters_CAL_FIRE_NIFC_FIRIS_public_view.shp')
all_names = fires['incident_n'].dropna().unique()

# Look for any name containing "DIXIE", "CALDOR", or "MONUMENT"
missing_fires = [name for name in all_names if "DIXIE" in str(name).upper() or "CALDOR" in str(name).upper() or "MONUMENT" in str(name).upper()]

print("Found these matches:", missing_fires)
