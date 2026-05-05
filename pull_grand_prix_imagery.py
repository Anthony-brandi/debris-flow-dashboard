import ee
ee.Initialize(project='gee-streamlit-app-490500')

# Tighter focus on the debris fan area — Cajon Pass / Cucamonga
geom = ee.Geometry.BBox(-117.65, 34.15, -117.35, 34.42)

# Pre-fire: summer 2003 using Landsat 5 (no scan line failure)
pre = (ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA")
       .filterBounds(geom)
       .filterDate("2003-07-01", "2003-10-15")
       .sort("CLOUD_COVER")
       .first()
       .select(["B3","B2","B1"])
       .visualize(min=0.05, max=0.35, gamma=1.4))

# Post-debris flow: Jan-Feb 2004
post = (ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA")
        .filterBounds(geom)
        .filterDate("2004-01-01", "2004-04-01")
        .sort("CLOUD_COVER")
        .first()
        .select(["B3","B2","B1"])
        .visualize(min=0.05, max=0.35, gamma=1.4))

pre_url  = pre.getThumbURL({"region": geom, "dimensions": 1024, "format": "png"})
post_url = post.getThumbURL({"region": geom, "dimensions": 1024, "format": "png"})

print("PRE-FIRE (Landsat 5):")
print(pre_url)
print()
print("POST-DEBRIS FLOW (Landsat 5):")
print(post_url)
