import ee
ee.Initialize(project='gee-streamlit-app-490500')

# Zoom tight on Upper Cucamonga Creek fan apex area
# This is where GrandPrix3 and GrandPrix4 were measured
geom = ee.Geometry.BBox(-117.58, 34.18, -117.42, 34.32)

pre = (ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA")
       .filterBounds(geom)
       .filterDate("2003-07-01", "2003-10-15")
       .sort("CLOUD_COVER")
       .first()
       .select(["B3","B2","B1"])
       .visualize(min=0.03, max=0.28, gamma=1.5))

post = (ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA")
        .filterBounds(geom)
        .filterDate("2004-01-10", "2004-04-01")
        .sort("CLOUD_COVER")
        .first()
        .select(["B3","B2","B1"])
        .visualize(min=0.03, max=0.28, gamma=1.5))

pre_url  = pre.getThumbURL({"region": geom, "dimensions": 1024, "format": "png"})
post_url = post.getThumbURL({"region": geom, "dimensions": 1024, "format": "png"})

print("PRE-FIRE zoomed:")
print(pre_url)
print()
print("POST-DEBRIS FLOW zoomed:")
print(post_url)
