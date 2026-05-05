import ee
ee.Initialize(project='gee-streamlit-app-490500')

geom = ee.Geometry.BBox(-117.65, 34.15, -117.35, 34.42)

# Force Landsat 5 only for BOTH images — no Landsat 7 fallback
pre = (ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA")
       .filterBounds(geom)
       .filterDate("2003-07-01", "2003-10-15")
       .sort("CLOUD_COVER")
       .first())

post = (ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA")
        .filterBounds(geom)
        .filterDate("2003-11-15", "2004-02-01")
        .sort("CLOUD_COVER")
        .first())

# Print which dates were selected so we can verify
print("Pre-fire date:", pre.get("DATE_ACQUIRED").getInfo())
print("Post-fire date:", post.get("DATE_ACQUIRED").getInfo())

pre_nbr  = pre.normalizedDifference(["B4","B7"])
post_nbr = post.normalizedDifference(["B4","B7"])
dnbr     = pre_nbr.subtract(post_nbr)

classified = (ee.Image(0)
    .where(dnbr.lt(0.10), 0)
    .where(dnbr.gte(0.10).And(dnbr.lt(0.27)), 1)
    .where(dnbr.gte(0.27).And(dnbr.lt(0.44)), 2)
    .where(dnbr.gte(0.44), 3)
    .clip(geom))

vis = classified.visualize(
    min=0, max=3,
    palette=["d3d3d3", "ffff00", "ff8c00", "cc0000"]
)

url = vis.getThumbURL({
    "region": geom,
    "dimensions": 1200,
    "format": "png"
})

print("\ndNBR URL:")
print(url)
