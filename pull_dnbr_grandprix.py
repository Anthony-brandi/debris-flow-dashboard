import ee
ee.Initialize(project='gee-streamlit-app-490500')

# Grand Prix fire area
geom = ee.Geometry.BBox(-117.65, 34.15, -117.35, 34.42)

# Pre-fire NBR (summer 2003, Landsat 5)
pre = (ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA")
       .filterBounds(geom)
       .filterDate("2003-07-01", "2003-10-15")
       .sort("CLOUD_COVER")
       .first())

# Post-fire NBR (Nov-Dec 2003, after Oct 21 ignition)
post = (ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA")
        .filterBounds(geom)
        .filterDate("2003-11-01", "2004-01-15")
        .sort("CLOUD_COVER")
        .first())

# dNBR = pre NBR minus post NBR
pre_nbr  = pre.normalizedDifference(["B4","B7"]).rename("pre_nbr")
post_nbr = post.normalizedDifference(["B4","B7"]).rename("post_nbr")
dnbr     = pre_nbr.subtract(post_nbr).rename("dNBR")

# Classify: unburned=grey, low=yellow, mod=orange, high=red
classified = (dnbr
    .where(dnbr.lt(0.1),   0)
    .where(dnbr.gte(0.1).And(dnbr.lt(0.27)), 1)
    .where(dnbr.gte(0.27).And(dnbr.lt(0.44)), 2)
    .where(dnbr.gte(0.44), 3))

vis = classified.visualize(
    min=0, max=3,
    palette=["d3d3d3", "ffff00", "ff8c00", "cc0000"]
)

url = vis.getThumbURL({
    "region": geom,
    "dimensions": 1200,
    "format": "png"
})

print("dNBR BURN SEVERITY MAP:")
print(url)
