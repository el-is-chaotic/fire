# %%
# app.py
import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import rasterio
from rasterio.mask import mask
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import base64
import imageio
import pandas as pd

st.set_page_config(layout="wide")
st.title("Vegetation Recovery & Fire Impacts - Biscuit Fire")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("NDVI Map Settings")
year_options = [2001, 2003, 2016, 2018, 2019, 2023]
selected_year = st.sidebar.selectbox("Select NDVI Year to Display", year_options)

st.sidebar.header("NDVI Change Settings")
change_pairs = [("2001","2003"), ("2003","2023"), ("2001","2023"), ("2016","2018"), ("2016","2019")]
selected_change = st.sidebar.selectbox("Select NDVI Change Period", change_pairs)

st.sidebar.header("Burn Class Settings")
burn_classes = [0, 1, 2, 3]
selected_burn_class = st.sidebar.selectbox("Select Burn Class to Analyze", burn_classes)

# ------------------------------
# Paths & Data
# ------------------------------
# Load GeoJSON
biscuit_gdf = gpd.read_file("biscuit_clean.geojson")

# Convert Timestamps in properties to strings, leave geometry intact
for col in biscuit_gdf.columns:
    if col != "geometry":
        biscuit_gdf[col] = biscuit_gdf[col].apply(
            lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x
        )

# Extract geometry for masking raster
biscuit_geom = [feature["geometry"] for feature in biscuit_gdf.__geo_interface__["features"]]

ndvi_dir = "NDVI_Annual_Images"

# ------------------------------
# Functions
# ------------------------------
def read_masked_ndvi(path, shapes):
    with rasterio.open(path) as src:
        out_image, out_transform = mask(src, shapes, crop=True)
        array = out_image[2]  # NDVI band (assuming band 3 is NDVI)
        array = array.astype(float)
        array[array == src.nodata] = np.nan
    return array, out_transform, src.crs

def array_to_rgb_png(arr, cmap_name="RdYlGn"):
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(arr), vcenter=0, vmax=np.nanmax(arr))
    rgba_img = cmap(norm(arr))
    rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)
    buf = io.BytesIO()
    imageio.imwrite(buf, rgb_img, format="png")
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ------------------------------
# Load NDVI for selected year
# ------------------------------
ndvi_path = f"{ndvi_dir}/NDVI_{selected_year}.tif"
ndvi_array, transform, crs = read_masked_ndvi(ndvi_path, biscuit_geom)

# ------------------------------
# Initialize map
# ------------------------------
m = folium.Map(location=[42.0, -123.5], zoom_start=9)

# Add Biscuit boundary
folium.GeoJson(
    biscuit_gdf,
    name="Biscuit Fire Boundary",
    style_function=lambda x: {"fillColor": "red", "color": "red", "weight": 2, "fillOpacity": 0.3}
).add_to(m)

# Add NDVI overlay
png_encoded = array_to_rgb_png(ndvi_array)
height, width = ndvi_array.shape
left, top = transform[2], transform[5]
right = left + width * transform[0]
bottom = top + height * transform[4]

folium.raster_layers.ImageOverlay(
    image='data:image/png;base64,' + png_encoded,
    bounds=[[bottom, left], [top, right]],
    name=f"NDVI {selected_year}",
    opacity=0.6
).add_to(m)

# ------------------------------
# NDVI Change Overlay
# ------------------------------
year1, year2 = selected_change
ndvi1, _, _ = read_masked_ndvi(f"{ndvi_dir}/NDVI_{year1}.tif", biscuit_geom)
ndvi2, _, _ = read_masked_ndvi(f"{ndvi_dir}/NDVI_{year2}.tif", biscuit_geom)
delta_ndvi = ndvi2 - ndvi1

png_delta = array_to_rgb_png(delta_ndvi, cmap_name="RdYlGn")
folium.raster_layers.ImageOverlay(
    image='data:image/png;base64,' + png_delta,
    bounds=[[bottom, left], [top, right]],
    name=f"NDVI Change {year1}-{year2}",
    opacity=0.7
).add_to(m)

# Layer control
folium.LayerControl().add_to(m)

# ------------------------------
# Display Map in Streamlit
# ------------------------------
st.subheader(f"NDVI Map - {selected_year}")
st_folium(m, width=800, height=500)

# ------------------------------
# Burn Class Statistics
# ------------------------------
burn_csv = "burn_class_delta_ndvi.csv"
try:
    df_burn = pd.read_csv(burn_csv)
    cls_mask = df_burn['burn_class'] == selected_burn_class
    delta_cls = df_burn.loc[cls_mask, 'delta_ndvi']

    st.subheader(f"NDVI Change Histogram for Burn Class {selected_burn_class}")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(delta_cls.dropna(), bins=50, color="green", edgecolor="black")
    ax.set_xlabel("ΔNDVI")
    ax.set_ylabel("Pixel Count")
    ax.set_title(f"NDVI Change Histogram - Burn Class {selected_burn_class}")
    ax.grid(True)
    st.pyplot(fig)
except FileNotFoundError:
    st.warning(f"{burn_csv} not found. Burn class statistics will not be shown.")

