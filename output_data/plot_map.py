import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# Adjust folder path
folder_path = "output_data/trajectory"

# Load files
file_pattern = os.path.join(folder_path, "TRAJ_COMP_*.csv")
files = glob.glob(file_pattern)

if not files:
    print("No matching files found!")
    exit()

lat_all = []
lon_all = []
dm_all = []

# Read data
for file in files:
    try:
        df = pd.read_csv(file)

        required_cols = ["Lat [deg]", "Lon [deg]", "d_m [kg]"]

        if all(col in df.columns for col in required_cols):
            df = df.dropna(subset=required_cols)

            lat_all.append(df["Lat [deg]"].values)
            lon_all.append(df["Lon [deg]"].values)
            dm_all.append(df["d_m [kg]"].values)
        else:
            print(f"Missing columns in {file}")

    except Exception as e:
        print(f"Error with {file}: {e}")

# Merge arrays
lat = np.concatenate(lat_all)
lon = np.concatenate(lon_all)
dm = np.concatenate(dm_all)

# 1° grid
lat_bins = np.arange(-90, 91, 1)
lon_bins = np.arange(-180, 181, 1)

# Weighted histogram
heatmap, _, _ = np.histogram2d(
    lat,
    lon,
    bins=[lat_bins, lon_bins],
    weights=dm
)

# Log scaling
heatmap = np.log1p(heatmap)

# Important: hide zero values
heatmap[heatmap == 0] = np.nan

# Plot
plt.figure(figsize=(14, 7))
ax = plt.axes(projection=ccrs.PlateCarree())

# Background
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="white")
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)

# Heatmap
mesh = ax.pcolormesh(
    lon_bins,
    lat_bins,
    heatmap,
    cmap="coolwarm",
    shading="auto",
    alpha=0.9,
    transform=ccrs.PlateCarree()
)

# Equator
ax.plot(
    [-180, 180], [0, 0],
    color="black",
    linewidth=1.2,
    linestyle="--",
    transform=ccrs.PlateCarree(),
    label="Equator"
)

# Longitude lines + labels
gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.5,
    color="gray",
    alpha=0.5,
    linestyle="--"
)

# Only show longitude labels at the bottom
gl.top_labels = False
gl.right_labels = False

# Step size (e.g. every 30°)
gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 30))
gl.ylocator = plt.FixedLocator(np.arange(-90, 91, 30))

# Formatting
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}

# Colorbar
cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02)
cbar.set_label("log(1 + Σ d_m [kg])")

# Title
plt.title("Summed Mass (d_m) - with Longitude Lines", fontsize=14)

ax.legend(loc="lower left")

plt.tight_layout()
plt.show()