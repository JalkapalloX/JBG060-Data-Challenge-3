from Utility.wrangling import cell_index

import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import shapely.geometry
import pyproj

path = "C:/Users/20175825/OneDrive - TU Eindhoven/Data Science Bachelor Year 3/Data Challenge 3/waterschap-aa-en-maas_sewage_2019_NEW/"
file = 'sewer_model/aa-en-maas_sewer_shp/Rioleringsdeelgebied.shp'


def loc_pump(shapefile, pump):
    """Function that selects the locations for the pump in the shapefile
    Input:
    shapefile: path to a shapefile
    pump: string corresponding to pump, either 'Haarsteeg' or 'Bokhoven'

    Output:
    Dataframe with locations for selected pump """

    system = gpd.read_file(shapefile)
    if pump == 'Haarsteeg':
        locs = system[system['RGDIDENT'].str.startswith('HEU')].copy()  # Selects all locations that lead to Haarsteeg
    elif pump == 'Bokhoven':
        locs = system[system['RGDIDENT'].str.startswith('HEU')].copy()
    else:
        print('No pump was selected')
    locs['area'] = locs.area
    locs["x"] = locs["geometry"].to_crs({'init': 'epsg:4326'}).centroid.x
    locs["y"] = locs["geometry"].to_crs({'init': 'epsg:4326'}).centroid.y
    vec_cell_index = np.vectorize(cell_index)
    locs['Cell'] = locs.apply(lambda row: vec_cell_index(row['x'], row['y']), axis=1)
    return locs


loc_pump(path + file, 'Haarsteeg')

rain_arr = np.loadtxt(
    path + "sewer_data/rain_grid_prediction/knmi/Knmi.Harmonie_99.0.1.61.105_2018-01-01T00h00m00s_2018-01-02T00h00m00s_2018-01-02T01h00m00s.asc",
    skiprows=7)


def lookup_prediction(locs, predictions):
    """Function that looks up predictions of a certain hour and assigns volume to it
    Input:
    Locs: Shapefile dataframe with a 'Cell' column
    Predictions: array of rain predictions

    Output:
    Dataframe with two extra columns
    """
    rain_pred = locs.copy()
    rain_pred['Pred'] = rain_pred.apply(lambda row: predictions[row['Cell']], axis=1)
    rain_pred['Predicted Volume'] = rain_pred['Pred'] * rain_pred['area']
    return rain_pred


lookup_prediction(loc_pump(path + file, 'Haarsteeg'), rain_arr)

## ===============================
## Create grid
## ===============================

import shapely.geometry
import pyproj

xmin =  -0.0185
ymax = 48.9885
ncols = 300
nrows = 300
dx = 0.037
dy = 0.023
xmax = -0.0185 + ncols * dx
ymin = 48.9885 - nrows * dy

# Set up projections
p_ll = pyproj.Proj(init='epsg:4326')
p_mt = pyproj.Proj(init='epsg:3857') # metric; same as EPSG:900913

stepsize = 1000 # 1 km grid step size

# Project corners to target projection
low = pyproj.transform(p_ll, p_mt, xmin, ymin) # Transform point to 3857
high = pyproj.transform(p_ll, p_mt, xmax, ymax) # .. same

low, high

# Iterate over 2D area
gridpoints = []
x = low[0]
for i in range(0, 10):  # test value
    print(i)
    y = low[1]
    while y < high[1]:
        p = shapely.geometry.Point(pyproj.transform(p_mt, p_ll, x, y))
        gridpoints.append(p)
        y += stepsize
    x += stepsize

gridpoints