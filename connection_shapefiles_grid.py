from Utility.wrangling import cell_index
from Utility.wrangling import vec_cell_index
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import shapely.geometry
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union, polygonize

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

def create_grid():
    low = (5.0875, 51.3345)
    high = (6.0865, 51.8635)

    dx = 0.037
    dy = 0.023
    crs = {'init': 'epsg:3857'}

    # Get vertical lines of grid
    x = low[0]
    linestrings = []
    while x < high[0]:
        y = low[1]
        gridpoints = []
        i = 0
        while y < high[1]:
            p = shapely.geometry.Point(x, y)
            gridpoints.append(p)
            if i > 0:
                A = gridpoints[i - 1]
                B = gridpoints[i]
                l = LineString([(A.x,A.y), (B.x,B.y)])
                linestrings.append(l)
            i += 1
            y += dy
        x += dx

    # Get horizontal lines of grid
    y = low[1]
    linestrings2 = []
    while y < high[1]:
        x = low[0]
        gridpoints = []
        i = 0
        while x < high[0]:
            p = shapely.geometry.Point(x, y)
            gridpoints.append(p)
            if i > 0:
                A = gridpoints[i - 1]
                B = gridpoints[i]
                l = LineString([(A.x,A.y), (B.x,B.y)])
                linestrings2.append(l)
            i += 1
            x += dx
        y += dy
    df = pd.DataFrame(linestrings, columns=['geometry'])
    df2 = pd.DataFrame(linestrings2, columns=['geometry'])
    df = df.append(df2, ignore_index=True)
    gpd_df = gpd.GeoDataFrame(df, crs=crs, geometry=df['geometry'])
    return gpd_df


system = loc_pump(path + file, 'Haarsteeg')
tmpWGS84 = system.to_crs({'proj':'longlat', 'ellps':'WGS84', 'datum':'WGS84'})
tmpWGS84["x"] = tmpWGS84["geometry"].to_crs({'init': 'epsg:4326'}).centroid.x
tmpWGS84["y"] = tmpWGS84["geometry"].to_crs({'init': 'epsg:4326'}).centroid.y

fig, ax = plt.subplots()
tmpWGS84.plot(ax= ax)
crs = {'init':'epsg:3857'}

create_grid().plot()
plt.show()


def divide_areas(df):
    """"Function that divides the areas in sewage system to cells corresponding to rain predictions
    Input:
    df: a Geopandas dataframe with areas in sewage system

    Output:
    Dataframe with areas split such that correct rain prediction can be chosen"""

    tmpWGS84 = df.to_crs({'proj':'longlat', 'ellps':'WGS84', 'datum':'WGS84'})
    grid = create_grid()
    lst_new_areas = []
    lst_ID = []
    for j in range(0, tmpWGS84['geometry'].count()):
        for i in range(0, grid['geometry'].count()):
            if grid['geometry'][i].intersects(tmpWGS84['geometry'][j]): # Find all the combinations that intersect
                try:
                    line = grid['geometry'][i]
                    poly = tmpWGS84['geometry'][j]
                    merged = linemerge([poly.boundary, line])
                    borders = unary_union(merged)
                    polygons = polygonize(borders)
                    for p in polygons:
                        lst_ID.append(tmpWGS84['RGDIDENT'][j])
                        lst_new_areas.append(p)
                except:
                    lst_new_areas.append(tmpWGS84['geometry'][j])
                    lst_ID.append(tmpWGS84['RGDIDENT'][j])
                    print('Did not do it right for', i, j)

    df_divided = pd.DataFrame({'RGDIDENT': lst_ID, 'geometry': lst_new_areas})
    crs = {'init': 'epsg:3857'}
    df_out = gpd.GeoDataFrame(df_divided, crs = crs, geometry = df_divided['geometry'])
    df_out = df_out.append(tmpWGS84.loc[~tmpWGS84['RGDIDENT'].isin(df_out['RGDIDENT'])][['RGDIDENT', 'geometry']],
                       ignore_index=True)
    df_out['x'] = df_out['geometry'].centroid.x
    df_out['y'] = df_out['geometry'].centroid.y
    df_out['area'] = df_out['geometry'].area
    df_out['Cell'] = df_out.apply(lambda row: np.vectorize(cell_index(row['x'], row['y'])), axis=1)
    return df_out
