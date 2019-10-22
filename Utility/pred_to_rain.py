# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Objective: Find the cell in the rain prediction grid    #
# corresponding to the centroid of an area.               #
# BY SEBASTIAN DANNEHL                                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

import pandas as pd
import geopandas as gpd
import numpy as np
import datetime
import wrangling


class pred_to_rain:
    def __init__(self, rain_data, rain_prediction, area_data):
        reduced = rain_prediction[1].shape[1] < 300

        if rain_data["Start"].dtype != "<M8[ns]":
            rain_data["Start"] = pd.to_datetime(rain_data["Start"])

        # Sort rain data by time
        rain_data.sort_values("Start", inplace=True)
        rain_data.reset_index(drop=True, inplace=True)

        # Narrow area data to streets that occur in rain data
        area_data = area_data.loc[area_data["area_name"].isin(rain_data.columns)].reset_index(drop=True)

        # Add column of row/column index in prediction grid
        area_data["x"] = area_data["geometry"].to_crs({'init': 'epsg:4326'}).centroid.x
        area_data["y"] = area_data["geometry"].to_crs({'init': 'epsg:4326'}).centroid.y

        # Get grid indices for x and y coordinates
        indices = wrangling.vec_cell_index(area_data["x"], area_data["y"], reduced=reduced)
        indices = list(zip(indices[0], indices[1]))
        area_data["loc"] = indices

        #
        rain_data_times = rain_data["Start"].apply(lambda i: i.replace(minute=0))
        base_time = rain_data_times[0]
        rain_data_int_times = rain_data_times.apply(lambda i: int((i - base_time).total_seconds()))

        date_to_index = pd.Series(range(len(rain_prediction[0])),
                                  index = rain_prediction[0]["start"]\
                                  .apply(lambda i: int((i - base_time).total_seconds())))

        # grid_layers: index of rain prediction grid layer for each 'Start'-TimeStamp
        # in rain_data
        grid_layers = date_to_index.loc[date_to_index.index.intersection(rain_data_int_times)]
        grid_layers = grid_layers.loc[~grid_layers.index.duplicated(keep='first')]

        # ADD DATA TO CLASS
        self.rain_data = rain_data
        self.rain_prediction = rain_prediction
        self.area_data = area_data
        self.grid_layers = grid_layers
        self.base_time = base_time

    def exact(self):
        rain_values = self.area_data["loc"].apply(lambda i: pd.Series(rain_prediction[1][self.grid_layers, i[1], i[0]]))
        rain_values.index = self.area_data["area_name"]
        rain_values = rain_values.T
        rain_values["Start"] = pd.Series(self.grid_layers.index)\
                                         .apply(lambda i: self.base_time + datetime.timedelta(seconds = i))
        rain_values["End"] = rain_values["Start"].apply(lambda i: i + datetime.timedelta(hours = 1))
        rain_values = rain_values.loc[:,~rain_values.columns.duplicated()]

        return rain_values

    def estimated(self):
        pass
