# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Objective: Build a model to estimate hourly flow        #
# through a pump 24 hours in advance.                     #
# BY SEBASTIAN DANNEHL                                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.api as sm

import tensorflow as tf
import keras

import wrangling
import utility

import holidays

def add_predictor_columns(data):
    """
    Will return predictive variables given a data-set with the 'TimeHour' column.
    'TimeHour' can be created by applying the .replace() method on the 'TimeStamp'
    column. The following variables will be added:

    NAME ~~~~~~~~~~~~~~~ COLUMN ~~~~~~~ FORMAT
    Hour of the day      hour_XX        Dummy, binary
    Month of the year    month_XX       Dummy, binary
    Holiday              is_holiday     Binary

    Holiday is based on all holidays in the Netherlands in 2018, 2019.
    """
    # Fetch holidays of given period
    NL_holidays = [i[0] for i in holidays.Netherlands(years = [2018, 2019]).items()]

    # Create dummies for hour of day and month of year
    hour_dummies = pd.get_dummies(data["TimeHour"].apply(lambda i: i.hour), prefix="hour")
    month_dummies = pd.get_dummies(data["TimeHour"].apply(lambda i: i.month), prefix="month")

    # Check each date whether in holidays
    is_holiday = data["TimeHour"].apply(lambda i: i.date() in NL_holidays).astype(int)

    # Concatenate and add constant/intercept
    X = pd.concat([hour_dummies, month_dummies, is_holiday], axis=1)
    X["Constant"] = 1

    return X


class flow_model:
    """
    Can be used to predict hourly flow based on rain prediction and time.
    Data set will be created at initialization based on flow_data, level_data
    and rain prediction grid data. The latter needs to be a tuple as read
    by load_files.get_rain_prediction().
    """

    def __init__(self, flow_data, level_data, rain_grid,
                 padding=5, multiple=True, steps=12, imputation="simple"):
        """
        Creates data set for model based on flow_data, level_data, rain prediction.
        """
        # Selects a grid around a specific pump.
        # Size will be (1+2*padding)x(1+2*padding).
        rain_grid = wrangling.grid_area(rain_prediction[1], "Drunen", padding=padding, reduced=True)

        # Omit minor data defficiencies
        flow_data = wrangling.clean_mes_data(flow_data, convert_timestamp=False)
        level_data = wrangling.clean_mes_data(level_data, convert_timestamp=False)

        # Merges flow and level on timestamps, as normal flow data is biased
        # given no measurements are made when there is no flow.
        flow_data, level_data = wrangling.merge_flow_level(flow_data, level_data)

        # Can perform simple imputation or LM-imputation
        if imputation == "simple":
            flow_data = wrangling.fill_flow(flow_data)
        elif imputation == "complex":
            flow_data = data_imputation.fill_flow(flow_data, level_data)
        else:
            pass

        # Groups flow by hour
        flow_data_by_hour = flow_by_hour(flow_data)

        # Selects parts of grid and hourly flow data where the other is available
        # for the same hour
        grid, flow_data_by_hour = match_by_timestamp((rain_prediction[0], rain_grid),
                                                      flow_data_by_hour,
                                                      multiple=multiple, steps=steps)

        # Concatenate grid data and other variables
        X = add_predictor_columns(flow_data_by_hour).values
        X = np.concatenate((grid, X), axis=1)

        # Select dependent variable
        y = flow_data_by_hour["Flow"].values

        # Add variables to class
        self.X = X
        self.y = y
        self.padding = padding
        self.steps = steps

    def StochasticGradientDescent(self, lr=0.03, epochs=400, batch_size=1024, validation_split=0.1, cv=False):
        """
        Builds a gradient descent model.
        Based on whether cv is True will return a cross validation score.
        If cv is True model will not be based on the whole data set.
        """
        if cv:
            cvscores = []
            for train, test in KFold(10).split(X):
                # Build single-node neural network
                model = Sequential()
                model.add(Dense(1, input_dim=self.X.shape[1]))

                model.compile(optimizer = optimizers.Adam(lr=lr), loss='mean_squared_error')
                model.fit(self.X, self.y,
                          validation_split=0,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=False)

                # Add cv score
                scores = model.evaluate(X[test], y[test], verbose=0)
                cvscores.append(scores[1])

            self.cvscores = cvscores

            self.model = model

            return np.mean([np.sqrt(i) for i in cvscores])
        else:
            # Build single-node neural network
            model = Sequential()
            model.add(Dense(1, input_dim=self.X.shape[1]))

            model.compile(optimizer = optimizers.Adam(lr=lr), loss='mean_squared_error')
            model.fit(self.X, self.y,
                      validation_split=validation_split,
                      epochs=epochs,
                      batch_size=batch_size,
                      shuffle=False)

            self.model = model

            return
