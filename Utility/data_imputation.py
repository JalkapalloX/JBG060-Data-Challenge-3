# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Objective: Estimate missing measurements of flow based  #
# on the level and slope of level at this time.           #
# BY BLAZEJ MANCZAK                                       #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

import load_files as lf
import wrangling
import numpy as np
import pandas as pd


def check_monotonicity(x, epsilon = 3):
    """
    Function that checks whether a list is increasing or
    decreasing with an epsilon terms that do not suit the pattern
    """
    # Take first difference of series
    dx = np.diff(x)

    # Get number of positive and negative values
    positives = int(np.sum(np.array(dx) >= 0, axis=1))
    negatives = int(dx.shape[1] - positives)

    # Return monotonicity based on delta
    if negatives >= dx.shape[1] - epsilon:
        return -1     # Decreasing
    elif positives >= dx.shape[1] - epsilon:
        return 1      # Increasing
    else:
        return 0      # Extremum


def calc_monotonicity(data, horizon = 5, epsilon = 3):
    """
    Calculates the monotonicity within a sliding window.
    Adds columns 'Window' and 'Monotonicity' to data.
    """
    data = data.copy()

    # Window column is a tuple of (index-horizon)-(index+horizon)
    data['Window'] = data.apply(lambda x: [data['Value'][x.name-horizon: x.name + horizon + 1]], axis=1)
    
    # Calculates monotonicity based on check_monotonicity()
    data['Monotonicity'] = data.apply(lambda x: check_monotonicity(x['Window'], epsilon = epsilon), axis = 1)

    return data


def fill_flow_apply(row, flow_data, level_data, on_level, epsilon = 0.01):
    """
    Function used in an apply method in fill flow function.
    It returns the a float prediction for the input as described above.

    ~~~~~ INPUT  ~~~~~
    flow_data:            not imputed flow data
    level_data:           not imputed level data
    row:                  row from apply function
    on_level:             Level where the pump turns on. Suggested 95% quantile of level value
    epsilon:              a distance from the level corresponding to the missing flow value to be considered
    timestamp_index_flow: boolean stating if the supplied flow_data has timestamp as an index (Recomended)
    """

    # Select specific predefined row
    level_row = level_data.loc[row['TimeStamp']]

    # Return missing value if no matching timestamp in level data
    if len(level_row) == 0:
        return np.nan

    # Return 0 if level is increasing
    is_zero = ((level_row['Monotonicity'] == 1) & (level_row['Value'] < on_level))
    if is_zero:
        return 0.0

    # If level is decreasing, around extremum (on-level) or above
    else:
        # Level at missing data point to be imputed
        level_value = level_row['Value']

        # Locate nearby timestamps
        same_level_timestamps = level_data[(abs(level_data['Value'] - level_value) < epsilon) &
                                           (level_data['Monotonicity'] != 1)].index

        try:
            # Get flow values from similar levels
            flow_values = flow_data.loc[same_level_timestamps]['Value']
        except:
            # Return missing value if no similar levels found
            return np.nan

        # If too much uncertainty in flow of similar level values return missing value
        if np.std(flow_values) > (0.5 * np.mean(flow_values)):
            return np.nan

        # Return average flow of similar level values
        else:
            return np.mean(flow_values)


def fill_flow(flow_data, level_data, epsilon=0.01, beta=4, horizon=5):
    """
    Function that applies fill_flow_apply (which operates on non-imputed data frames) to the missing values.
    Note that we need the merged data frame of flow and level as well here.
    
    Returns a series with imputed values.
    """
    flow_data = flow_data.copy()
    level_data = level_data.copy()

    # Calculate monotonicity
    level_data = calc_monotonicity(level_data, horizon = horizon, epsilon = beta)

    # Merges flow and level on timestamps, as normal flow data is biased
    # given no measurements are made when there is no flow.
    merged_flow_data, _ = wrangling.merge_flow_level(flow_data, level_data)

    # Separate missing value flow data, and non-missing value flow data
    na_indices = merged_flow_data.index[merged_flow_data['Value'].isna()]
    non_na_indices = merged_flow_data.index[~merged_flow_data['Value'].isna()]

    merged_flow_data_missing = merged_flow_data[merged_flow_data.index.isin(list(na_indices))]

    # Set timestamp as index to dpeed up selection
    flow_data.set_index("TimeStamp", inplace=True)
    level_data.set_index("TimeStamp", inplace=True)

    # Calculate max level boundary
    on_level = np.quantile(level_data['Value'], q = 0.95)

    # Impute values
    merged_flow_data_missing['Value'] = merged_flow_data_missing.apply(lambda row: fill_flow_apply(row, on_level = on_level,
                                                                                     epsilon = epsilon,
                                                                                     level_data = level_data,
                                                                                     flow_data = flow_data), axis = 1)
    
    # Return Series of imputed values
    return pd.concat([pd.Series(merged_flow_data_missing['Value'].values,
                                index = merged_flow_data_missing['Value'].index),
                      pd.Series(merged_flow_data.loc[non_na_indices]["Value"].values,
                                index = merged_flow_data.loc[non_na_indices]["Value"].index)]).sort_index()
