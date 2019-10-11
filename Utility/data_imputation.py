import load_files as lf
import wrangling
import numpy as np
import pandas as pd


def check_monotonicity(x, epsilon = 3):
    """
    Function that checks whether a list is increasing or
    decreasing with an epsilon terms that do not suit the pattern
    """
    # TAKE FIRST DIFFERENCE OF SERIES
    dx = np.diff(x)

    # N OF POSITIVE AND NEGATIVE NUMBERS
    positives = int(np.sum(np.array(dx) >= 0, axis=1))
    negatives = int(dx.shape[1] - positives)

    # RETURN MONOTONICITY BASED ON DELTA
    if negatives >= dx.shape[1] - epsilon:
        return -1     # Decreasing
    elif positives >= dx.shape[1] - epsilon:
        return 1      # Increasing
    else:
        return 0      # Extremum


def calc_monotonicity(data, horizon = 5, epsilon = 3):
    data = data.copy()

    data['Window'] = data.apply(lambda x: [data['Value'][x.name-horizon: x.name + horizon + 1]], axis=1)
    data['Monotonicity'] = data.apply(lambda x: check_monotonicity(x['Window'], epsilon = epsilon), axis = 1)

    return data


def fill_flow_apply(row, flow_data, level_data, on_level, epsilon = 0.01):
    """
    Function used in an apply method in fill flow function.
    It returns the a float prediction for the input as described above.

    Note that the solution is much faster if the flow data is indexed by timestamp.

    ~~~~~ INPUT  ~~~~~
    row:                  row from apply function
    on_level:             The on level of the pump. Suggested 95% quantile of level value
    epsilon:              a distance from the level corresponding to the missing flow value to be considered
    timestamp_index_flow: boolean stating if the supplied flow_data has timestamp as an index (Recomended)
    level_data:           not imputed level data
    flow_data:            not imputed flow data
    """

    level_row = level_data.loc[row['TimeStamp']]

    # RETURN MISSING VALUE IF NO MATCHING TIMESTAMP IN LEVEL DATA
    if len(level_row) == 0:
        return np.nan

    # RETURN 0 IF LEVEL IS INCREASING
    is_zero = ((level_row['Monotonicity'] == 1) & (level_row['Value'] < on_level))
    if is_zero:
        return 0.0

    # AT DECREASING LEVEL OR AROUND EXTREMUM OR ABOVE
    else:
        # LEVEL AT MISSING DATA POINT OF FLOW DATA
        level_value = level_row['Value']

        # LOCATE NEARBY TIMESTAMPS
        same_level_timestamps = level_data[(abs(level_data['Value'] - level_value) < epsilon) &
                                           (level_data['Monotonicity'] != 1)].index

        try:
            # GET FLOW VALUES FROM SIMILAR LEVELS
            flow_values = flow_data.loc[same_level_timestamps]['Value']
        except:
            # RETURN MISSING VALUE IF NO SIMILAR LEVELS FOUND
            return np.nan

        # IF TOO MUCH UNCERTAINTY IN FLOW OF SIMILAR LEVEL VALUES RETURN MISSING VALUE
        if np.std(flow_values) > (0.5 * np.mean(flow_values)):
            return np.nan
        # RETURN AVERAGE FLOW OF SIMILAR LEVEL VALUES
        else:
            return np.mean(flow_values)


def fill_flow(flow_data, level_data, epsilon=0.01, beta=4, horizon=5):
    """
    Function that applies fill_flow_apply (which operates on non-imputed data frames) to the missing values.
    Note that we need the merged data frame of flow and level as well here.
    """
    flow_data = flow_data.copy()
    level_data = level_data.copy()

    level_data = calc_monotonicity(level_data, horizon = horizon, epsilon = beta)

    # MERGE FLOW AND LEVEL TIMESTAMPS
    merged_flow_data, _ = wrangling.merge_flow_level(flow_data, level_data)

    # SELECT ONLY NA-VALUE FLOW DATA
    na_indices = merged_flow_data.index[merged_flow_data['Value'].isna()]
    non_na_indices = merged_flow_data.index[~merged_flow_data['Value'].isna()]

    merged_flow_data_missing = merged_flow_data[merged_flow_data.index.isin(list(na_indices))]

    # SET TIMESTAMP AS INDEX TO SPEED UP SELECTION
    flow_data.set_index("TimeStamp", inplace=True)
    level_data.set_index("TimeStamp", inplace=True)

    # CALC MAX LEVEL BOUNDARY
    on_level = np.quantile(level_data['Value'], q = 0.95)

    # IMPUTE VALUES
    merged_flow_data_missing['Value'] = merged_flow_data_missing.apply(lambda row: fill_flow_apply(row, on_level = on_level,
                                                                                     epsilon = epsilon,
                                                                                     level_data = level_data,
                                                                                     flow_data = flow_data), axis = 1)

    return pd.concat([pd.Series(merged_flow_data_missing['Value'].values,
                                index = merged_flow_data_missing['Value'].index),
                      pd.Series(merged_flow_data.loc[non_na_indices]["Value"].values,
                                index = merged_flow_data.loc[non_na_indices]["Value"].index)]).sort_index()
