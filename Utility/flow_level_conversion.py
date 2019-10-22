# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Objective: Estimate a coefficient between total flow    #
# through a pump and level change. With this, sewer       #
# capacity and sewer intake over a period of time can be  #
# estimated.                                              #
# BY SEBASTIAN DANNEHL                                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import wrangling
import utility


class generate_coefficient:
    """
    Class used to estimate a coefficient of conversion between flow in a flow peak
    and level change in a level drop period. Data will be cleaned at initialization.
    
    Linear regression has to be used afterwards on 'Flow ~ I + AdjDelta'.
    
    ~~~~~ FUTURE IMPROVEMENTS ~~~~~
    Clustering may have to be employed in the future to select typical flow peaks
    rather than heuristic thresholds.
    A better adjustment by rain could be beneficial.
    """
    def __init__(self, flow_data, level_data):
        # Omit minor data defficiencies
        flow_data = wrangling.clean_mes_data(flow_data)
        level_data = wrangling.clean_mes_data(level_data)

        # Merges flow and level on timestamps, as normal flow data is biased
        # given no measurements are made when there is no flow.
        flow_data, level_data = wrangling.merge_flow_level(flow_data, level_data)

        # Impute missing measurements
        flow_data = wrangling.fill_flow(flow_data)
        level_data = wrangling.fill_level(level_data)

        # Rename Value column, calculate flow/s
        flow_data.rename(columns={"Value": "Flow/s"}, inplace=True)
        level_data.rename(columns={"Value": "Level"}, inplace=True)
        flow_data["Flow/s"] = flow_data["Flow/s"] / 3600

        # Create 'TimeSpan' aka time since last measurement
        flow_data["TimeSpan"] = flow_data["TimeStamp"].diff(1).apply(lambda i: i.seconds)
        level_data["TimeSpan"] = level_data["TimeStamp"].diff(1).apply(lambda i: i.seconds)

        # Calculate Flow
        flow_data["Flow"] = flow_data["Flow/s"] * flow_data["TimeSpan"]

        # Add data to class
        self.flow_data = flow_data
        self.level_data = level_data


    def to_dry_data(self, rain_data, area_data, min_dry_series=1, village_code=None, dry_threshold=1):
        """
        Readjusts data stored in class to only consider dry days.
        A dry day is defined as a day where the mean rainfall of all areas with village code
        (XXX-VIL-XXX) village_code has not been higher than dry_threshold for a consecutive
        min_dry_series days.
        """
        # Calculate which days are dry
        dry_days = wrangling.summarize_rain_data(rain_data, area_data,
                                                 village_code=village_code,
                                                 dry_threshold=dry_threshold)

        # Get list of dry dates
        dry_days = dry_days.loc[dry_days["DrySeries"] >= min_dry_series, "Date"].reset_index(drop=True)

        # Select all dry days
        self.flow_data = self.flow_data.loc[self.flow_data["TimeStamp"]\
                                       .apply(lambda i: i.date() in dry_days.to_list()),:]\
                                       .reset_index(drop=True)
        self.level_data = self.level_data.loc[self.level_data["TimeStamp"]\
                                         .apply(lambda i: i.date() in dry_days.to_list()),:]\
                                         .reset_index(drop=True)


    def add_groups(self):
        """
        Will add flow peak (self.flow_groups) and level drop data (self.level_groups),
        plus additional columns for analysis.
        """
        # Add 'group' column. Every non-zero integer identifies a flow peak or
        # level drop that the measurement lies within
        self.flow_data["group"] = wrangling.flow_group(self.flow_data["Flow/s"])
        self.level_data["group"] = wrangling.level_group(self.level_data["Level"])

        # Initialize group data frames
        flow_groups = self.flow_data.sort_values("TimeStamp")\
                                    .groupby("group")\
                                    .apply(lambda i: i.iloc[0])[["TimeStamp", "group"]]
        level_groups = self.level_data.sort_values("TimeStamp")\
                                      .groupby("group")\
                                      .apply(lambda i: i.iloc[0])[["TimeStamp", "group"]]

        # Add variables for analysis
        # LEVEL GROUPS
        # Absolute change in level
        level_groups["Delta"] = self.level_data.groupby("group")["Level"].min() -\
                                self.level_data.groupby("group")["Level"].max()
        # Calcuate difference in first and last measurement time in seconds
        level_groups["TimeSpan"] = (self.level_data.groupby("group").apply(lambda i: i.iloc[-1]["TimeStamp"]) -\
                                    self.level_data.groupby("group").apply(lambda i: i.iloc[0]["TimeStamp"]))\
                                   .apply(lambda i: i.total_seconds())
        # Increase prior to drop
        level_groups["PriorIncrease"] = self.level_data.groupby("group")["Level"].max() -\
                                        self.level_data.groupby("group")["Level"].min().shift(1)
        # Time of increase prior to drop
        level_groups["PriorIncreaseTime"] = (self.level_data.groupby("group")\
                                                 .apply(lambda i: i.iloc[0]["TimeStamp"]) -\
                                             self.level_data.groupby("group")\
                                                 .apply(lambda i: i.iloc[-1]["TimeStamp"]).shift(1))\
                                            .apply(lambda i: i.total_seconds())
        level_groups["max_level"] = self.level_data.groupby("group")["Level"].max()


        # FLOW GROUPS
        # Total flow
        flow_groups["Flow"] = self.flow_data.groupby("group")["Flow"].sum() # Add total flow of peak
        # Calcuate difference in first and last measurement time in seconds
        flow_groups["TimeSpan"] = (self.flow_data.groupby("group").apply(lambda i: i.iloc[-1]["TimeStamp"]) -\
                                   self.flow_data.groupby("group").apply(lambda i: i.iloc[0]["TimeStamp"]))\
                                  .apply(lambda i: i.total_seconds())
        # Get id of level drop closest to flow peak in time
        flow_groups["level_group"] = flow_groups["TimeStamp"]\
                                     .apply(lambda i: (level_groups["TimeStamp"] - i).abs().idxmin())

        # Add columns from level to flow groups
        for i in ["Delta", "PriorIncrease", "PriorIncreaseTime", "max_level"]:
            var = level_groups.iloc[flow_groups["level_group"]][i]
            var.index = flow_groups.index
            flow_groups[i] = var

        # Calculate adjusted level change (adds prior increase per time times time of flow peak)
        flow_groups["AdjDelta"] = flow_groups["Delta"] - flow_groups["PriorIncrease"] / flow_groups["PriorIncreaseTime"]\
                                  * flow_groups["TimeSpan"]

        # Add data to class
        self.flow_groups = flow_groups
        self.level_groups = level_groups
