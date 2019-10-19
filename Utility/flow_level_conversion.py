import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import wrangling
import utility

class mes_converter:
    def __init__(self, flow_data, level_data):
        # BASIC DATA CLEANING
        flow_data = wrangling.clean_mes_data(flow_data)
        level_data = wrangling.clean_mes_data(level_data)

        # MATCH TIMESTAMPS OF LEVEL AND FLOW DATA
        flow_data, level_data = wrangling.merge_flow_level(flow_data, level_data)

        # INTERPOLATE MEASUREMENTS
        flow_data = wrangling.fill_flow(flow_data)
        level_data = wrangling.fill_level(level_data)

        # RENAME VALUE COLUMNS
        flow_data.rename(columns={"Value": "Flow/s"}, inplace=True)
        level_data.rename(columns={"Value": "Level"}, inplace=True)
        flow_data["Flow/s"] = flow_data["Flow/s"] / 3600

        # BASIC DATA WRANGLING
        flow_data["TimeSpan"] = flow_data["TimeStamp"].diff(1).apply(lambda i: i.seconds)
        level_data["TimeSpan"] = level_data["TimeStamp"].diff(1).apply(lambda i: i.seconds)

        flow_data["Flow"] = flow_data["Flow/s"] * flow_data["TimeSpan"]
        level_data["Delta/s"] = level_data["Level"].diff(1) / level_data["TimeSpan"]

        # ADD DATA TO CLASS
        self.flow_data = flow_data
        self.level_data = level_data


    def to_dry_data(self, rain_data, area_data, min_dry_series=1, village_code=None, dry_threshold=1):
        self.rain_data = rain_data
        self.area_data = area_data

        # CALCULATE WHICH DAYS ARE DRY
        dry_days = wrangling.summarize_rain_data(self.rain_data, self.area_data,
                                                 village_code=village_code,
                                                 dry_threshold=dry_threshold)

        # GET LIST OF DRY DATES
        dry_days = dry_days.loc[dry_days["DrySeries"] >= min_dry_series, "Date"].reset_index(drop=True)

        # SELECT ALL DRY DAYS
        self.flow_data = self.flow_data.loc[self.flow_data["TimeStamp"]\
                                       .apply(lambda i: i.date() in dry_days.to_list()),:]\
                                       .reset_index(drop=True)
        self.level_data = self.level_data.loc[self.level_data["TimeStamp"]\
                                         .apply(lambda i: i.date() in dry_days.to_list()),:]\
                                         .reset_index(drop=True)


    def add_groups(self):
        # GROUP FLOW AND LEVEL DATA INTO PHASES
        self.flow_data["group"] = wrangling.flow_group(self.flow_data["Flow/s"])
        self.level_data["group"] = wrangling.level_group(self.level_data["Level"])

        # INITIALIZE GROUP DATA FRAMES
        flow_groups = self.flow_data.sort_values("Flow/s", ascending=False)\
                                    .groupby("group")\
                                    .apply(lambda i: i.iloc[0])[["TimeStamp", "group"]]
        level_groups = self.level_data.sort_values("TimeStamp")\
                                      .groupby("group")\
                                      .apply(lambda i: i.iloc[0])[["TimeStamp", "group"]]

        # ADD VARIABLES
        # LEVEL GROUPS
        level_groups["Delta"] = self.level_data.groupby("group")["Level"].min() -\
                                self.level_data.groupby("group")["Level"].max()
        level_groups["TimeSpan"] = (self.level_data.groupby("group").apply(lambda i: i.iloc[-1]["TimeStamp"]) -\
                                    self.level_data.groupby("group").apply(lambda i: i.iloc[0]["TimeStamp"]))\
                                   .apply(lambda i: i.total_seconds())
        level_groups["Delta/s"] = level_groups["Delta"] / level_groups["TimeSpan"]
        level_groups["PriorIncrease"] = self.level_data.groupby("group")["Level"].max() -\
                                        self.level_data.groupby("group")["Level"].min().shift(1)

        level_groups["PriorIncreaseTime"] = (self.level_data.groupby("group")\
                                                 .apply(lambda i: i.iloc[0]["TimeStamp"]) -\
                                             self.level_data.groupby("group")\
                                                 .apply(lambda i: i.iloc[-1]["TimeStamp"]).shift(1))\
                                            .apply(lambda i: i.total_seconds())


        # FLOW GROUPS
        flow_groups["Flow"] = self.flow_data.groupby("group")["Flow"].sum() # Add total flow of peak
        flow_groups["peak_flow"] = self.flow_data.groupby("group")["Flow"].max()
        flow_groups["TimeSpan"] = (self.flow_data.groupby("group").apply(lambda i: i.iloc[-1]["TimeStamp"]) -\
                                   self.flow_data.groupby("group").apply(lambda i: i.iloc[0]["TimeStamp"]))\
                                  .apply(lambda i: i.total_seconds())
        flow_groups = flow_groups.loc[flow_groups["Flow"] > 1]              # Drop missclassified peaks
        flow_groups["level_group"] = flow_groups["TimeStamp"]\
                                    .apply(lambda i: (level_groups["TimeStamp"] - i).abs().idxmin())
                                                                            # ID of level drop

        # ADD COLUMNS FROM LEVEL GROUPS TO FLOW GROUPS
        for i in ["Delta", "Delta/s", "PriorIncrease", "PriorIncreaseTime"]:
            var = level_groups.iloc[flow_groups["level_group"]][i]
            var.index = flow_groups.index
            flow_groups[i] = var

        flow_groups["AdjDelta"] = flow_groups["Delta"] - flow_groups["PriorIncrease"] / flow_groups["PriorIncreaseTime"]\
                                  * flow_groups["TimeSpan"]

        # ADD DATA TO CLASS
        self.flow_groups = flow_groups
        self.level_groups = level_groups
