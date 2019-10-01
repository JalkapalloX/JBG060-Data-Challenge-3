import load_files as lf
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import utility
import wrangling


class measurement_analysis:
    """
    Versatile class useful for adding important columns,
    plotting basic properties of the data fast, and creating the DWAAS table.
    """
    def __init__(self, flow_data, level_data, rain_data,
                 min_dry_series=1, area_data=None, village_code=None, dry_threshold=0, max_interval=None):
        # CLEAN DATA
        flow_data = wrangling.clean_mes_data(flow_data)
        level_data = wrangling.clean_mes_data(level_data)

        # Check if rain_data is already summarized
        if not all(i in rain_data.columns for i in ['Date', 'Total', 'DrySeries']):
            rain_data = wrangling.summarize_rain_data(rain_data, area_data, village_code, dry_threshold)

        # Adding basic variables to the data
        flow_data["Date"] = flow_data["TimeStamp"].apply(lambda i: i.date())
        flow_data["Hour"] = flow_data["TimeStamp"].apply(lambda i: i.hour)
        flow_data["Month"] = flow_data["Date"].apply(lambda i: i.month)
        flow_data["Weekend"] = flow_data["Date"].apply(lambda i: int(i.weekday() >= 5))
        flow_data["TimeSpan"] = flow_data["TimeStamp"].diff(1).apply(lambda i: i.seconds).fillna(5)
        flow_data["Freq"] = 1 / flow_data["TimeSpan"]
        flow_data["Flow"] = flow_data["Value"] * flow_data["TimeSpan"] / 3600

        # Binary variable indicating whether flow peaked at a maximum or minimum
        flow_data["max"] = ((flow_data["Value"].diff(1) > 0) & (flow_data["Value"].diff(-1) > 0)).astype(int)
        flow_data["min"] = ((flow_data["Value"].diff(1) < 0) & (flow_data["Value"].diff(-1) < 0)).astype(int)

        # Adding basic variables to the data
        level_data["Date"] = level_data["TimeStamp"].apply(lambda i: i.date())
        level_data["Hour"] = level_data["TimeStamp"].apply(lambda i: i.hour)
        level_data["Month"] = level_data["Date"].apply(lambda i: i.month)
        level_data["Weekend"] = level_data["Date"].apply(lambda i: int(i.weekday() >= 5))
        level_data["TimeSpan"] = level_data["TimeStamp"].diff(1).apply(lambda i: i.seconds)
        level_data["Freq"] = 1 / level_data["TimeSpan"]
        level_data["Delta"] = level_data["Value"].diff(1)

        # Binary variable indicating whether level peaked at a maximum or minimum
        level_data["max"] = ((level_data["Value"].diff(1) > 0) & (level_data["Value"].diff(-1) > 0)).astype(int)
        level_data["min"] = ((level_data["Value"].diff(1) < 0) & (level_data["Value"].diff(-1) < 0)).astype(int)

        # Calculate area in square-kilometres
        self.area = area_data.loc[area_data["village_ID"] == village_code, "geometry"]\
                             .to_crs({"init": "epsg:3395"}).map(lambda p: p.area / 10**6).sum()

        # STORE DATA
        self.min_dry_series = min_dry_series
        self.area_data = area_data
        self.village_code = village_code
        self.dry_threshold = dry_threshold
        self.max_interval = max_interval

        self.flow_data = flow_data
        self.level_data = level_data
        self.rain_data = rain_data


    def plot(self):
        plt.figure(figsize=(14, 7))

        # Plots flow (Y) vs. index (X)
        plt.subplot(4, 2, 1)
        flow = self.flow_data["Value"].plot()
        # Plots level (Y) vs. index (X)
        plt.subplot(4, 2, 2)
        level = self.level_data["Value"].plot()

        # Plots flow (Y) vs. Hour of the day (X)
        plt.subplot(4, 2, 3)
        flow_by_time = self.flow_data.groupby("Hour")["Flow"].sum().plot()
        # Plots flow (Y) vs. Month (X)
        plt.subplot(4, 2, 5)
        flow_by_month = self.flow_data.groupby("Month")["Flow"].sum().plot()

        # Plots level (Y) vs. Hour of the day (X)
        plt.subplot(4, 2, 4)
        mean_level_by_hour = self.level_data.groupby("Hour")["Value"].mean().plot()
        # Plots level (Y) vs. Month (X)
        plt.subplot(4, 2, 6)
        mean_level_by_month = self.level_data.groupby("Month")["Value"].mean().plot()

        # Plots flow (Y) vs. rainfall (X)
        plt.subplot(4, 2, 7)
        rain_flow_merge = pd.merge(self.flow_data.groupby("Date")["Flow"].sum(),
                                   self.rain_data.set_index("Date"), left_index=True, right_index=True)
        rain_flow_merge = rain_flow_merge.loc[rain_flow_merge["Total"] <= 15]
        sns.regplot(y = rain_flow_merge["Flow"], x = rain_flow_merge["Total"], order=2)
        # Plots rainfall (Y) vs. date (X)
        plt.subplot(4, 2, 8)
        self.rain_data.set_index("Date")["Total"].plot()

        plt.tight_layout()
        plt.show()


    def DWAAS_HAAS(self):
        # CREATES THE DWAAS TABLE COMPARING THEORETICAL DWF AGAINST ACTUAL VALUES
        # Selects dates that are classified dry by function definition
        dry_dates = self.rain_data.loc[self.rain_data["DrySeries"] >= self.dry_threshold, "Date"]
        rainy_dates = self.rain_data.loc[self.rain_data["DrySeries"] == 0, "Date"]

        # Create binary column whether day is classified as dry
        self.flow_data["Dry"] = self.flow_data["TimeStamp"].apply(lambda i: i.date() in dry_dates.to_list()).astype(int)

        # Select only flow from dry days
        dry_flow = self.flow_data.loc[self.flow_data["Dry"] == 1]

        # Measure Names
        measure_names = ["Theoretical DWF (Q80)",
                         "Summer",
                         "Winter",
                         "Workday",
                         "Weekend",
                         "Average"]

        # Aggregates data for DWF measures
        measures = [dry_flow.groupby("Date")["Flow"].sum().quantile(0.2),
                    dry_flow.loc[dry_flow["Month"] <= 3].groupby("Date")["Flow"].sum().mean(),
                    dry_flow.loc[(dry_flow["Month"] >= 6) & (dry_flow["Month"] <= 9)].groupby("Date")["Flow"].sum().mean(),
                    dry_flow.loc[dry_flow["Weekend"] == 0].groupby("Date")["Flow"].sum().mean(),
                    dry_flow.loc[dry_flow["Weekend"] == 1].groupby("Date")["Flow"].sum().mean(),
                    dry_flow.groupby("Date")["Flow"].sum().mean()
                   ]

        # Computes relation to theoretical DWF
        relative_measures = [1,
                             measures[1] / measures[0],
                             measures[2] / measures[0],
                             measures[3] / measures[0],
                             measures[4] / measures[0],
                             measures[5] / measures[0]
                            ]

        # Sums up data in single data frame
        DWAAS_table1 = pd.DataFrame({"Name": measure_names, "Value": measures, "Rel. Value": relative_measures})

        return DWAAS_table1
