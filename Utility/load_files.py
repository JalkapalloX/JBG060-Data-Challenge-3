import pandas as pd
import numpy as np
import datetime
import geopandas as gpd
import os


def get_measurements(path, convert_time=False):
    """
    Will read all measurement data from given path and store them in separate dataframes.
    ~~~ EXAMPLE CALL ~~~
    flow_data, level_data = get_measurements("C:/mypath/RG8150")
    ~~~~~~~~~~~~~~~~~~~~
    """
    files = os.listdir(path)
    
    data = [pd.read_csv(path + "/" + i, sep = ";") for i in files]
    data =  pd.concat(data, sort = False, ignore_index = True)
    
    data["RG_ID"] = data["Tagname"].str.slice(9,13).astype(int)
    data["Value"] = data["Value"].str.replace(",", ".").astype(float)
    data["DataQuality"] = (data["DataQuality"] == "Good").astype(int)
    if convert_time == True:
        data["TimeStamp"] = pd.to_datetime(data["TimeStamp"])
        
    data = data[["Tagname", "RG_ID", "TimeStamp", "Value", "DataQuality"]]
    
    flow_data = data[data["Tagname"].str.contains("Debietmeting")].reset_index(drop = True)
    level_data = data[data["Tagname"].str.contains("Niveaumeting")].reset_index(drop = True)
    
    flow_data.drop("Tagname", axis=1, inplace=True)
    level_data.drop("Tagname", axis=1, inplace=True)
    
    return flow_data, level_data


def get_rain_prediction(path, from_date=None, to_date=None):
    """
    Will read rain prediction data + dates from file names from given path and store those
    in separate dataframes.
    ~~~ EXAMPLE CALL ~~~
    pred_dates, pred_data = get_rain_prediction("C:/mypath/knmi....")
    ~~~~~~~~~~~~~~~~~~~~
    """
    files = os.listdir(path)
    
    dates = pd.Series(pd.to_datetime([i.split("_")[3] for i in files]))
    
    if (from_date is not None) & (to_date is not None):
        boolean_ = (dates >= pd.to_datetime(from_date)) & (dates < pd.to_datetime(to_date))
        files = pd.Series(files)[boolean_]
    
    pred_date = pd.Series(pd.to_datetime([i.split("_")[2] for i in files]))
    start_date = pd.Series(pd.to_datetime([i.split("_")[3] for i in files]))
    end_date = pd.Series(pd.to_datetime([i.split("_")[4][:20] for i in files]))
    
    data = np.array([np.loadtxt(path + "/" + i, skiprows=7) for i in files if ".aux" not in i])
    
    date_data = pd.concat([pred_date, start_date, end_date], axis=1)
    date_data.columns = ["pred", "start", "end"]
    date_data.drop_duplicates(inplace=True)
    
    return date_data, data


def get_rain(path, convert_time=False):
    """
    Will read all rain data from given path and store them in a single dataframe.
    ~~~ EXAMPLE CALL ~~~
    rain_data = get_rain("C:/mypath/rain_timeseries")
    ~~~~~~~~~~~~~~~~~~~~
    """
    
    files = os.listdir(path)
    
    data = [pd.read_csv(path + "/" + i, skiprows=2) for i in files]
    data =  pd.concat(data, sort = False, ignore_index = True)
    if convert_time == True:
        data["Begin"] = pd.to_datetime(data["Begin"])
        data["Eind"] = pd.to_datetime(data["Eind"])
    
    data.rename({"Begin": "Start", "Eind": "End"}, axis=1, inplace = True)
    
    return data


def get_system_register(path):
    """
    Will read system information data from given path and store it in a single dataframe.
    ~~~ EXAMPLE CALL ~~~
    system_data = get_system_register("C:/mypath/sewer_model")
    ~~~~~~~~~~~~~~~~~~~~
    (!) This is almost the same data as the Rioleringsdeelgebied.shp file provides and it
    is recommended to use this one instead, see the class sdf.
    """
    
    system_data = pd.read_excel(path + "/" + "20180717_dump riodat rioleringsdeelgebieden.xlsx", skiprows=9)
    system_data = system_data[["Volgnr", "Code", "Naam / lokatie", "RWZI"]]
    system_data.columns = ["area_ID", "sewer_system", "area_name", "RWZI"]
    
    return system_data


class sdf:
    """
    Will read all shp files from given path and store them within this class as data frames.
    ~~~ EXAMPLE CALL ~~~
    data = sdf("C:/mypath/aa-en-maas_sewer_shp")
    data.area_data
    ~~~~~~~~~~~~~~~~~~~~
    """
    def __init__(self, path):
        # Sewage area data
        area_data = gpd.read_file(path + "/" + "Rioleringsdeelgebied.shp")
        area_data["area"] = area_data.area
        area_data = area_data[["RGDIDENT", "NAAMRGD", "RGDID", "area", "geometry"]]
        area_data.columns = ["sewer_system", "area_name", "area_ID", "area", "geometry"]
        
        # RG data
        RG_data = gpd.read_file(path + "/" + "Rioolgemaal.shp")
        RG_data = RG_data[["ZRE_ID", "ZREIDENT", "ZRW_ZRW_ID", "ZRGCAPA1",
                               "ZRE_ZRE_ID", "ZRGRGCAP",
                               "ZRGGANGL", "geometry"]]
        RG_data.columns = ["unit_ID", "RG_ID", "RWZI_ID", "min_capacity", "to_unit_ID", "max_capacity",
                           "RG_name", "geometry"]
        
        # RWZI regions
        RWZI_regions = gpd.read_file(path + "/" + "Zuiveringsregio.shp")
        RWZI_regions = RWZI_regions[["GAGNAAM", "geometry"]]
        RWZI_regions.columns = ["RWZI_name", "geometry"]
        
        # RWZI data
        RWZI_data = gpd.read_file(path + "/" + "RWZI.shp")
        RWZI_data = RWZI_data[["ZRW_ID", "ZRWIDENT", "ZRWNAAM", "geometry"]]
        RWZI_data.columns = ["RWZI_ID", "RWZI_identifier", "RWZI_name", "geometry"]
        
        # Pipe data
        pipe_data = gpd.read_file(path + "/" + "Leidingtrace.shp")
        pipe_data = pipe_data[["LDG_ID", "IDENTIFICA", "TRACE_NAAM", "STATUS", "geometry"]]
        pipe_data.columns = ["LDG_ID", "LD_identifier", "LD_name", "status", "geometry"]
        
        
        # Store data in class
        self.area_data = area_data
        self.RG_data = RG_data
        self.RWZI_regions = RWZI_regions
        self.RWZI_data = RWZI_data
        self.pipe_data = pipe_data


def create_sql_db(path=None, data_path=None,
              measurement_path=None, rain_path=None, rain_pred_path=None, shp_path=None,
              pumps="all"):
        """
        Function for generating an SQLite database consisting of measurement data
        and rain data. Other data sources are not integrated as their format is not
        supported by SQLite.

        ~~~~~ INPUT ~~~~~
        path      :   Path to directory where database should be created.
                      If no information is provided, the current directory will be chosen.
        data_path :   Directory in shape of the original .zip
                      If folder does not have the correct structure, an error will occur.
        ..._path  :   Directories for specific subsets of the data. Not needed unless
                      folder does not have the same structure as original .zip.
        
        ~~~~~ DB STRUCTURE ~~~~~
        "flow"    :   Flow data.
        "level"   :   Level data.
        "rain"    :   Rain data.

        """
        
        # Create connection with database
        if path is None:
            path = os.getcwd()
        conn = sqlite3.connect(path + "/" + "sewer_data.db")
    
        # MEASUREMENT DATA
        # Finding all pump names to be scraped
        if data_path is not None:
            if pumps == "all":
                files = os.listdir(data_path + "/sewer_data/data_pump")
                folder_files = ["." not in i for i in files]
                files = np.array(files)[folder_files]
            else:
                files = pumps
            
            # Loading pump data into sql
            for i in files:
                flow_data, level_data = get_measurements(data_path + "/sewer_data/data_pump" + "/" + i + "/" + i)
                flow_data.to_sql("flow", conn, if_exists="append", index=False)
                level_data.to_sql("level", conn, if_exists="append", index=False)
            
            # RAIN DATA
            rain_data = get_rain(data_path + "/sewer_data/rain_timeseries")
            rain_data.to_sql("rain", conn, if_exists="replace", index=False)
        
        else:
            # MEASUREMENT DATA
            if measurement_path is not None:
                if pumps == "all":
                    files = os.listdir(measurement_path + "/sewer_data/data_pump")
                    folder_files = ["." not in i for i in files]
                    files = np.array(files)[folder_files]
                else:
                    files = pumps
                    
                # Loading pump data into sql
                for i in files:
                    flow_data, level_data = get_measurements(measurement_path + "/sewer_data/data_pump" + "/" + i + "/" + i)
                    flow_data.to_sql("flow", conn, if_exists="append", index=False)
                    level_data.to_sql("level", conn, if_exists="append", index=False)
            
            # RAIN DATA
            if rain_path is not None:
                rain_data = get_rain(rain_path + "/sewer_data/rain_timeseries")
                rain_data.to_sql("rain", conn, if_exists="replace", index=False)
