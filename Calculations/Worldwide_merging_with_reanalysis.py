import datetime as dt
import pandas as pd
import numpy as np
import glob
import tqdm
import os 

directory = os.getcwd()

filenames_tuned = glob.glob('E:/ML_PROJECT_new_analysis/Validation Data/Final Results/Worldwide Stations/Datasets/ML/*.csv')
filenames_merra = glob.glob('E:/ML_PROJECT_new_analysis/Validation Data/Final Results/Worldwide Stations/Datasets/MERRA-2/*.csv')
filenames_cams = glob.glob('E:/ML_PROJECT_new_analysis/Validation Data/Final Results/Worldwide Stations/Datasets/CAMS/*.csv')

MLs = ['LGBM', 'RF', 'MARS', 'KNN', 'ANN']
kts = ['kt_DNI_BSRN', 'kt_DNI_CAMS', 'kt_cams', 'kt_BSRN']

for station in filenames_tuned:
   
    upd_st_merra, upd_st_cams = [], []

    name = station.split('\\')[1][:-9]
    print(name)
    
    ML = filenames_tuned[[idx for idx, s in enumerate(filenames_tuned) if name in s][0]]
    merra = filenames_merra[[idx for idx, s in enumerate(filenames_merra) if name in s][0]]
    cams = filenames_cams[[idx for idx, s in enumerate(filenames_cams) if name in s][0]]
      
    df_ML_station = pd.read_csv(ML, parse_dates=True, index_col='datetime')    
    df_ML_station.index = df_ML_station.index.rename('datetime')
    df_ML_station = df_ML_station.tz_localize(tz='UTC')
        
    df_merra_station = pd.read_csv(merra, parse_dates=True, index_col='time')   
    df_merra_station.index = df_merra_station.index.rename('datetime')
    df_merra_station = df_merra_station.tz_localize(tz='UTC')
    
    df_merra_station = df_merra_station[str(df_ML_station.index[0])[:10]:str(df_ML_station.index[-1])[:10]]
    
    df_cams_station = pd.read_csv(cams, parse_dates=True, index_col='time')    
    df_cams_station.index = df_cams_station.index.rename('datetime')
    df_cams_station = df_cams_station.tz_localize(tz='UTC')
    df_cams_station = df_cams_station[str(df_ML_station.index[0])[:10]:str(df_ML_station.index[-1])[:10]]
   
    for merra_2_index in tqdm.tqdm(df_merra_station.index):

        start_date = merra_2_index - dt.timedelta(minutes=30)
        end_date = merra_2_index + dt.timedelta(minutes=30)
        df = df_ML_station[(df_ML_station.index >= start_date) & (df_ML_station.index <= end_date)]
      
        if len(df) > 0: 
          
            df_ML_station_ML_kt = df

            upd_st_merra.append({"datetime":merra_2_index, "AOD_tested_AERONET":np.mean(df_ML_station_ML_kt['AOD_tested_AERONET']),
                 "AOD_tested_ML":np.mean(df_ML_station_ML_kt['AOD_tested_LGBM']), "AOD_tested_MERRA": df_merra_station.loc[merra_2_index, "merra_AOD_cor"],
                 "N_values": len(df_ML_station_ML_kt)})     
 
    for cams_index in tqdm.tqdm(df_cams_station.index):

        start_date = cams_index - dt.timedelta(minutes=90)
        end_date = cams_index + dt.timedelta(minutes=90)
        df = df_ML_station[(df_ML_station.index >= start_date) & (df_ML_station.index <= end_date)]
        
        if len(df) > 0:
  
            df_ML_station_ML_kt = df

            upd_st_cams.append({"datetime":cams_index, "AOD_tested_AERONET":np.mean(df_ML_station_ML_kt['AOD_tested_AERONET']),
                                 "AOD_tested_ML":np.mean(df_ML_station_ML_kt['AOD_tested_LGBM']),
                                 "AOD_tested_CAMS": df_cams_station.loc[cams_index,"CAMS_AOD_cor"],
                                 "N_values": len(df_ML_station_ML_kt)})
               
    df_station_merra = pd.DataFrame(upd_st_merra)
    df_station_cams = pd.DataFrame(upd_st_cams)
        
    df_station_merra.to_csv('E:/ML_PROJECT_new_analysis/Validation Data/Final Results/Worldwide Stations/Datasets/MERRA-2/Merged_with_ML/' + name + "_merra2.csv", index=0)
    df_station_cams.to_csv('E:/ML_PROJECT_new_analysis/Validation Data/Final Results/Worldwide Stations/Datasets/CAMS/Merged_with_ML/' + name + "_cams.csv", index=0)
