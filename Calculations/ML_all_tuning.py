#Editor: Logothetis Stavros-Andreas
#Libraries section
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
from keras.models import Sequential
from lightgbm import LGBMRegressor
from kerastuner import HyperModel
from pyearth import Earth
import kerastuner as kt
import pandas as pd
import numpy as np
import scipy.stats
import warnings
import glob
import time
import tqdm
import os        
#Machine Learning Models:

def LGBM_r(X_train, X_test, Y_train, Y_test, params):
    
    warnings.simplefilter(action='ignore', category=FutureWarning)

    estimator = LGBMRegressor()

    tuning = RandomizedSearchCV(estimator=estimator, param_distributions = params,
                              n_iter = 100, scoring='neg_mean_squared_error', 
                              cv = 10, verbose=10, random_state=1, n_jobs=30)   
    
    tuning.fit(X_train,Y_train.reshape(-1))

    best_est = tuning.best_estimator_

    Model_tuned = LGBMRegressor(n_estimators=best_est.n_estimators, max_depth=best_est.max_depth,
                                      learning_rate=best_est.learning_rate,
                                      num_leaves=best_est.num_leaves,
                                      reg_alpha=best_est.reg_alpha,
                                      subsample_freq=best_est.subsample_freq,
                                      reg_lambda=best_est.reg_lambda,
                                      min_split_gain=best_est.min_split_gain,
                                      subsample=best_est.subsample,
                                      colsample_bytree=best_est.colsample_bytree,
                                      random_state=1)
    
    Model_tuned.fit(X_train, Y_train.reshape(-1))
        
    return(Model_tuned, best_est)

def RF_r(X_train, X_test, Y_train, Y_test, params):

    estimator = RandomForestRegressor()
    
    tuning = RandomizedSearchCV(estimator=estimator, param_distributions = params,
                              n_iter = 100, scoring='neg_mean_squared_error', 
                              cv = 10, verbose=10, random_state=1, n_jobs=30, pre_dispatch = 6)   
    
    tuning.fit(X_train,Y_train.reshape(-1))

    best_est = tuning.best_estimator_

    Model_tuned = RandomForestRegressor(n_estimators=best_est.n_estimators, max_depth=best_est.max_depth,
                               min_samples_split= best_est.min_samples_split, min_samples_leaf=best_est.min_samples_leaf,
                               max_features=best_est.max_features, random_state=1)
    
    Model_tuned.fit(X_train, Y_train.reshape(-1))
        
    return(Model_tuned, best_est)

def MARS_r(X_train, X_test, Y_train, Y_test, params):
    
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # define the model
    model = Earth()
    
    tuning = RandomizedSearchCV(estimator=model, param_distributions = params,
                              n_iter = 100, scoring='neg_mean_squared_error', 
                              cv = 10, verbose=10, random_state=1, n_jobs=-1)  

    tuning.fit(X_train,Y_train.reshape(-1))
   
    best_est = tuning.best_estimator_
    
    Model_tuned = Earth(max_degree=best_est.max_degree,  max_terms=best_est.max_terms)
    
    Model_tuned.fit(X_train, Y_train.reshape(-1))
    
    model.fit(X_train, Y_train)
    
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
     
    Y_d_trained = scaler_y.inverse_transform(Y_train)
    Y_d_trained_predicted = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1))
    
    Y_d_tested = scaler_y.inverse_transform(Y_test)
    Y_d_tested_predicted = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1))
        
    X_d_trained = scaler_x.inverse_transform(X_train)
    X_d_tested = scaler_x.inverse_transform(X_test)
    
    Results = {"Descaled":[X_d_trained, X_d_tested, Y_d_trained, Y_d_tested, Y_d_trained_predicted, Y_d_tested_predicted],
               "Scaled":[X_train, X_test, Y_train, Y_test, y_train_pred, y_test_pred]}
               
    return(Results, best_est)

def KNN_r(X_train, X_test, Y_train, Y_test):

    df = pd.DataFrame()
    n_range = np.arange(1,31,1)

    for n in n_range:
        
        reg = KNeighborsRegressor(n_neighbors=n)
        reg.fit(X_train, Y_train)
        Y_test_predicted = reg.predict(X_test)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pd.DataFrame(Y_test)[0], pd.DataFrame(Y_test_predicted)[0])    
        MBE = sum(pd.DataFrame(Y_test_predicted)[0] - pd.DataFrame(Y_test)[0])/len(pd.DataFrame(Y_test_predicted)[0])
        RMSE = np.sqrt(mean_squared_error(Y_test, Y_test_predicted)) 
        df = df.append({"n": n, "Accuracy": reg.score(X_test, Y_test), "R^2" : r_value**2, "MBE": MBE, "RMSE": RMSE}, ignore_index=True)
        number = df['Accuracy'].argmax()
        x = pd.DataFrame(df.loc[number,:]).T.reset_index(drop=True)
   
    reg = KNeighborsRegressor(n_neighbors= int(x['n'][0]))
    reg.fit(X_train, Y_train)
    
    y_test_pred = reg.predict(X_test)
    y_train_pred = reg.predict(X_train)

    Y_d_trained = scaler_y.inverse_transform(Y_train)
    Y_d_trained_predicted = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1))
    
    Y_d_tested = scaler_y.inverse_transform(Y_test)
    Y_d_tested_predicted = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1))
        
    X_d_trained = scaler_x.inverse_transform(X_train)
    X_d_tested = scaler_x.inverse_transform(X_test)
    
    Results = {"Descaled":[X_d_trained, X_d_tested, Y_d_trained, Y_d_tested, Y_d_trained_predicted, Y_d_tested_predicted],
               "Scaled":[X_train, X_test, Y_train, Y_test, y_train_pred, y_test_pred]}
               
    return(Results, x['n'][0],  x['Accuracy'][0])

def Tuned_model(X_train, X_test, Y_train, Y_test, Model_tuned, ML):
    
    Model_tuned.fit(X_train, Y_train.reshape(-1))
    
    y_train_pred = Model_tuned.predict(X_train)
    y_test_pred = Model_tuned.predict(X_test)
    
    Y_d_trained = scaler_y.inverse_transform(Y_train)
    Y_d_trained_predicted = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1))

    Y_d_tested = scaler_y.inverse_transform(Y_test)
    Y_d_tested_predicted = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1))
    
    X_d_trained = scaler_x.inverse_transform(X_train)
    X_d_tested = scaler_x.inverse_transform(X_test)
    
    importance = Model_tuned.feature_importances_

    Results = {"Descaled":[X_d_trained, X_d_tested, Y_d_trained, Y_d_tested, Y_d_trained_predicted, Y_d_tested_predicted],
               "Scaled":[X_train, X_test, Y_train, Y_test, y_train_pred, y_test_pred]}
    
    return(Results, importance)

class RegressionHyperModel(HyperModel):
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def build(self, hp):
        model = Sequential()
        model.add(
            layers.Dense(
                units=hp.Int('units', 8, 64, 4, default=8),
                activation='relu',
                input_shape=input_shape))
            
        model.add(
            layers.Dense(
                units=hp.Int('units', 8, 32, 4, default=8),
                activation='relu'))

        model.add(layers.Dense(1, activation='linear'))
        
        model.compile(optimizer='adam',loss='mean_squared_error', metrics=['mean_squared_error','mean_absolute_error'])
        
        return model

def dataframes_creation(Results):
    
    train_val_min_d = pd.DataFrame(Results['Descaled'][2], columns = ['Train'])
    test_val_min_d = pd.DataFrame(Results['Descaled'][3], columns = ['Test'])
    train_pred_min_d = pd.DataFrame(Results['Descaled'][4], columns = ['Train_pred'])
    test_pred_min_d = pd.DataFrame(Results['Descaled'][5], columns = ['Test_pred'])
    x_min_val = pd.concat([test_val_min_d, test_pred_min_d], axis=1)
    x_min_train = pd.concat([train_val_min_d, train_pred_min_d], axis=1)
    
    return(x_min_val, x_min_train)

def statistics(data_observed, data_predictions):
   
    MBE = sum(data_predictions - data_observed)/len(data_predictions)
    nMBE = MBE/np.mean(data_observed)
    RMSE = np.sqrt(mean_squared_error(data_observed, data_predictions)) 
    nRMSE = RMSE/np.mean(data_observed)
    MAE= mean_absolute_error(data_observed, data_predictions)
    nMAE = MAE/np.mean(data_observed)

    R = scipy.stats.pearsonr(data_observed, data_predictions)[0]
    dif = data_predictions - data_observed
    suma = data_predictions + data_observed
    FB = (2/len(data_predictions))*sum((dif)/(suma))
    FGE = (2/len(data_predictions))*sum((np.abs(dif))/(np.abs(suma)))    

    return (MBE, nMBE, RMSE, nRMSE, MAE, nMAE, R, FB, FGE)

files = glob.glob("*.csv")

LGBM_params = {'n_estimators': [50,100,500,1000,1500,2000],
        'learning_rate': [0.3,0.2,0.1,0.05,0.01,0.001,0.0001],
        'max_depth': [2,5,10,12,15,20],
        'num_leaves': [2,5,10,12,15,20]}

RF_params = {'n_estimators': [100, 300, 500, 700, 900, 1000, 1500, 2000],
                   'max_features': [1,2,3],
                   'max_depth': [None, 10, 15, 20, 25],
                   'min_samples_split': [2,3,4,5,6],
                   'min_samples_leaf': [1,2,3,4,5,6]}

MARS_params = {'max_terms': [1,2,3], 'max_degree': [1,2,3]}

input_kt = ['kt_DNI_BSRN', 'kt_DNI_CAMS', 'kt_cams', 'kt_BSRN']

MLs = ['LGBM', 'RF', 'MARS', 'KNN', 'ANN']

for k in range(0, len(files)):
    
    Best_params_LGBM, Best_params_RF, Best_params_MARS, Best_params_KNN, Best_params_ANN = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    Statistics_df, Final_Parameters, df_train_all, df_test_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    for kt_ in tqdm.tqdm(input_kt):
        
        name = files[k]
        # station_name = name.split("\\")[1][:3].upper()
        station_name = name[:3].upper()

        print(station_name)
        
        df_min = pd.read_csv(files[k])
        df_min['kt_DNI_BSRN'] = df_min['DNI']/df_min['TOA_cams'] ; df_min = df_min[df_min['kt_DNI_BSRN'] > 0 ]
        df_min['kt_DNI_CAMS'] = df_min['Clear sky BNI_cams']/df_min['TOA_cams'] ; df_min = df_min[df_min['kt_DNI_CAMS'] > 0]
        df_min = df_min[df_min['kt_cams'] > 0 ] ; df_min = df_min[df_min['kt_BSRN'] > 0 ]
        
        df_min = df_min[df_min['AERONET AOD_550nm'] > 0] ; df_min = df_min[df_min['SZA'] <= 75] 
      
        df_min = df_min[['datetime','AERONET AOD_550nm', kt_, 'm','tcwv_cams']]
        df_min = df_min.reset_index(drop=True)
        time_v = df_min.iloc[:, 0].values ; time_v = time_v.reshape(-1, 1)

        X = df_min.iloc[:, 2:].values
        Y = df_min.iloc[:, 1].values ; Y = Y.reshape(-1, 1)
                  
        scaler_x = MinMaxScaler() ; scaler_y = MinMaxScaler()
        
        X_train, X_test, Y_train, Y_test = X[:int(round(0.7*len(X),1))], X[int(round(0.7*len(X),1)):], Y[:int(round(0.7*len(Y),1))], Y[int(round(0.7*len(Y),1)):]
    
        scaler_x.fit(X_train) ; scaler_y.fit(Y_train)
 
        X_scaled_train = scaler_x.transform(X_train) ; Y_scaled_train = scaler_y.transform(Y_train)     
        X_scaled_test = scaler_x.transform(X_test) ; Y_scaled_test = scaler_y.transform(Y_test)
        
        #splitting
        X_train, X_test, Y_train, Y_test = X_scaled_train, X_scaled_test, Y_scaled_train, Y_scaled_test
        time_train, time_test = time_v[:int(round(0.7*len(time_v),1))], time_v[int(round(0.7*len(time_v),1)):]

        for ML in MLs: 
            
            print('kt: ', kt_, " ML: ", ML)
            
            if ML == 'LGBM':
                
                start_time = time.time()
                Model_tuned, Best_params = LGBM_r(X_train, X_test, Y_train, Y_test, LGBM_params)
                Results, importance = Tuned_model(X_train, X_test, Y_train, Y_test, Model_tuned, ML)
                importance = (Model_tuned.feature_importances_ / sum(Model_tuned.feature_importances_)) * 100
                Data_test, Data_train = dataframes_creation(Results)
                MBE_test, nMBE_test, RMSE_test, nRMSE_test, MAE_test, nMAE_test, R_test, FB_test, FGE_test = statistics(Data_test['Test'], Data_test['Test_pred'])
                MBE_train, nMBE_train, RMSE_train, nRMSE_train, MAE_train, nMAE_train, R_train, FB_train, FGE_train = statistics(Data_train['Train'], Data_train['Train_pred'])

                exec_time = time.time() - start_time
                
                Best_params_LGBM = Best_params_LGBM.append({"Station":station_name, "ML":ML, "kt":kt_, "learning_rate":Best_params.learning_rate,"max_depth":Best_params.max_depth,
                                    "n_estimators":Best_params.n_estimators, "num_leaves":Best_params.num_leaves,
                                    "reg_alpha":Best_params.reg_alpha, "subsample":Best_params.subsample, "colsample_bytree":Best_params.colsample_bytree,
                                    "reg_lambda":Best_params.reg_lambda, "min_split_gain":Best_params.min_split_gain, "subsample_freq":Best_params.subsample_freq,
                                    "importance_kt": importance[0], "importance_m":importance[1],"importance_water":importance[2], "time":exec_time},ignore_index=True)   
                              
            elif ML == 'RF':
                
                start_time = time.time()
                Model_tuned, Best_params = RF_r(X_train, X_test, Y_train, Y_test, RF_params)
                Results, importance = Tuned_model(X_train, X_test, Y_train, Y_test, Model_tuned, ML)
                Data_test, Data_train = dataframes_creation(Results)
                MBE_test, nMBE_test, RMSE_test, nRMSE_test, MAE_test, nMAE_test, R_test, FB_test, FGE_test = statistics(Data_test['Test'], Data_test['Test_pred'])
                MBE_train, nMBE_train, RMSE_train, nRMSE_train, MAE_train, nMAE_train, R_train, FB_train, FGE_train = statistics(Data_train['Train'], Data_train['Train_pred'])

                exec_time = time.time() - start_time
                
                Best_params_RF = Best_params_RF.append({"Station":station_name, "ML":ML, "kt":kt_, "min_samples_split":Best_params.min_samples_split, "max_depth":Best_params.max_depth,
                                    "max_features":Best_params.max_features, "min_samples_leaf":Best_params.min_samples_leaf, "n_estimators_min":Best_params.n_estimators,
                                    "importance_kt": importance[0], "importance_m":importance[1],"importance_water":importance[2], "time":exec_time},ignore_index=True) 
                            
            elif ML == 'MARS':
                
                start_time = time.time()
                Results, Best_params = MARS_r(X_train, X_test, Y_train, Y_test, MARS_params)
                Data_test, Data_train = dataframes_creation(Results)
                MBE_test, nMBE_test, RMSE_test, nRMSE_test, MAE_test, nMAE_test, R_test, FB_test, FGE_test = statistics(Data_test['Test'], Data_test['Test_pred'])
                MBE_train, nMBE_train, RMSE_train, nRMSE_train, MAE_train, nMAE_train, R_train, FB_train, FGE_train = statistics(Data_train['Train'], Data_train['Train_pred'])

                exec_time = time.time() - start_time
                
                Best_params_MARS = Best_params_MARS.append({"Station":station_name, "ML":ML,"kt":kt_, "max_degree":Best_params.max_degree,  "max_terms":Best_params.max_terms, "time":exec_time},ignore_index=True) 

                
            elif ML == 'KNN':
                
                start_time = time.time()
                Results, n,  Accuracy = KNN_r(X_train, X_test, Y_train, Y_test)
                Data_test, Data_train = dataframes_creation(Results)
                MBE_test, nMBE_test, RMSE_test, nRMSE_test, MAE_test, nMAE_test, R_test, FB_test, FGE_test = statistics(Data_test['Test'], Data_test['Test_pred'])
                MBE_train, nMBE_train, RMSE_train, nRMSE_train, MAE_train, nMAE_train, R_train, FB_train, FGE_train = statistics(Data_train['Train'], Data_train['Train_pred'])

                exec_time = time.time() - start_time
                
                Best_params_KNN = Best_params_KNN.append({"Station":station_name, "ML":ML,"kt":kt_, "n":n, "Accuracy":Accuracy, "time":exec_time},ignore_index=True) 

            elif ML == 'ANN':
                
                start_time = time.time()
                input_shape = (3,)
                hypermodel = RegressionHyperModel(input_shape)   
     
                tuner_rs = kt.RandomSearch(
                            hypermodel,
                            objective='mean_squared_error',
                            seed=1,
                            max_trials=50,
                            executions_per_trial=5,directory=os.path.normpath('C:/'), overwrite=True)
                
                exec_time = time.time() - start_time
               
                tuner_rs.search(X_train, Y_train, batch_size=32, epochs=50, validation_split=0.2, verbose=100)
                
                model = tuner_rs.get_best_models(num_models=1)[0]
               
                layers_list = []
                for layer in model.layers:
                  layers_list.append(layer.get_output_at(0).get_shape().as_list()[1])
              
                y_test_pred = model.predict(X_test)
                y_train_pred = model.predict(X_train)
                 
                Y_d_trained = scaler_y.inverse_transform(Y_train)
                Y_d_trained_predicted = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1))
                
                Y_d_tested = scaler_y.inverse_transform(Y_test)
                Y_d_tested_predicted = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1))
                    
                X_d_trained = scaler_x.inverse_transform(X_train)
                X_d_tested = scaler_x.inverse_transform(X_test)
               
                Best_params_ANN = Best_params_ANN.append({"Station":station_name, "ML":ML,"kt":kt_, "Layer_1":layers_list[0],
                                                          "Layer_2":layers_list[1], "Layer_3":layers_list[2], "time":exec_time},
                                                         ignore_index=True) 
        
                Results = {"Descaled": [X_d_trained, X_d_tested, Y_d_trained, Y_d_tested, Y_d_trained_predicted, Y_d_tested_predicted],
                               "Scaled": [X_train, X_test, Y_train, Y_test, y_train_pred, y_test_pred]}
                            
                Data_test, Data_train = dataframes_creation(Results)
                MBE_test, nMBE_test, RMSE_test, nRMSE_test, MAE_test, nMAE_test, R_test, FB_test, FGE_test = statistics(Data_test['Test'], Data_test['Test_pred'])
                MBE_train, nMBE_train, RMSE_train, nRMSE_train, MAE_train, nMAE_train, R_train, FB_train, FGE_train = statistics(Data_train['Train'], Data_train['Train_pred'])
            
            Data_test['datetime'], Data_test['ML'], Data_test['kt'] = pd.DataFrame(time_test), ML, kt_
            Data_train['datetime'], Data_train['ML'], Data_train['kt'] = pd.DataFrame(time_train), ML, kt_
           
            df_train_all, df_test_all = df_train_all.append(Data_train), df_test_all.append(Data_test)
           
            Statistics_df = Statistics_df.append({"Station":station_name, "ML":ML, "kt":kt_, "MBE_test":MBE_test,"nMBE_test":nMBE_test, "RMSE_test":RMSE_test,
                                                "nRMSE_test":nRMSE_test, "MAE_test": MAE_test, "nMAE_test":nMAE_test, "R_test":R_test,
                                                "FB_test":FB_test, "FGE_test":FGE_test,
                                                "MBE_train":MBE_train,"nMBE_train":nMBE_train, "RMSE_train":RMSE_train,
                                                "nRMSE_train":nRMSE_train, "MAE_train": MAE_train, "nMAE_train":nMAE_train, "R_train":R_train,
                                                "FB_train":FB_train, "FGE_train":FGE_train},ignore_index=True)
       
    Best_params_LGBM.to_excel("Best_params/LGBM_" + station_name + '_best_params_c' + '.xlsx', header=1, index=0)     
    Best_params_RF.to_excel("Best_params/RF_" + station_name + '_best_params_c' + '.xlsx', header=1, index=0)
    Best_params_MARS.to_excel("Best_params/MARS_" + station_name + '_best_params_c' + '.xlsx', header=1, index=0)
    Best_params_KNN.to_excel("Best_params/KNN_" + station_name + '_best_params_c' + '.xlsx', header=1, index=0)
    Best_params_ANN.to_excel("Best_params/ANN_" + station_name + '_best_params_c' + '.xlsx', header=1, index=0)   
   
    Statistics_df.to_excel("Statistics/" + station_name + '_statistics_c_new.xlsx', header=1, index=0)
   
    df_train_all.to_csv('Datasets/' + station_name + '_train_c_new.csv', header=1, index=0)
    df_test_all.to_csv('Datasets/' + station_name + '_test_c_new.csv', header=1, index=0)