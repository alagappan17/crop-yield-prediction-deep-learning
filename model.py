from tkinter import *
from tkinter import ttk
from csv import writer
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D


def runModel(userData):
    usr_data = userData
    print("Params: ", userData)
    data = pd.read_excel(r'Final_Dataset.xlsx')
    data = data.loc[(data['District'] == usr_data[0])
                    & (data['Crop'] == usr_data[1])]
    predictors = ['District', 'Crop', 'Average_Temperature', 'Precipitation', 'Sea_Level_Pressure', 'Wind', 'Area', 'Nitrogen_Consumption',
                  'Nitrogen_Share_in_NPK', 'Phosphate_Consumption', 'Phosphate_Share_in_NPK', 'Potash_Consumption', 'Potash_Share_in_NPK']
    target = ['Yield']
    y = pd.DataFrame(data[target].values)
    X = pd.DataFrame(data[predictors].values)
    X.loc[len(data.index)] = usr_data

    labelencoder_X_1 = LabelEncoder()
    X.loc[:, 0] = labelencoder_X_1.fit_transform(X.iloc[:, 0])
    X.loc[:, 1] = labelencoder_X_1.fit_transform(X.iloc[:, 1])
    user_data_df = X.iloc[[len(X.index)-1]]
    X = X.drop(len(X.index)-1)
    X = X.values
    y = y.values

    sc = StandardScaler()
    sc1 = StandardScaler()
    PredictorScalerFit = sc.fit(X)
    TargetVarScalerFit = sc1.fit(y)
    X = PredictorScalerFit.transform(X)
    y = TargetVarScalerFit.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    user_x = user_data_df.values
    user_x = PredictorScalerFit.transform(user_x)
    user_x = np.reshape(user_x, (user_x.shape[0], user_x.shape[1], 1))
    regressor = Sequential()
    regressor.add(Conv1D(32, 2, activation="relu",
                         input_shape=(X_train.shape[1], 1)))
    regressor.add(LSTM(units=50, return_sequences=True,
                       input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))
    regressor.compile(loss='mean_squared_error',
                      optimizer='adam', metrics=['mae', 'mse'])
    history = regressor.fit(X_train, y_train, validation_data=(
        X_test, y_test), epochs=60, batch_size=1)
    y_pred = regressor.predict(user_x)
    y_pred = TargetVarScalerFit.inverse_transform(y_pred)
    result = int(y_pred[0])
    print(y_pred)
    return y_pred
