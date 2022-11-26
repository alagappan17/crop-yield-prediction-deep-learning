import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np


def convertToInputFormat(usr_data):
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

    user_x = user_data_df.values
    user_x = PredictorScalerFit.transform(user_x)
    user_x = np.reshape(user_x, (user_x.shape[0], user_x.shape[1], 1))
    return user_x


def convertToOutputFormat(usr_data, predData):
    data = pd.read_excel(r'Final_Dataset.xlsx')
    data = data.loc[(data['District'] == usr_data[0])
                    & (data['Crop'] == usr_data[1])]
    target = ['Yield']
    y = pd.DataFrame(data[target].values)
    y = y.values

    sc1 = StandardScaler()
    TargetVarScalerFit = sc1.fit(y)
    y = TargetVarScalerFit.transform(y)

    predData = TargetVarScalerFit.inverse_transform(predData)
    return predData
