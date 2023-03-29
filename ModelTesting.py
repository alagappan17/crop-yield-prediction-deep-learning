from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

CNNStackedLSTM = load_model("CNNStackedLSTM.h5")
StackedLSTMCNN = load_model("StackedLSTMCNN.h5")
CNNBiLSTM = load_model("CNNBiLSTM.h5")
BiLSTMCNN = load_model("BiLSTMCNN.h5")

usr_data = ['Chengalpattu', 'Rice', '30.77', '1180', '2.68', '13.35', '3417.77', '43210',
            '58.18', '17817', '23.99', '13245', '17.83']
actualOutput = 171.96

# usr_data = ['Madurai', 'Sugarcane', '31', '1225.9', '3.91', '7.83', '14.81', '63394',
#             '52.57', '26168', '21.7', '31028', '25.73']
# actualOutput = 11975.69

# usr_data = ['Thanjavaur', 'Minor Pulses', '31.1', '1309.3', '2.9', '12.86', '165.6', '45217',
#             '50.48', '19991', '22.32', '24366', '27.2']
# actualOutput = 259.15

# usr_data = ['Tiruchirappalli', 'Groundnut', '30.95', '910.9', '2.86', '12.18', '72.27', '47740',
#             '49.57', '23070', '23.95', '25504', '26.48']
# actualOutput = 1419.54

dataMain = pd.read_excel(r'New_Dataset.xlsx')
data = dataMain.loc[(dataMain['District'] == usr_data[0])
                    & (dataMain['Crop'] == usr_data[1])]
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

result1 = CNNStackedLSTM.predict(user_x)
result1 = TargetVarScalerFit.inverse_transform(result1)

result2 = StackedLSTMCNN.predict(user_x)
result2 = TargetVarScalerFit.inverse_transform(result2)

result3 = CNNBiLSTM.predict(user_x)
result3 = TargetVarScalerFit.inverse_transform(result3)

result4 = BiLSTMCNN.predict(user_x)
result4 = TargetVarScalerFit.inverse_transform(result4)

print("CNN Stacked LSTM: ", result1)
print("Stacked LSTM CNN: ", result2)
print("CNN Bi LSTM: ", result3)
print("Bi LSTM CNN: ", result4)
print("Actual Output: ", actualOutput)
