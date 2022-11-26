import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Bidirectional

data = pd.read_excel(r'Final_Dataset.xlsx')
predictors = ['District', 'Crop', 'Average_Temperature', 'Precipitation', 'Sea_Level_Pressure', 'Wind', 'Area', 'Nitrogen_Consumption',
              'Nitrogen_Share_in_NPK', 'Phosphate_Consumption', 'Phosphate_Share_in_NPK', 'Potash_Consumption', 'Potash_Share_in_NPK']
target = ['Yield']
y = pd.DataFrame(data[target].values)
X = pd.DataFrame(data[predictors].values)
labelencoder_X_1 = LabelEncoder()
X.loc[:, 0] = labelencoder_X_1.fit_transform(X.iloc[:, 0])
X.loc[:, 1] = labelencoder_X_1.fit_transform(X.iloc[:, 1])
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
model = Sequential()
model.add(Conv1D(32, 2, activation="relu",
                 input_shape=(X_train.shape[1], 1)))
model.add(Bidirectional(
    LSTM(100, activation='relu', input_shape=(100, 1))))  # elu
model.add(Dense(1))
model.compile(loss='mean_squared_error',
                   optimizer='adam', metrics=['mae', 'mse'])
history = model.fit(X_train, y_train, validation_data=(
    X_test, y_test), epochs=200, batch_size=1)
pred = model.predict(X_test)
print("y_test: ", y_test)
print("y_pred", pred)

MSE = mean_squared_error(y_test, pred)
MAE = mean_absolute_error(y_test, pred)
RMSE = math.sqrt(MSE)
R2 = r2_score(y_test, pred)
print("\n\n\n\nMean Square Error CNN Bi LSTM:\n")
print(MSE)
print("\nRoot Mean Square Error CNN Bi LSTM:\n")
print(RMSE)
print("\nMean Absolute Error CNN Bi LSTM:\n")
print(MAE)
print("\nR Squared Error CNN Bi LSTM:\n")
print(R2)

plt.plot(history.history['mse'])  # tb
plt.plot(history.history['val_mse'])
plt.title('Mean Squared Error - CNN Bi LSTM')
plt.ylabel('MSE')
plt.xlabel('epochs')
plt.legend(['Training', 'Testing'], loc='upper left')
plt.show()

plt.plot(history.history['mae'])  # tb
plt.plot(history.history['val_mae'])
plt.title('Mean Absolute Error - CNN Bi LSTM')
plt.ylabel('MAE')
plt.xlabel('epochs')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

model.save("CNNBiLSTM.h5")
