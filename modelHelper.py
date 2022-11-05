# import numpy as np
# from keras.models import load_model
# from sklearn.preprocessing import StandardScaler

# model = load_model('CNNLSTM.h5')
# params = np.array([['Chengalpattu', 'Rice', '30.77', '1309.8', '2.68',
#                    '13.35', '3711.06', '36821', '46.5', '20576', '25.98', '21788', '27.52']])

# print(params.shape)
# print(params)

# params = np.reshape(params, (params.shape[0], params.shape[1], 1))
# y_pred = model.predict(params)
# print(y_pred)
from model import runModel

params = ['Chengalpattu', 'Rice', '30.77', '1309.8', '2.68',
          '13.35', '3711.06', '36821', '46.5', '20576', '25.98', '21788', '27.52']

runModel(params)
