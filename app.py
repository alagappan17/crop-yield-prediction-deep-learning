import numpy as np
from flask import Flask, request, jsonify, render_template
# import pickle

application = Flask(__name__)  # Initialize the flask App
# model = pickle.load(open('model.pkl', 'rb'))


@application.route('/')
def home():
    return render_template('index.html')


@application.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # features = []
    # features.append(request.form.get("District"))
    # features.append(request.form.get("Crop"))
    # features.append(int(request.form.get("Average_Temperature")))
    # features.append(request.form.get("Precipitation"))
    # features.append(request.form.get("Sea_Level_Pressure"))
    # features.append(request.form.get("Wind"))
    # features.append(request.form.get("Area"))
    # features.append(request.form.get("Nitrogen_Consumption"))
    # features.append(request.form.get("Nitrogen_Share_in_NPK"))
    # features.append(request.form.get("Phosphate_Consumption"))
    # features.append(request.form.get("Phosphate_Share_in_NPK"))
    # features.append(request.form.get("Potash_Consumption"))
    # features.append(request.form.get("Potash_Share_in_NPK"))
    # print(request.form.get("district"))
    # print(request.form.get("avgtemp"))
    features = [str(x) for x in request.form.values()]
    # features = [np.array(features)]

    # for x in request.form.values():
    #     features.append(x)
    # features1 = [int(x) for x in request.form.values()]
    # features = np.concatenate((features, features1))
    # features = [np.array(features)]
    print(features)
    # prediction = model.predict(final_features)
    from model import runModel
    prediction = runModel(features)

    # output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Crop Yield is {} Kg/ha'.format(float(prediction)))


if __name__ == "__main__":
    application.run(debug=True)
