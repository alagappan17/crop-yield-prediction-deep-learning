import numpy as np
from flask import Flask, request, jsonify, render_template
# import pickle

application = Flask(__name__)


@application.route('/')
def home():
    return render_template('index.html')


@application.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [str(x) for x in request.form.values()]
    features.insert(1, "")

    from model import runModel

    crops = ["Rice", "Sugarcane", "Sunflower", "Minor Pulses", "Groundnut"]
    predictedOutput = []
    bestCrop = ""
    bestYield = -999999999999

    for crop in crops:
        features[1] = crop
        prediction = runModel(features)
        predictedOutput.append(prediction)
        if (prediction > bestYield):
            bestYield = prediction
            bestCrop = crop

    text = "The best crop for " + str(features[0]) + " District is " + str(
        bestCrop) + " with a yield of " + str(bestYield)[2:-2] + "Kg/ha"
    return render_template('index.html', prediction_text=text, cropYield=predictedOutput, cropName=crops, length=len(crops))


if __name__ == "__main__":
    application.run(debug=True)
