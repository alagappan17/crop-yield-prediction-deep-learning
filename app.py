import numpy as np
from flask import Flask, jsonify, render_template, request

# import pickle

application = Flask(__name__)


@application.route('/')
def home():
    return render_template("index.html")


@application.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    features = [str(x) for x in request.form.values()]
    features.insert(1, "")

    from model import runModel, runModel2

    crops = ["Rice", "Sugarcane", "Sunflower", "Minor Pulses", "Groundnut"]
    predictedOutput = []
    predictedOutput2 = []
    bestCrop = ""
    bestYield = -999999999999

    for crop in crops:
        features[1] = crop
        prediction = runModel(features)
        prediction2 = runModel2(features)
        predictedOutput.append(prediction)
        predictedOutput2.append(prediction2)
        if (prediction > bestYield):
            bestYield = prediction
            bestCrop = crop

    print("Model1: ", predictedOutput)
    print("Model2: ", predictedOutput2)
    text = "The best crop for " + str(features[0]) + " District is " + str(
        bestCrop) + " with a yield of " + str(bestYield)[2:-2] + "Kg/ha"
    return render_template("predict.html", prediction_text=text, cropYield=predictedOutput, cropYield2=predictedOutput2, cropName=crops, len=len(crops))


if __name__ == "__main__":
    application.run(debug=True)
