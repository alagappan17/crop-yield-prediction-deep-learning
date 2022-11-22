import numpy as np
from flask import Flask, jsonify, render_template, request
import threading
from threading import *

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

    from model import CNNStackedLSTM, StackedLSTMCNN, CNNBiLSTM

    crops = ["Rice", "Sugarcane", "Sunflower", "Minor Pulses", "Groundnut"]
    predictedOutput1 = []
    predictedOutput2 = []
    predictedOutput3 = []

    def model1(crops):
        global bestCrop, bestYield
        bestYield = -999999999
        for crop in crops:
            features[1] = crop
            prediction = CNNStackedLSTM(features)
            predictedOutput1.append(prediction)
            if (prediction > bestYield):
                bestYield = prediction
                bestCrop = crop

    def model2(crops):
        # global mse1
        for crop in crops:
            features[1] = crop
            prediction1 = StackedLSTMCNN(features)
            predictedOutput2.append(prediction1)

    def model3(crops):
        for crop in crops:
            features[1] = crop
            prediction2 = StackedLSTMCNN(features)
            predictedOutput3.append(prediction2)

    th1 = threading.Thread(target=model1, args=(crops, ))
    th2 = threading.Thread(target=model2, args=(crops, ))
    th3 = threading.Thread(target=model3, args=(crops, ))
    th1.start()
    th2.start()
    th3.start()
    th1.join()
    th2.join()
    th3.join()

    text = "The best crop for " + str(features[0]) + " District is " + str(
        bestCrop)
    return render_template("predict.html", prediction_text=text, cropYield1=predictedOutput1, cropYield2=predictedOutput2, cropYield3=predictedOutput3, cropName=crops, len=len(crops))


if __name__ == "__main__":
    application.run(debug=False)
