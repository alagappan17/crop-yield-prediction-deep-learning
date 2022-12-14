import numpy as np
from flask import Flask, jsonify, render_template, request
import threading
from threading import *
from DataFormatter import convertToInputFormat, convertToOutputFormat
from keras.models import load_model


# import pickle

application = Flask(__name__)


@application.route('/')
def home():
    return render_template("index.html")


@application.route('/predict', methods=['POST'])
def predict():

    crops = ["Rice", "Sugarcane", "Sunflower", "Minor Pulses", "Groundnut"]
    predictedOutput1 = []
    predictedOutput2 = []
    predictedOutput3 = []
    predictedOutput4 = []
    CNNStackedLSTM = load_model("CNNStackedLSTM.h5")
    StackedLSTMCNN = load_model("StackedLSTMCNN.h5")
    CNNBiLSTM = load_model("CNNBiLSTM.h5")
    BiLSTMCNN = load_model("BiLSTMCNN.h5")

    userInput = [str(x) for x in request.form.values()]
    userInput.insert(1, "")

    # from model import CNNStackedLSTM, StackedLSTMCNN, CNNBiLSTM
    global bestCrop, bestYield
    bestYield = -9999999999
    for crop in crops:
        userInput[1] = crop
        print(userInput)
        transformedData = convertToInputFormat(userInput)
        pred1 = CNNStackedLSTM(transformedData)
        pred1 = convertToOutputFormat(userInput, pred1)
        print(pred1)
        predictedOutput1.append(pred1)

        pred2 = StackedLSTMCNN(transformedData)
        pred2 = convertToOutputFormat(userInput, pred2)
        print(pred2)
        predictedOutput2.append(pred2)

        pred3 = CNNBiLSTM(transformedData)
        pred3 = convertToOutputFormat(userInput, pred3)
        print(pred3)
        predictedOutput3.append(pred3)

        pred4 = BiLSTMCNN(transformedData)
        pred4 = convertToOutputFormat(userInput, pred4)
        print(pred4)
        predictedOutput4.append(pred4)

        if (pred1 > bestYield):
            bestYield = pred1
            bestCrop = crop

    # def model1(crops):
    #     global bestCrop, bestYield
    #     bestYield = -999999999
    #     for crop in crops:
    #         features[1] = crop
    #         prediction = CNNStackedLSTM(features)
    #         predictedOutput1.append(prediction)
    #         if (prediction > bestYield):
    #             bestYield = prediction
    #             bestCrop = crop

    # def model2(crops):
    #     # global mse1
    #     for crop in crops:
    #         features[1] = crop
    #         prediction1 = StackedLSTMCNN(features)
    #         predictedOutput2.append(prediction1)

    # def model3(crops):
    #     for crop in crops:
    #         features[1] = crop
    #         prediction2 = CNNBiLSTM(features)
    #         predictedOutput3.append(prediction2)

    # th1 = threading.Thread(target=model1, args=(crops, ))
    # th2 = threading.Thread(target=model2, args=(crops, ))
    # th3 = threading.Thread(target=model3, args=(crops, ))
    # th1.start()
    # th2.start()
    # th3.start()
    # th1.join()
    # th2.join()
    # th3.join()

    text = "The best crop for " + str(userInput[0]) + " District is " + str(
        bestCrop)
    return render_template("predict.html", prediction_text=text, cropYield1=predictedOutput1, cropYield2=predictedOutput2, cropYield3=predictedOutput3, cropYield4=predictedOutput4,  cropName=crops, len=len(crops))


if __name__ == "__main__":
    application.run(debug=False)
