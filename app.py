import numpy as np
from flask import Flask, jsonify, render_template, request
import threading
from threading import *
from DataFormatter import convertToInputFormat, convertToOutputFormat
from keras.models import load_model
import requests
import time
import json
import warnings

warnings.filterwarnings('ignore')


application = Flask(__name__)


@application.route('/')
def home():
    return render_template("index.html")


@application.route('/predict', methods=['POST'])
def predict():

    crops = ["Rice", "Sugarcane", "Sunflower", "Minor Pulses", "Groundnut",
             "Cotton", "Sesamum", "Millets", "Pigeon Pea", "Oilseeds"]
    predictedOutput1 = []
    predictedOutput2 = []
    predictedOutput3 = []
    predictedOutput4 = []
    CNNStackedLSTM = load_model("CNNStackedLSTM.h5")
    StackedLSTMCNN = load_model("StackedLSTMCNN.h5")
    CNNBiLSTM = load_model("CNNBiLSTM.h5")
    BiLSTMCNN = load_model("BiLSTMCNN.h5")

    userInput = [str(x) for x in request.form.values()]
    print("Form Data: ")
    print(userInput)
    userInput.insert(1, "")
    district = userInput[0]

    if (district == "Ramananthapuram"):
        district = "Ramanathapuram"
    elif (district == "Thanjavaur"):
        district = "Thanjavur"
    elif (district == "Thirunelveli"):
        district = "Tirunelveli"

    current_weather_url = "http://api.weatherapi.com/v1/current.json?key=10aa7005b9c041adbaa80125232404&q=" + district + "&aqi=no"
    data = requests.get(current_weather_url)
    decoded = data.content.decode("utf-8")
    data = json.loads(decoded)
    print("\n")
    print("Weather API data: ")
    print(data)
    temp = data['current']['temp_c']
    precipitation = data['current']['precip_mm']
    pressure = data['current']['pressure_mb']
    wind = data['current']['wind_kph']
    userInput.insert(3, temp)
    userInput.insert(4, precipitation)
    userInput.insert(5, round(pressure / 600, 2))
    userInput.insert(6, wind)

    npk_sensor_url = "https://api.thingspeak.com/channels/2118429/feeds.json?api_key=4PIT8MV6LRZDLUTQ&results=1"
    npk_response = requests.get(npk_sensor_url)
    print("\n")
    print("IoT Sensor Data: ")
    print(npk_response.text)
    npk_data = json.loads(npk_response.text)
    print(npk_data)
    n_value = int(npk_data['feeds'][0]['field1'])
    p_value = int(npk_data['feeds'][0]['field2'])
    k_value = int(npk_data['feeds'][0]['field3'])

    npk_sum = n_value + p_value + k_value
    n_share = (n_value/npk_sum) * 100
    p_share = (p_value/npk_sum) * 100
    k_share = (k_value/npk_sum) * 100

    userInput.insert(7, n_share)
    userInput.insert(8, p_share)
    userInput.insert(9, k_share)

    # print(npk_data['feeds'][0]['field1'])
    # print(npk_data['feeds'][0]['field2'])
    print("Input to Model: ")
    print(userInput)

    # from model import CNNStackedLSTM, StackedLSTMCNN, CNNBiLSTM

    global bestCrop, bestYield
    bestYield = -9999999999
    for crop in crops:
        print("Crop - " + crop)
        userInput[1] = crop
        print("Model Input: ")
        print(userInput)
        transformedData = convertToInputFormat(userInput)
        pred1 = CNNStackedLSTM(transformedData)
        pred1 = convertToOutputFormat(userInput, pred1)
        print("Predicted Output for " + crop + " using CNN Stacked LSTM: ")
        print(pred1)
        pred1 = np.round(pred1, 2)
        predictedOutput1.append(pred1)

        pred2 = StackedLSTMCNN(transformedData)
        pred2 = convertToOutputFormat(userInput, pred2)
        print("Predicted Output for " + crop + " using Stacked LSTM CNN: ")
        print(pred2)
        pred2 = np.round(pred2, 2)
        predictedOutput2.append(pred2)

        pred3 = CNNBiLSTM(transformedData)
        pred3 = convertToOutputFormat(userInput, pred3)
        print("Predicted Output for " + crop + " using CNN Bi LSTM: ")
        print(pred3)
        pred3 = np.round(pred3, 2)
        predictedOutput3.append(pred3)

        pred4 = BiLSTMCNN(transformedData)
        pred4 = convertToOutputFormat(userInput, pred4)
        print("Predicted Output for " + crop + " using Bi LSTM CNN: ")
        print(pred4)
        pred4 = np.round(pred4, 2)
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
    return render_template("predict.html", prediction_text=text, cropYield1=predictedOutput1, cropYield2=predictedOutput2, cropYield3=predictedOutput3, cropYield4=predictedOutput4,  cropName=crops, len=len(crops), factors=userInput)


if __name__ == "__main__":
    application.run(debug=False)
