# Load libraries
import csv

import numpy as np
from sklearn.metrics import confusion_matrix
import requests
from flask import Flask
import json
from json import JSONEncoder
import time


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


app = Flask(__name__)

time_get_ac_status_confusion = 0
time_get_speed_confusion = 0


def write_to_csv(fileName, data):
    with open(fileName, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Data:", data])


@app.route('/ac_control/confusion_matrix', methods=['GET'])
def ac_status_confuion():
    global time_get_ac_status_confusion
    start_time = time.time()
    ac_control_confusion_matrix_function()
    time_get_ac_status_confusion = time.time() - start_time
    print("--- %s seconds ---" % (time.time() - start_time))
    write_to_csv('time_get_ac_status_confusion.csv', time_get_ac_status_confusion)
    return "confusion matrix"
    # return str(confusion_matrix_value)


@app.route('/speed/confusion_matrix', methods=['GET'])
def speed_confuion():
    global time_get_speed_confusion
    start_time = time.time()
    speed_confusion_matrix_function()
    time_get_speed_confusion = time.time() - start_time
    print("--- %s seconds ---" % (time.time() - start_time))
    write_to_csv('time_get_speed_confusion.csv', time_get_speed_confusion)
    return "confusion matrix"
    # return str(confusion_matrix_value)


def get_ac_control_y_test_data():
    try:
        req = requests.get("http://localhost:5001/ac_control/y_test")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def get_ac_control_predict_data():
    try:
        req = requests.get("http://localhost:5003/ac_control/predict")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def get_speed_y_test_data():
    try:
        req = requests.get("http://localhost:5001/speed/y_test")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def get_speed_predict_data():
    try:
        req = requests.get("http://localhost:5201/speed/predict")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def ac_control_confusion_matrix_function():
    print('confusion_matrix: ')
    confusion_matrix_value = confusion_matrix(get_ac_control_y_test_data(), get_ac_control_predict_data())
    print(confusion_matrix_value)
    # return confusion_matrix_value


def speed_confusion_matrix_function():
    print('confusion_matrix: ')
    confusion_matrix_value = confusion_matrix(get_speed_y_test_data(), get_speed_predict_data())
    print(confusion_matrix_value)
    # return confusion_matrix_value


if __name__ == '__main__':
    app.run(port=5004, host='0.0.0.0', debug=True)
