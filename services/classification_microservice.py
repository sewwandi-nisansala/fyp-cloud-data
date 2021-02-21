# Load libraries
import csv

import numpy as np

from sklearn.metrics import classification_report

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

time_ac_classification_report_output = 0
time_speed_classification_report_output = 0


def write_to_csv(fileName, data):
    with open(fileName, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Data:", data])


@app.route('/ac_control/classification_report', methods=['GET'])
def ac_classification_report_output():
    global time_ac_classification_report_output
    start_time = time.time()
    ac_control_classification_report_function()
    time_ac_classification_report_output = time.time() - start_time
    print("---classification_report ac %s seconds ---" % (time.time() - start_time))
    write_to_csv('time_ac_classification_report_output.csv', time_ac_classification_report_output)
    return "classification report"
    # return str(confusion_matrix_value)


@app.route('/speed/classification_report', methods=['GET'])
def speed_classification_report_output():
    global time_speed_classification_report_output
    start_time = time.time()
    speed_classification_report_function()
    print("---classification_report speed %s seconds ---" % (time.time() - start_time))
    time_speed_classification_report_output = time.time() - start_time
    write_to_csv('time_speed_classification_report_output.csv', time_speed_classification_report_output)
    return "classification report"
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


def ac_control_classification_report_function():
    print('classification_report: ')
    print(classification_report(get_ac_control_y_test_data(), get_ac_control_predict_data()))


def speed_classification_report_function():
    print('classification_report: ')
    print(classification_report(get_speed_y_test_data(), get_speed_predict_data()))


if __name__ == '__main__':
    app.run(port=5005, host='0.0.0.0', debug=True)
