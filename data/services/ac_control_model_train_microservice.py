# Load libraries
import csv

import numpy as np
from numpy import argmax

import requests
from flask import Flask

import json
from json import JSONEncoder

import time
from threading import Timer


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def predict(wh, bh, wo, bo, X_test):
    zh = np.dot(X_test, wh) + bh
    ah = sigmoid(zh)
    zo = np.dot(ah, wo) + bo
    ao = softmax(zo)
    return ao


app = Flask(__name__)

y_predict_array = []


time_ac_control_predict = 0
time_ac_control_output = 0
time_ac_control_get_x_train_data = 0
time_ac_control_get_y_train_data = 0
time_ac_control_get_x_test_data = 0
time_ac_control_get_y_test_data = 0
time_ac_control_get_input_data = 0
time_ac_control_get_fog_wh = 0
time_ac_control_get_fog_bh = 0
time_ac_control_get_fog_wo = 0
time_ac_control_get_fog_bo = 0
time_ac_control_get_cloud_wh = 0
time_ac_control_get_cloud_bh = 0
time_ac_control_get_cloud_wo = 0
time_ac_control_get_cloud_bo = 0
time_ac_control_get_roof_accuracy = 0
time_ac_control_get_fog_accuracy = 0
time_ac_control_get_cloud_accuracy = 0
time_ac_control_model_train = 0
total = 0

# Weight Matrix Define

input_nodes = 2
hidden_nodes = 8
output_labels = 6
wh = np.random.rand(input_nodes, hidden_nodes)
bh = np.random.randn(hidden_nodes)
wo = np.random.rand(hidden_nodes, output_labels)
bo = np.random.randn(output_labels)


def write_to_csv(fileName, data):
    with open(fileName, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Data:", data])


@app.route('/ac_control/predict', methods=['GET'])
# @cache.cached(timeout=300)
def predict_data():
    global time_ac_control_predict
    start_time = time.time()
    number_array = predict_output()
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    time_ac_control_predict = time.time() - start_time
    print("---predict_data %s seconds ---" % (time.time() - start_time))
    write_to_csv('time_ac_control_predict.csv',time_ac_control_predict)
    return encodedNumpyData


@app.route('/ac_control/output', methods=['GET'])
# @cache.cached(timeout=300)
def output_data():
    global time_ac_control_output
    start_time = time.time()
    number_array = output()
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    time_ac_control_output = time.time() - start_time
    print("---output_data %s seconds ---" % (time.time() - start_time))
    write_to_csv('time_ac_control_output.csv', time_ac_control_output)
    return encodedNumpyData


@app.route('/cloud/wh', methods=['GET'])
# @cache.cached(timeout=300)
def wh_data():
    global wh
    global time_ac_control_get_fog_wh

    start_time = time.time()
    number_array = wh
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    time_ac_control_get_fog_wh = time.time() - start_time
    print("---output_data %s seconds ---" % (time.time() - start_time))
    write_to_csv('time_ac_control_get_fog_wh.csv', time_ac_control_get_fog_wh)
    return encodedNumpyData


@app.route('/cloud/bh', methods=['GET'])
# @cache.cached(timeout=300)
def bh_data():
    global bh
    global time_ac_control_get_fog_bh

    start_time = time.time()
    number_array = bh
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    time_ac_control_get_fog_bh = time.time() - start_time
    print("---output_data %s seconds ---" % (time.time() - start_time))
    write_to_csv('time_ac_control_get_fog_bh.csv', time_ac_control_get_fog_bh)
    return encodedNumpyData


@app.route('/cloud/wo', methods=['GET'])
# @cache.cached(timeout=300)
def wo_data():
    global wo
    global time_ac_control_get_cloud_wo

    start_time = time.time()
    number_array = wo
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    time_ac_control_get_cloud_wo = time.time() - start_time
    print("---output_data %s seconds ---" % (time.time() - start_time))
    write_to_csv('time_ac_control_get_cloud_wo.csv', time_ac_control_get_cloud_wo)
    return encodedNumpyData


@app.route('/cloud/bo', methods=['GET'])
# @cache.cached(timeout=300)
def bo_data():
    global bo
    global time_ac_control_get_cloud_bo

    start_time = time.time()
    number_array = bo
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    time_ac_control_get_cloud_bo = time.time() - start_time
    write_to_csv('time_ac_control_get_cloud_bo.csv', time_ac_control_get_cloud_bo)
    print("---output_data %s seconds ---" % (time.time() - start_time))
    return encodedNumpyData


def get_x_train_data():
    try:
        req = requests.get("http://localhost:5001/ac_control/x_train")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def get_y_train_data():
    try:
        req = requests.get("http://localhost:5001/ac_control/y_train")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def get_x_test_data():
    try:
        req = requests.get("http://localhost:5001/ac_control/x_test")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def get_y_test_data():
    try:
        req = requests.get("http://localhost:5001/ac_control/y_test")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def get_input_data():
    try:
        req = requests.get("http://localhost:5001/ac_control/input")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def model_train():
    global time_ac_control_model_train
    start_time = time.time()
    global y_predict_array, wh, bh, wo, bo
    y_train = get_x_train_data()
    x_train = get_x_train_data()
    x_test = get_x_test_data()
    y_test = get_y_test_data()

    if len(x_test) == len(y_test):

        # create a matrix for one hot encoding
        one_hot_labels = np.zeros((len(y_train), 6))
        for i in range(len(y_train)):
            one_hot_labels[i, y_train[i]] = 1

        # input_nodes = 2
        # hidden_nodes = 8
        # output_labels = 6
        # wh = np.random.rand(input_nodes, hidden_nodes)
        # bh = np.random.randn(hidden_nodes)
        # wo = np.random.rand(hidden_nodes, output_labels)
        # bo = np.random.randn(output_labels)
        lr = 10e-4

        error_cost = []

        for epoch in range(5000):
            ############# feedforward

            # Phase 1
            zh = np.dot(x_train, wh) + bh
            ah = sigmoid(zh)
            zo = np.dot(ah, wo) + bo
            ao = softmax(zo)

            ########## Back Propagation

            ########## Phase 1

            dcost_dzo = ao - one_hot_labels
            dzo_dwo = ah

            dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

            dcost_bo = dcost_dzo

            ########## Phases 2

            dzo_dah = wo
            dcost_dah = np.dot(dcost_dzo, dzo_dah.T)
            dah_dzh = sigmoid_der(zh)
            dzh_dwh = x_train
            dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

            dcost_bh = dcost_dah * dah_dzh

            # Update Weights ================

            wh -= lr * dcost_wh
            bh -= lr * dcost_bh.sum(axis=0)

            wo -= lr * dcost_wo
            bo -= lr * dcost_bo.sum(axis=0)

            if epoch % 200 == 0:
                loss = np.sum(-one_hot_labels * np.log(ao))
                # print('Loss function value: ', loss)
                error_cost.append(loss)

        # End of for loop (End of training phase)

        # Make predictions
        predictions = predict(wh, bh, wo, bo, x_test)

        y_predict = []
        for i in range(len(y_test)):
            n = argmax(predictions[i])
            y_predict.append(n)
        y_predict_array = np.array(y_predict)

        print("y_pred", y_predict_array)
        time_ac_control_model_train = time.time() - start_time
        write_to_csv('time_ac_control_model_train.csv', time_ac_control_model_train)


def predict_output():
    global y_predict_array
    # y_predict = model_train()

    return y_predict_array


def output():
    global y_predict_array
    # y_predict = model_train()

    return y_predict_array


@app.route('/roof/ac_control/time', methods=['GET'])
# @cache.cached(timeout=300)
def ac_time():
    global total
    global time_ac_control_predict
    global time_ac_control_output
    global time_ac_control_get_x_train_data
    global time_ac_control_get_y_train_data
    global time_ac_control_get_x_test_data
    global time_ac_control_get_y_test_data
    global time_ac_control_get_input_data
    global time_ac_control_model_train
    total = time_ac_control_predict + time_ac_control_output + time_ac_control_get_x_train_data + time_ac_control_get_y_train_data + \
            time_ac_control_get_x_test_data + time_ac_control_get_y_test_data + time_ac_control_get_input_data + time_ac_control_model_train
    write_to_csv('ac_control_Total.csv', total)
    time_ac_control_predict = 0
    time_ac_control_output = 0
    time_ac_control_get_x_train_data = 0
    time_ac_control_get_y_train_data = 0
    time_ac_control_get_x_test_data = 0
    time_ac_control_get_y_test_data = 0
    time_ac_control_get_input_data = 0
    time_ac_control_model_train = 0
    return total


model_train_automated = RepeatedTimer(90, model_train)
time_automated = RepeatedTimer(1, ac_time)


if __name__ == '__main__':
    app.run(port=5003, host='0.0.0.0', debug=True)
