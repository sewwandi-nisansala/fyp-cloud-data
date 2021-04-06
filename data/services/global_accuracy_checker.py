# Load libraries
from sklearn.metrics import accuracy_score
import numpy as np

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


app = Flask(__name__)

global_ac_accuracy = 0

global_ac_wh = 0
global_ac_bh = 0
global_ac_wo = 0
global_ac_bo = 0

global_speed_accuracy = 0

global_speed_wh = 0
global_speed_bh = 0
global_speed_wo = 0
global_speed_bo = 0


@app.route('/ac/accuracy', methods=['GET'])
def ac_status_accuracy():
    global global_ac_accuracy

    start_time = time.time()
    accuracy_value = global_ac_accuracy
    print("---ac accuracy %s seconds ---" % (time.time() - start_time))
    return str(accuracy_value)


@app.route('/ac/wh', methods=['GET'])
def ac_wh_output():
    global global_ac_wh

    start_time = time.time()
    number_array = global_ac_wh
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    print("---output_data %s seconds ---" % (time.time() - start_time))
    return encodedNumpyData


@app.route('/ac/wo', methods=['GET'])
def ac_wo_output():
    global global_ac_wo

    start_time = time.time()
    number_array = global_ac_wo
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    print("---output_data %s seconds ---" % (time.time() - start_time))
    return encodedNumpyData


@app.route('/ac/bh', methods=['GET'])
def ac_bh_output():
    global global_ac_bh

    start_time = time.time()
    number_array = global_ac_bh
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    print("---output_data %s seconds ---" % (time.time() - start_time))
    return encodedNumpyData


@app.route('/ac/bo', methods=['GET'])
def ac_bo_output():
    global global_ac_bo

    start_time = time.time()
    number_array = global_ac_bo
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    print("---output_data %s seconds ---" % (time.time() - start_time))
    return encodedNumpyData


@app.route('/speed/accuracy', methods=['GET'])
def speed_accuracy_output():
    global global_speed_accuracy

    start_time = time.time()
    accuracy_value = global_speed_accuracy
    print("---speed accuracy %s seconds ---" % (time.time() - start_time))
    return str(accuracy_value)


@app.route('/speed/wh', methods=['GET'])
def speed_wh_output():
    global global_speed_wh

    start_time = time.time()
    number_array = global_speed_wh
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    print("---output_data %s seconds ---" % (time.time() - start_time))
    return encodedNumpyData


@app.route('/speed/wo', methods=['GET'])
def speed_wo_output():
    global global_speed_wo

    start_time = time.time()
    number_array = global_speed_wo
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    print("---output_data %s seconds ---" % (time.time() - start_time))
    return encodedNumpyData


@app.route('/speed/bh', methods=['GET'])
def speed_bh_output():
    global global_speed_bh

    start_time = time.time()
    number_array = global_speed_bh
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    print("---output_data %s seconds ---" % (time.time() - start_time))
    return encodedNumpyData


@app.route('/speed/bo', methods=['GET'])
def speed_bo_output():
    global global_speed_bo

    start_time = time.time()
    number_array = global_speed_bo
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    print("---output_data %s seconds ---" % (time.time() - start_time))
    return encodedNumpyData


def get_vehicle_ac_status_accuracy():
    global global_ac_accuracy, global_ac_wh, global_ac_wo, global_ac_bh, global_ac_bo

    try:
        req = requests.get("http://localhost:5002/ac_control/accuracy")
        accuracy = float(req.text)

        length = get_ac_control_predict_data_length()
        print("ac length", length)
        if length > 1000:
            if accuracy > global_ac_accuracy:
                global_ac_accuracy = accuracy

                global_ac_wo = get_ac_control_wo()
                global_ac_wh = get_ac_control_wh()
                global_ac_bo = get_ac_control_bo()
                global_ac_bh = get_ac_control_bh()
    except requests.exceptions.ConnectionError:
        return "Service unavailable"


def get_vehicle_speed_accuracy():
    global global_speed_accuracy, global_speed_bh, global_speed_bo, global_speed_wh, global_speed_wo
    try:
        req = requests.get("http://localhost:5002/speed/accuracy")
        accuracy = float(req.text)

        length = get_speed_predict_data_length()
        print("speed length", length)
        if length > 100:
            if accuracy > global_speed_accuracy:
                global_speed_accuracy = accuracy

                global_speed_wo = get_speed_fog_wo()
                global_speed_wh = get_speed_fog_wh()
                global_speed_bo = get_speed_fog_bo()
                global_speed_bh = get_speed_fog_bh()

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return accuracy


def get_ac_control_predict_data_length():
    try:
        req = requests.get("http://localhost:5003/ac_control/predict")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return len(finalNumpyArray)


def get_ac_control_wh():
    try:
        req = requests.get("http://localhost:5003/fog/wh")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return len(finalNumpyArray)


def get_ac_control_bh():
    try:
        req = requests.get("http://localhost:5003/fog/bh")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return len(finalNumpyArray)


def get_ac_control_wo():
    try:
        req = requests.get("http://localhost:5003/fog/wo")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return len(finalNumpyArray)


def get_ac_control_bo():
    try:
        req = requests.get("http://localhost:5003/fog/bo")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return len(finalNumpyArray)


def get_speed_predict_data_length():
    try:
        req = requests.get("http://localhost:5201/speed/predict")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return len(finalNumpyArray)


def get_speed_fog_wh():
    try:
        req = requests.get("http://localhost:5201/fog/wh")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def get_speed_fog_bh():
    try:
        req = requests.get("http://localhost:5201/fog/bh")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def get_speed_fog_wo():
    try:
        req = requests.get("http://localhost:5201/fog/wo")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def get_speed_fog_bo():
    try:
        req = requests.get("http://localhost:5201/fog/bo")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


ac_accuracy_automated_1 = RepeatedTimer(40, get_vehicle_ac_status_accuracy)

speed_accuracy_automated_1 = RepeatedTimer(40, get_vehicle_speed_accuracy)

if __name__ == '__main__':
    app.run(port=5500)
