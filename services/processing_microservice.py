# Load libraries
import csv
from threading import Timer

from sklearn.model_selection import train_test_split
import numpy as np

import requests
from flask import Flask, jsonify
from flask_caching import Cache

import json

from json import JSONEncoder

import time


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


# config = {
#     "DEBUG": True,  # some Flask specific configs
#     "CACHE_TYPE": "simple",  # Flask-Caching related configs
#     "CACHE_DEFAULT_TIMEOUT": 300
# }

app = Flask(__name__)


time_get_fog_speed_data = 0
time_get_fog_driver_rush_data = 0
time_get_fog_visibility_data = 0
time_get_fog_rain_intensity_data = 0
time_get_fog_pitch_data = 0
time_get_fog_ac_data = 0
time_get_fog_passenger_data = 0
time_get_fog_window_data = 0
time_get_speed_input = 0
time_get_speed_x_train = 0
time_get_speed_x_test = 0
time_get_speed_y_test = 0
time_get_speed_y_train = 0
time_get_ac_control_input = 0
time_get_ac_control_x_test = 0
time_get_ac_control_x_train = 0
time_get_ac_control_y_test = 0
time_get_ac_control_y_train = 0
time_testbed_pitch_data = 0
time_testbed_rain_intensity_data = 0
time_testbed_visibility_data = 0
time_testbed_driver_rush_data = 0
time_testbed_vehicle_speed_data = 0
time_testbed_air_condition_data = 0
time_testbed_passenger_count_data = 0
time_testbed_window_opening_data = 0
time_function_ac_control_train_split = 0
time_function_speed_train_split = 0
total = 0


# app.config.from_mapping(config)
# cache = Cache(app)

air_condition_data_array = []
passenger_count_data_array = []
window_opening_data_array = []
pitch_data_array = []
rain_intensity_data_array = []
visibility_data_array = []
driver_rush_data_array = []
speed_data_array = []

ac_x_train = []
ac_x_test = []
ac_y_train = []
ac_y_test = []
ac_input = []

speed_x_train_data = []
speed_x_test_data = []
speed_y_train_data = []
speed_y_test_data = []
speed_input = []


def write_to_csv(fileName, data):
    with open(fileName, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Data:", data])


# Speed REST Apis

@app.route('/speed/input', methods=['GET'])
# @cache.cached(timeout=300)
def speed_input_list():
    global speed_input
    global time_get_speed_input
    start_time = time.time()
    number_array = speed_input
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    time_get_speed_input = time.time() - start_time
    print("---input %s seconds ---" % (time.time() - start_time))
    write_to_csv('time_get_speed_input.csv', time_get_speed_input)
    return encodedNumpyData


@app.route('/speed/x_train', methods=['GET'])
# @cache.cached(timeout=300)
def speed_x_train():
    global speed_x_train_data
    global time_get_speed_x_train

    start_time = time.time()
    number_array = speed_x_train_data
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    time_get_speed_x_train = time.time() - start_time
    write_to_csv('time_get_speed_x_train.csv', time_get_speed_x_train)
    print("---x_train %s seconds ---" % (time.time() - start_time))
    return encodedNumpyData


@app.route('/speed/x_test', methods=['GET'])
# @cache.cached(timeout=300)
def speed_x_test():
    global speed_x_test_data
    global time_get_speed_x_test
    start_time = time.time()
    number_array = speed_x_test_data
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    time_get_speed_x_test = time.time() - start_time
    print("---x_test %s seconds ---" % (time.time() - start_time))
    write_to_csv('time_get_speed_x_test.csv', time_get_speed_x_test)
    return encodedNumpyData


@app.route('/speed/y_test', methods=['GET'])
# @cache.cached(timeout=300)
def speed_y_test():
    global speed_y_test_data
    global time_get_speed_y_test
    start_time = time.time()
    number_array = speed_y_test_data
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    time_get_speed_y_test = time.time() - start_time
    print("---y_test %s seconds ---" % (time.time() - start_time))
    write_to_csv('time_get_speed_y_test.csv', time_get_speed_y_test)
    return encodedNumpyData


@app.route('/speed/y_train', methods=['GET'])
# @cache.cached(timeout=300)
def speed_y_train():
    global speed_y_train_data
    global time_get_speed_y_train
    start_time = time.time()
    number_array = speed_y_train_data
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    time_get_speed_y_train = time.time() - start_time
    write_to_csv('time_get_speed_y_train.csv', time_get_speed_y_train)
    print("---y_train %s seconds ---" % (time.time() - start_time))
    return encodedNumpyData


# AC REST Apis

@app.route('/ac_control/input', methods=['GET'])
# @cache.cached(timeout=300)
def ac_control_input_list():
    global ac_input
    global time_get_ac_control_input

    start_time = time.time()
    number_array = ac_input
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    time_get_ac_control_input = time.time() - start_time
    print("---input %s seconds ---" % (time.time() - start_time))
    write_to_csv('time_get_ac_control_input.csv', time_get_ac_control_input)
    return encodedNumpyData


@app.route('/ac_control/x_train', methods=['GET'])
# @cache.cached(timeout=300)
def ac_control_x_train():
    global ac_x_train
    global time_get_ac_control_x_train

    start_time = time.time()
    number_array = ac_x_train
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    print("---x_train %s seconds ---" % (time.time() - start_time))
    time_get_ac_control_x_train = time.time() - start_time
    write_to_csv('time_get_ac_control_x_train.csv', time_get_ac_control_x_train)
    return encodedNumpyData


@app.route('/ac_control/x_test', methods=['GET'])
# @cache.cached(timeout=300)
def ac_control_x_test():
    global ac_x_test
    global time_get_ac_control_x_test

    start_time = time.time()
    number_array = ac_x_test
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    time_get_ac_control_x_test = time.time() - start_time
    print("---x_test %s seconds ---" % (time.time() - start_time))
    write_to_csv('time_get_ac_control_x_test.csv', time_get_ac_control_x_test)
    return encodedNumpyData


@app.route('/ac_control/y_test', methods=['GET'])
# @cache.cached(timeout=300)
def ac_control_y_test():
    global ac_y_test
    global time_get_ac_control_y_test

    start_time = time.time()
    number_array = ac_y_test
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    print("---y_test %s seconds ---" % (time.time() - start_time))
    time_get_ac_control_y_test = time.time() - start_time
    write_to_csv('time_get_ac_control_y_test.csv', time_get_ac_control_y_test)
    return encodedNumpyData


@app.route('/ac_control/y_train', methods=['GET'])
# @cache.cached(timeout=300)
def ac_control_y_train():
    global ac_y_train
    global time_get_ac_control_y_train

    start_time = time.time()
    number_array = ac_y_train
    numpyData = {"array": number_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    time_get_ac_control_y_train = time.time() - start_time
    print("---y_train %s seconds ---" % (time.time() - start_time))
    write_to_csv('time_get_ac_control_y_train.csv', time_get_ac_control_y_train)
    return encodedNumpyData


# get data from fog
def get_air_condition_data_fog():
    global air_condition_data_array
    try:
        req = requests.get("http://localhost:3101/cloud/get_fog_ac_data")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])
        air_condition_data_array = finalNumpyArray.copy()

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def get_passenger_count_data_fog():
    global passenger_count_data_array
    try:
        req = requests.get("http://localhost:3101/cloud/get_fog_passenger_data")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])
        passenger_count_data_array = finalNumpyArray.copy()

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def get_window_opening_data_fog():
    global window_opening_data_array
    try:
        req = requests.get("http://localhost:3101/cloud/get_fog_window_data")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])
        window_opening_data_array = finalNumpyArray.copy()

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def get_speed_data_fog():
    global speed_data_array
    try:
        req = requests.get("http://localhost:3101/cloud/get_fog_speed_data")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])
        speed_data_array = finalNumpyArray.copy()

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def get_driver_rush_data_fog():
    global driver_rush_data_array
    try:
        req = requests.get("http://localhost:3101/cloud/get_fog_driver_rush_data")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])
        driver_rush_data_array = finalNumpyArray.copy()

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def get_visibility_data_fog():
    global visibility_data_array
    try:
        req = requests.get("http://localhost:3101/cloud/get_fog_visibility_data")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])
        visibility_data_array = finalNumpyArray.copy()

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def get_rain_intensity_data_fog():
    global rain_intensity_data_array
    try:
        req = requests.get("http://localhost:3101/cloud/get_rain_intensity_data")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])
        rain_intensity_data_array = finalNumpyArray.copy()

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def get_pitch_data_fog():
    global pitch_data_array
    try:
        req = requests.get("http://localhost:3101/cloud/get_fog_pitch_data")
        decodedArrays = json.loads(req.text)

        finalNumpyArray = np.asarray(decodedArrays["array"])
        pitch_data_array = finalNumpyArray.copy()

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return finalNumpyArray


def get_performance_data_fog():
    try:
        req = requests.get("http://192.168.1.112:4006/roof/performance")
        performance_data = float(req.text)

    except requests.exceptions.ConnectionError:
        return "Service unavailable"
    return performance_data

# AC Train Split


def ac_control_train_split():
    global air_condition_data_array, window_opening_data_array, passenger_count_data_array, ac_x_train, ac_x_test, \
        ac_y_train, ac_y_test, ac_input

    window_opening_data = [int(i) for i in window_opening_data_array]
    passenger_count_data = [int(i) for i in passenger_count_data_array]

    air_condition_data = [int(i) for i in air_condition_data_array]

    X = np.array((passenger_count_data, window_opening_data)).T
    Y = air_condition_data

    ac_input = X.copy()

    ac_x_train, ac_x_test, ac_y_train, ac_y_test = train_test_split(X, Y, test_size=0.20, random_state=0)


# Speed Train Split

def speed_train_split():
    global pitch_data_array, passenger_count_data_array, rain_intensity_data_array, \
        visibility_data_array, driver_rush_data_array, speed_data_array, \
        speed_x_train_data, speed_x_test_data, speed_y_train_data, speed_y_test_data, speed_input

    pitch_data = [int(i) for i in pitch_data_array]
    passenger_count_data = [int(i) for i in passenger_count_data_array]
    rain_intensity_data = [int(i) for i in rain_intensity_data_array]
    visibility_data = [int(i) for i in visibility_data_array]
    driver_rush_data = [int(i) for i in driver_rush_data_array]

    speed_data = [float(i) for i in speed_data_array]

    X = np.array(
        (pitch_data, passenger_count_data, rain_intensity_data, visibility_data, driver_rush_data)).T
    Y = speed_data

    if len(X) == len(Y):

        for i in range(len(Y)):
            if Y[i] == 0:
                Y[i] = 0
            elif Y[i] <= 5:
                Y[i] = 1
            elif Y[i] <= 10:
                Y[i] = 2
            elif Y[i] <= 15:
                Y[i] = 3
            elif Y[i] <= 20:
                Y[i] = 4
            elif Y[i] > 20:
                Y[i] = 5
                Y[i] = 5

        speed_input = X.copy()

        speed_x_train_data, speed_x_test_data, speed_y_train_data, speed_y_test_data = train_test_split(X, Y,
                                                                                                        test_size=0.20
                                                                                                        ,
                                                                                                        random_state=0)


# def send_data():
#     global pitch_data_array, passenger_count_data_array, rain_intensity_data_array, \
#         visibility_data_array, driver_rush_data_array, speed_data_array, window_opening_data_array, \
#         air_condition_data_array
#
#     doc_ref.set({
#         'pitch': pitch_data_array,
#         'passenger_count': passenger_count_data_array,
#         'rain_intensity': rain_intensity_data_array,
#         'visibility': visibility_data_array,
#         'driver_rush': driver_rush_data_array,
#         'speed': speed_data_array,
#         'window_opening': window_opening_data_array,
#         'air_condition_status': air_condition_data_array,
#     })

def get_performance_fog():
    performance_data = get_performance_data_fog()
    if performance_data == 0:
        automated_functions()
    else:
        return "service unavailable"


@app.route('/fog/processing/time', methods=['GET'])
# @cache.cached(timeout=300)
def processing_time():
    global total
    global time_get_fog_speed_data
    global time_get_fog_driver_rush_data
    global time_get_fog_visibility_data
    global time_get_fog_rain_intensity_data
    global time_get_fog_pitch_data
    global time_get_fog_ac_data
    global time_get_fog_passenger_data
    global time_get_fog_window_data
    global time_get_speed_input
    global time_get_speed_x_train
    global time_get_speed_x_test
    global time_get_speed_y_test
    global time_get_speed_y_train
    global time_get_ac_control_input
    global time_get_ac_control_x_test
    global time_get_ac_control_x_train
    global time_get_ac_control_y_test
    global time_get_ac_control_y_train
    global time_function_ac_control_train_split
    global time_function_speed_train_split
    total = time_get_fog_speed_data + time_get_fog_driver_rush_data + time_get_fog_visibility_data + time_get_fog_rain_intensity_data + \
            time_get_fog_pitch_data + time_get_fog_ac_data + time_get_fog_passenger_data + time_get_fog_passenger_data + time_get_fog_window_data + \
            time_get_speed_input + time_get_speed_x_train + time_get_speed_x_test + time_get_speed_y_test + time_get_speed_y_train + time_get_ac_control_input + \
            time_get_ac_control_x_test + time_get_ac_control_x_train + time_get_ac_control_y_test + time_get_ac_control_y_train + time_function_ac_control_train_split + \
            time_function_ac_control_train_split + time_function_speed_train_split
    write_to_csv('processing_time_Total.csv', total)
    return total


performance_automated = RepeatedTimer(1, get_performance_fog)


def automated_functions():
    passenger_data_automated = RepeatedTimer(60, get_passenger_count_data_fog)
    window_data_automated = RepeatedTimer(60, get_window_opening_data_fog)
    ac_data_automated = RepeatedTimer(60, get_air_condition_data_fog)
    pitch_data_automated = RepeatedTimer(60, get_pitch_data_fog)
    rain_intensity_data_automated = RepeatedTimer(60, get_rain_intensity_data_fog)
    visibility_data_automated = RepeatedTimer(60, get_visibility_data_fog)
    driver_rush_data_automated = RepeatedTimer(60, get_driver_rush_data_fog)
    speed_data_automated = RepeatedTimer(60, get_speed_data_fog)


ac_train_split_automated = RepeatedTimer(70, ac_control_train_split)
speed_train_split_automated = RepeatedTimer(70, speed_train_split)
time_automated = RepeatedTimer(5, processing_time)

# data_sent_automated = RepeatedTimer(70, send_data)

if __name__ == '__main__':
    app.run(port=5001, host='0.0.0.0', debug=True)
