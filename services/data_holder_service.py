# Load libraries
import numpy as np
from flask import Flask, request
import json
from json import JSONEncoder
import queue


class DataHolder:
    __instance = None

    @staticmethod
    def get_instance():
        if DataHolder.__instance is None:
            DataHolder()
        return DataHolder.__instance

    def __init__(self):
        if DataHolder.__instance is not None:
            raise Exception("This is a singleton")
        else:
            self.__fog_speed_data_q = queue.Queue()
            self.__fog_driver_rush_data_q = queue.Queue()
            self.__fog_visibility_data_q = queue.Queue()
            self.__fog_rain_intensity_data_q = queue.Queue()
            self.__fog_pitch_data_q = queue.Queue()
            self.__fog_ac_data_q = queue.Queue()
            self.__fog_passenger_data_q = queue.Queue()
            self.__fog_window_data_q = queue.Queue()
            DataHolder.__instance = self

    def add_fog_speed_data(self, data):
        if data is not None:
            self.__fog_speed_data_q.put(data)

    def get_fog_speed_data(self):
        if not self.__fog_speed_data_q.empty():
            return self.__fog_speed_data_q.get(timeout=100)
        return "No data found"

    def add_fog_driver_rush_data(self, data):
        if data is not None:
            self.__fog_driver_rush_data_q.put(data)

    def get_fog_driver_rush_data(self):
        if not self.__fog_driver_rush_data_q.empty():
            return self.__fog_driver_rush_data_q.get(timeout=100)
        return "No data found"

    def add_fog_visibility_data(self, data):
        if data is not None:
            self.__fog_visibility_data_q.put(data)

    def get_fog_visibility_data(self):
        if not self.__fog_visibility_data_q.empty():
            return self.__fog_visibility_data_q.get(timeout=100)
        return "No data found"

    def add_fog_rain_intensity_data(self, data):
        if data is not None:
            self.__fog_rain_intensity_data_q.put(data)

    def get_fog_rain_intensity_data(self):
        if not self.__fog_rain_intensity_data_q.empty():
            return self.__fog_rain_intensity_data_q.get(timeout=100)
        return "No data found"

    def add_fog_pitch_data(self, data):
        if data is not None:
            self.__fog_pitch_data_q.put(data)

    def get_fog_pitch_data(self):
        if not self.__fog_pitch_data_q.empty():
            return self.__fog_pitch_data_q.get(timeout=100)
        return "No data found"

    def add_fog_ac_data(self, data):
        if data is not None:
            self.__fog_ac_data_q.put(data)

    def get_fog_ac_data(self):
        if not self.__fog_ac_data_q.empty():
            return self.__fog_ac_data_q.get(timeout=100)
        return "No data found"

    def add_fog_passenger_data(self, data):
        if data is not None:
            self.__fog_passenger_data_q.put(data)

    def get_fog_passenger_data(self):
        if not self.__fog_passenger_data_q.empty():
            return self.__fog_passenger_data_q.get(timeout=100)
        return "No data found"

    def add_fog_window_data(self, data):
        if data is not None:
            self.__fog_window_data_q.put(data)

    def get_fog_window_data(self):
        if not self.__fog_window_data_q.empty():
            return self.__fog_window_data_q.get(timeout=100)
        return "No data found"


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


config = {
    "DEBUG": True,  # some Flask specific configs
    "CACHE_TYPE": "simple",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}

app = Flask(__name__)
app.config.from_mapping(config)


@app.route('/cloud/add_fog_speed_data', methods=['POST'])
def add_fog_speed_data_to_queue():
    record = json.loads(request.data)
    DataHolder.get_instance().add_fog_speed_data(record)
    return "success"


@app.route('/cloud/get_fog_speed_data', methods=['GET'])
def get_fog_speed_to_queue():
    return DataHolder.get_instance().get_fog_speed_data()


@app.route('/cloud/add_fog_driver_rush_data', methods=['POST'])
def add_fog_driver_rush_data_to_queue():
    record = json.loads(request.data)
    DataHolder.get_instance().add_fog_driver_rush_data(record)
    return "success"


@app.route('/cloud/get_fog_driver_rush_data', methods=['GET'])
def get_fog_driver_rush_data_to_queue():
    return DataHolder.get_instance().get_fog_driver_rush_data()


@app.route('/cloud/add_fog_visibility_data', methods=['POST'])
def add_fog_visibility_data_to_queue():
    record = json.loads(request.data)
    DataHolder.get_instance().add_fog_visibility_data(record)
    return "success"


@app.route('/cloud/get_fog_visibility_data', methods=['GET'])
def get_fog_visibility_data_to_queue():
    return DataHolder.get_instance().get_fog_visibility_data()


@app.route('/cloud/add_rain_intensity_data', methods=['POST'])
def add_fog_rain_intensity_data_to_queue():
    record = json.loads(request.data)
    DataHolder.get_instance().add_fog_rain_intensity_data(record)
    return "success"


@app.route('/cloud/get_rain_intensity_data', methods=['GET'])
def get_fog_rain_intensity_data_to_queue():
    return DataHolder.get_instance().get_fog_rain_intensity_data()


@app.route('/cloud/add_fog_pitch_data', methods=['POST'])
def add_fog_pitch_data_to_queue():
    record = json.loads(request.data)
    DataHolder.get_instance().add_fog_pitch_data(record)
    return "success"


@app.route('/cloud/get_fog_pitch_data', methods=['GET'])
def get_fog_pitch_data_to_queue():
    return DataHolder.get_instance().get_fog_pitch_data()


@app.route('/cloud/add_fog_ac_data', methods=['POST'])
def add_fog_ac_data_to_queue():
    record = json.loads(request.data)
    DataHolder.get_instance().add_fog_ac_data(record)
    return "success"


@app.route('/cloud/get_fog_ac_data', methods=['GET'])
def get_fog_ac_data_to_queue():
    return DataHolder.get_instance().get_fog_ac_data()


@app.route('/cloud/add_fog_passenger_data', methods=['POST'])
def add_fog_passenger_data_to_queue():
    record = json.loads(request.data)
    DataHolder.get_instance().add_fog_passenger_data(record)
    return "success"


@app.route('/cloud/get_fog_passenger_data', methods=['GET'])
def get_fog_passenger_data_to_queue():
    return DataHolder.get_instance().get_fog_passenger_data()


@app.route('/cloud/add_fog_window_data', methods=['POST'])
def add_fog__window_data_to_queue():
    record = json.loads(request.data)
    DataHolder.get_instance().add_fog_window_data(record)
    return "success"


@app.route('/cloud/get_fog_window_data', methods=['GET'])
def get_fog__window_data_to_queue():
    return DataHolder.get_instance().get_fog_window_data()


if __name__ == '__main__':
    app.run(port=3101, host='0.0.0.0', debug=True)
