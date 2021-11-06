import json
import numpy as np


class RawDataLoader:
    def __init__(self):
        pass

    @classmethod
    def loadFromJson(cls, filename):
        json_file = open(filename,)
        data = json.load(json_file)
        data = data['data']

        x_odom = [entry['pose']
                  for entry in data if ([] not in entry.values())]
        x_true = [entry['truth']
                  for entry in data if ([] not in entry.values())]
        x_amcl = [entry['amcl']
                  for entry in data if ([] not in entry.values())]
        scans_raw = [entry['scan']
                     for entry in data if ([] not in entry.values())]
        angles = np.linspace(0, 2*np.pi, len(scans_raw[0]), endpoint=False)
        measurement = []
        for scan in scans_raw:
            measurement.append(
                [(angle, range) for angle, range in zip(angles, scan) if range != None])

        return x_odom, measurement, x_true, x_amcl
