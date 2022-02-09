import json
import numpy as np
from customtypes import SimulationData


class RawDataLoader:
    def __init__(self):
        pass

    @classmethod
    def loadFromJson(cls, filename):
        file_path = "/Users/domonkoscsuzdi/dhf_loc/dhflocalization/resources/simulations/{}.json".format(
            filename
        )
        try:
            json_file = open(
                file_path,
            )
        except AttributeError:
            raise ValueError("File not found at {}".format(file_path))

        data = json.load(json_file)
        data = data["data"]

        x_odom = [entry["pose"] for entry in data if ([] not in entry.values())]
        x_true = [entry["truth"] for entry in data if ([] not in entry.values())]
        x_amcl = [entry["amcl"] for entry in data if ([] not in entry.values())]
        scans_raw = [entry["scan"] for entry in data if ([] not in entry.values())]
        angles = np.linspace(0, 2 * np.pi, len(scans_raw[0]), endpoint=False)
        measurement = []
        for scan in scans_raw:
            measurement.append(
                [
                    (angle, range)
                    for angle, range in zip(angles, scan)
                    if range is not None
                ]
            )

        return SimulationData(
            x_odom=x_odom, x_amcl=x_amcl, x_true=x_true, measurement=measurement
        )
