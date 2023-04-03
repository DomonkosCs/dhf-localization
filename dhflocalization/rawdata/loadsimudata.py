import json
import numpy as np
from ..customtypes import SimulationData
from ..rawdata.filehandler import FileHandler


class RawDataLoader(FileHandler):
    def __init__(self):
        pass

    @classmethod
    def load_from_json(cls, filename):
        relative_path = "../resources/simulations/" + filename + ".json"
        file_path = super().convert_path_to_absolute(cls, relative_path)
        try:
            json_file = open(
                file_path,
            )
        except AttributeError:
            raise ValueError("File not found at {}".format(file_path))

        data = json.load(json_file)
        data = data["data"]

        if "pose" in data[5]:
            x_odom = np.array(
                [entry["pose"] for entry in data if ([] not in entry.values())]
            )
        else:
            x_odom = []

        if "truth" in data[5]:
            x_true = np.array(
                [entry["truth"] for entry in data if ([] not in entry.values())]
            )
        else:
            x_true = []

        if "amcl" in data[5]:
            x_amcl = np.array(
                # [entry["amcl"] for entry in data if ([] not in entry.values())]
                [entry["amcl"] for entry in data]
            )
        else:
            x_amcl = []

        if "scan" in data[5]:
            scans_raw = np.array(
                [entry["scan"] for entry in data if ([] not in entry.values())]
            )
            # TODO move this to another function
            angles = np.linspace(0, 2 * np.pi, len(scans_raw[0]))
            measurement = []
            for scan in scans_raw:
                measurement.append(
                    [
                        (angle, range)
                        for angle, range in zip(angles, scan)
                        if range is not None
                    ]
                )
        else:
            measurement = []

        # times = np.array([entry["t"] for entry in data if ([] not in entry.values())])
        times = np.array([entry["t"] for entry in data])

        return SimulationData(
            x_odom=x_odom,
            x_amcl=x_amcl,
            x_true=x_true,
            measurement=measurement,
            times=times,
        )
