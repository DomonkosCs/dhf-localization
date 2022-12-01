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

        # TODO write for loop
        x_odom = np.array(
            [entry["pose"] for entry in data if ([] not in entry.values())]
        )
        x_true = np.array(
            [entry["truth"] for entry in data if ([] not in entry.values())]
        )
        if "amcl" in data[5]:
            x_amcl = np.array(
                # [entry["amcl"] for entry in data if ([] not in entry.values())]
                [entry["amcl"] for entry in data]
            )
        else:
            x_amcl = []
        # TODO move this to another function
        scans_raw = np.array(
            [entry["scan"] for entry in data if ([] not in entry.values())]
        )

        # times = np.array([entry["t"] for entry in data if ([] not in entry.values())])
        times = np.array([entry["t"] for entry in data])

        if len(scans_raw):
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
        else:
            measurement = []

        return SimulationData(
            x_odom=x_odom,
            x_amcl=x_amcl,
            x_true=x_true,
            measurement=measurement,
            times=times,
        )
