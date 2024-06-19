import json
import numpy as np
import matplotlib.pyplot as plt
from dhflocalization.rawdata.filehandler import FileHandler
from dhflocalization.evaluator import metrics
from dhflocalization.gridmap import GridMap
from dhflocalization.visualization import TrackPlotter


def get_rot_matrix(angle_deg):
    angle_rad = angle_deg * 2 * np.pi / 360
    rot_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1],
        ]
    )
    return rot_matrix


fh = FileHandler()
filenames = [
    "amcl__mc1_take_lowcov_1_0611",
    "amclp__mc1_take_lowcov_1_0611",
    "medh__mc1_take_lowcov_1_0611",
]
alg_names = ["amcl", "amclplus", "medh"]
relative_paths = [
    "../resources/results/ros/" + filename + ".json" for filename in filenames
]
file_paths = [
    fh.convert_path_to_absolute(relative_path) for relative_path in relative_paths
]

ogm = GridMap.load_map_from_config("bme_map_withbushes")
track_plotter = TrackPlotter(background_map=ogm)

for deg in range(15, 16, 1):
    filtered_algos = {}
    ANGLE_DEG = deg  # rotate GT trajectory
    for file_path, alg_name in zip(file_paths, alg_names):
        try:
            json_file = open(
                file_path,
            )
        except AttributeError:
            raise ValueError("File not found at {}".format(file_path))

        data = json.load(json_file)
        data = data["data"]

        truth = np.array([entry["truth"] for entry in data])
        truth = truth @ get_rot_matrix(ANGLE_DEG)
        filtered = np.array([entry["pose"] for entry in data])
        filtered_algos[alg_name] = {"track": filtered, "truth": truth}
        comptime = np.array([entry["comptime"] for entry in data])

        (err_mean_sqare, err_mean_abs, std) = metrics.calc_error_metrics(
            truth, filtered
        )
        print(f"{alg_name}, {deg} : {err_mean_sqare['pos']}")

        # print(alg_name)
        # print(err_mean_sqare)
        # print("---")
        # print(err_mean_abs)
        # print("---")
        # print(std)
        # print("---")
        # print(comptime.mean())

# track_plotter.plot_results(filtered_algos["medh"]["truth"], filtered_algos)
# plt.show()
