import json
import numpy as np
from dhflocalization.rawdata.filehandler import FileHandler
from dhflocalization.evaluator import metrics
from dhflocalization.gridmap import GridMap
from dhflocalization.visualization import TrackPlotter

fh = FileHandler()
# filename = "amcl_5hz_o1e-4_l1e-2_filtered"
filename = "edh_mc1_5hz_o1e-4_l1e-2_filtered"
relative_path = "../resources/results/ros/" + filename + ".json"
file_path = fh.convert_path_to_absolute(relative_path)
try:
    json_file = open(
        file_path,
    )
except AttributeError:
    raise ValueError("File not found at {}".format(file_path))

data = json.load(json_file)
data = data["data"]

truth = np.array([entry["truth"] for entry in data])
filtered = np.array([entry["pose"] for entry in data])
filtered_algo = {"edh": {"track": filtered}}
comptime = np.array([entry["comptime"] for entry in data])

(err_mean_sqare, err_mean_abs, std) = metrics.calc_error_metrics(truth, filtered)

print(err_mean_sqare)
print("---")
print(err_mean_abs)
print("---")
print(std)
print("---")
print(comptime.mean())

ogm = GridMap.load_map_from_config("gt_map_01_table")
track_plotter = TrackPlotter(background_map=ogm)
track_plotter.plot_results(truth, filtered_algo)
