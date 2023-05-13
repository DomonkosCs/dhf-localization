import json
import numpy as np
from dhflocalization.rawdata.filehandler import FileHandler
from dhflocalization.evaluator import metrics
from dhflocalization.gridmap import GridMap
from dhflocalization.visualization import TrackPlotter


def eval_mc(filenames, folder=""):
    rmses = []
    comptimes = []

    for filename in filenames:
        rmse, comptime = eval_one(filename, folder)
        rmses.append(rmse)
        comptimes.append(comptime)

    comptimes = np.array(comptimes)
    comp_mean = comptimes.mean()
    comp_std = comptimes.std()

    rmse_pos = np.array([rmse["pos"] for rmse in rmses])
    rmse_ori = np.array([rmse["ori"] for rmse in rmses])

    rmse_pos_mean = rmse_pos.mean()
    rmse_pos_std = rmse_pos.std()

    rmse_ori_mean = rmse_ori.mean()
    rmse_ori_std = rmse_ori.std()

    return {
        "comp_mean": comp_mean,
        "comp_std": comp_std,
        "pos_mean": rmse_pos_mean,
        "pos_std": rmse_pos_std,
        "ori_mean": rmse_ori_mean,
        "ori_std": rmse_ori_std,
    }


def eval_one(filename, folder=""):
    relative_path = "../resources/results/ros/" + folder + filename + ".json"
    fh = FileHandler()
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
    comptime = np.array([entry["comptime"] for entry in data])

    rmse, _, _ = metrics.calc_error_metrics(truth, filtered)

    return rmse, comptime.mean()


## full path, lambda=10
# folder = "0511-0512-night/"
# filename = "amcl_full_final_0511_mc1_5hz_o1e-4_l1e-2_filtered"

# amcl_filenames = [
#     f"amcl_full_final_0511_mc{mc+1}_5hz_o1e-4_l1e-2_filtered" for mc in range(20)
# ]
# edh_filenames = [
#     f"edh_full_final_0511_mc{mc+1}_5hz_o1e-4_l1e-2_filtered" for mc in range(20)
# ]
# ekf_filenames = [
#     f"ekf_full_final_0511_mc{mc+1}_5hz_o1e-4_l1e-2_filtered" for mc in range(20)
# ]
# naedh_filenames = [f"naedh_full_mc{mc+1}_5hz_o1e-4_l1e-2_filtered" for mc in range(3)]
# print(eval_mc(naedh_filenames))


## lambda inspection
# medh_filename = "medh_short_lambda1_mc1_5hz_o1e-4_l1e-2_filtered"
# naedh_filename = "naedh_short_lambda1_mc1_5hz_o1e-4_l1e-2_filtered"
folder = "lambda/"
medh_filenames_l3 = [
    f"medh_short_lambda3_mc{mc+1}_5hz_o1e-4_l1e-2_filtered" for mc in range(10)
]
medh_filenames_l5 = [
    f"medh_short_lambda5_mc{mc+1}_5hz_o1e-4_l1e-2_filtered" for mc in range(10)
]
naedh_filenames_l3 = [
    f"naedh_short_lambda3_mc{mc+1}_5hz_o1e-4_l1e-2_filtered" for mc in range(10)
]
naedh_filenames_l5 = [
    f"naedh_short_lambda5_mc{mc+1}_5hz_o1e-4_l1e-2_filtered" for mc in range(10)
]

print(eval_mc(naedh_filenames_l3, folder))

# print(eval_one(naedh_filename))

# ogm = GridMap.load_map_from_config("gt_map_01_table")
# track_plotter = TrackPlotter(background_map=ogm)
# track_plotter.plot_results(truth, filtered_algo)
