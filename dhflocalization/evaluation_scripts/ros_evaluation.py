import json
import numpy as np
import matplotlib.pyplot as plt
from dhflocalization.rawdata.filehandler import FileHandler
from dhflocalization.evaluator import metrics
from dhflocalization.gridmap import GridMap
from dhflocalization.visualization import TrackPlotter


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


def eval_mc(filenames, folder=""):
    PRECISION = 2

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

    rmse_pos_mean = rmse_pos.mean() * 1000
    rmse_pos_std = rmse_pos.std() * 1000

    rmse_ori_mean = rmse_ori.mean() * 1000
    rmse_ori_std = rmse_ori.std() * 1000

    return {
        "comp_mean": round(comp_mean, PRECISION),
        "comp_std": round(comp_std, PRECISION),
        "pos_mean": round(rmse_pos_mean, PRECISION),
        "pos_std": round(rmse_pos_std, PRECISION),
        "ori_mean": round(rmse_ori_mean, PRECISION),
        "ori_std": round(rmse_ori_std, PRECISION),
    }


def eval_filter(filter, mc_max, info="", noise="high", folder=""):
    bag_str = "5hz_o5e-4_l2e-2" if noise == "high" else "5hz_o1e-4_l1e-2"
    filenames = [f"{filter}_{info}_mc{mc+1}_{bag_str}" for mc in range(mc_max)]

    results = eval_mc(filenames, folder)
    return results


def create_row(filter, result):
    result_string = f"{filter.upper()}&{result['pos_mean']}$\pm{{{result['pos_std']}}}$&{{{result['ori_mean']}}}$\pm{{{result['ori_std']}}}$&{{{result['comp_mean']}}}$\pm{{{result['comp_std']}}}$\\\\"
    return result_string


def create_double_row(row_label, label_value, filters, results):
    # \multirow{2}{*}{N\_a = 5} & MEDH  & 11.1 + 0.01 & 11.1 + 0.01 & 11.1 + 0.01 \\
    # & NAEDH & 11.1 + 0.01 & 11.1 + 0.01 & 11.1 + 0.01 \\ \hline

    string = ""
    string += f"\\multirow{{2}}{{*}}{{{row_label} = {label_value}}}&"
    string += create_row(filters[0], results[0])
    string += "&"
    string += create_row(filters[1], results[1])
    string += "\\hline"

    return string


def create_general_table(filters, results):
    print("\\begin{tabular}{@{}llll@{}}")
    print("\\hline")
    print("& RMSE pos. (mm) & RMSE ori. (mrad) & TIME (ms) \\\\ \hline ")
    for filter, result in zip(filters, results):
        print(create_row(filter, result))
    print("\\hline")
    print("\\end{tabular}")


def create_double_row_table(filters, results, row_label, label_values):
    print("\\begin{tabular}{@{}lllll@{}}")
    print("\\hline")
    print(" & & RMSE pos. (mm) & RMSE ori. (mrad) & TIME (ms) \\\\ \hline ")
    for label_value, result in zip(label_values, results):
        print(create_double_row(row_label, label_value, filters, result))
    print("\\end{tabular}")


def eval_particle():
    pass


def create_lownoise_general_table():
    mc_max = 20
    noise = "low"
    info = "full"
    folder = "lownoise_all/"
    filters = ["ekf", "medh", "naedh", "amcl"]
    results = [eval_filter(filter, mc_max, info, noise, folder) for filter in filters]
    create_general_table(filters, results)


def create_highnoise_general_table():
    mc_max = 25
    noise = "high"
    info = "full"
    folder = "highnoise_all/"
    filters = ["ekf", "medh", "naedh", "amcl"]
    results = [eval_filter(filter, mc_max, info, noise, folder) for filter in filters]
    create_general_table(filters, results)


def create_highnoise_lambda_table():
    mc_max = 20
    noise = "high"
    folder = "highnoise_lambda/"
    row_label = "$N_\lambda$"
    filters = ["medh", "naedh"]
    label_values = [3, 4, 5, 7, 10, 15, 25]

    results = [
        [
            eval_filter(filter, mc_max, f"_l{label_value}", noise, folder)
            for filter in filters
        ]
        for label_value in label_values
    ]

    create_double_row_table(filters, results, row_label, label_values)


def create_highnoise_particle_table():
    mc_max = 10
    noise = "high"
    folder = "highnoise_particle/"
    row_label = "$N_p$"
    filters = ["medh", "naedh"]
    label_values = [1, 10, 100, 1000, 10000]

    results = [
        [
            eval_filter(filter, mc_max, f"full_p{label_value}", noise, folder)
            for filter in filters
        ]
        for label_value in label_values
    ]

    create_double_row_table(filters, results, row_label, label_values)


def create_rmse_comptime_plot():
    plt.rcParams["text.usetex"] = True
    mc_max_particle = 10
    mc_max_lambda = 20
    mc_max_ekf = 25
    mc_max_amclb = 10
    folder_particle = "highnoise_particle/"
    folder_lambda = "highnoise_lambda/"
    folder_ekf = "highnoise_all/"
    folder_amclb = "highnoise_amcl_basic/"
    particle_nums = [1, 10, 100, 1000, 10000]
    lambda_nums = [3, 4, 5, 7, 10, 15, 25]
    noise = "high"
    filter = "medh"
    particle_results_pos = [
        eval_filter(
            filter, mc_max_particle, f"full_p{particle_num}", noise, folder_particle
        )["pos_mean"]
        for particle_num in particle_nums
    ]
    particle_results_comp = [
        eval_filter(
            filter, mc_max_particle, f"full_p{particle_num}", noise, folder_particle
        )["comp_mean"]
        for particle_num in particle_nums
    ]
    lambda_results_pos = [
        eval_filter(filter, mc_max_particle, f"_l{lambda_num}", noise, folder_lambda)[
            "pos_mean"
        ]
        for lambda_num in lambda_nums
    ]
    lambda_results_comp = [
        eval_filter(filter, mc_max_particle, f"_l{lambda_num}", noise, folder_lambda)[
            "comp_mean"
        ]
        for lambda_num in lambda_nums
    ]
    ekf_result_pos = eval_filter("ekf", mc_max_ekf, "full", noise, folder_ekf)[
        "pos_mean"
    ]
    ekf_result_comp = eval_filter("ekf", mc_max_ekf, "full", noise, folder_ekf)[
        "comp_mean"
    ]
    amclp_result_pos = eval_filter("amcl", mc_max_ekf, "full", noise, folder_ekf)[
        "pos_mean"
    ]
    amclp_result_comp = eval_filter("amcl", mc_max_ekf, "full", noise, folder_ekf)[
        "comp_mean"
    ]
    amclb_result_pos = eval_filter(
        "amcl", mc_max_amclb, "noselective", noise, folder_amclb
    )["pos_mean"]
    amclb_result_comp = eval_filter(
        "amcl", mc_max_amclb, "noselective", noise, folder_amclb
    )["comp_mean"]

    plt.scatter(
        lambda_results_comp,
        lambda_results_pos,
        marker="x",
    )
    plt.scatter(
        particle_results_comp, particle_results_pos, facecolors="none", edgecolors="red"
    )
    plt.scatter(
        ekf_result_comp,
        ekf_result_pos,
        marker="s",
        facecolors="none",
        edgecolors="black",
    )
    plt.scatter(
        amclp_result_comp,
        amclp_result_pos,
        marker="p",
        facecolors="none",
        edgecolors="black",
    )
    plt.scatter(
        amclb_result_comp,
        amclb_result_pos,
        marker="v",
        facecolors="none",
        edgecolors="black",
    )

    for i, lambda_value in enumerate(lambda_nums):
        label_text = rf"$N_\lambda = {lambda_value}$"
        plt.annotate(
            label_text, (lambda_results_comp[i] - 2, lambda_results_pos[i] + 0.2)
        )

    for i, particle_value in enumerate(particle_nums):
        label_text = rf"$N_p = {particle_value}$"
        plt.annotate(
            label_text, (particle_results_comp[i] - 2, particle_results_pos[i] + 0.2)
        )

    plt.show()


create_rmse_comptime_plot()
# create_highnoise_general_table()
# create_highnoise_particle_table()
# create_highnoise_general_table()
# create_lownoise_general_table()
