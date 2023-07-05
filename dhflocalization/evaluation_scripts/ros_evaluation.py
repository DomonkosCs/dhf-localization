import json
import numpy as np
import matplotlib.pyplot as plt
from dhflocalization.rawdata.filehandler import FileHandler
from dhflocalization.evaluator import metrics
from dhflocalization.gridmap import GridMap
from dhflocalization.visualization import TrackPlotter
from matplotlib.ticker import FormatStrFormatter, FixedLocator


def eval_one(filename, folder=""):
    relative_path = "../resources/results/final/" + folder + filename + ".json"
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


def create_lownoise_general_table():
    mc_max = 10
    noise = "low"
    info = "full"
    folder = "low_general/"
    filters = ["ekf", "medh", "naedh", "amcl", "amclb"]
    results = [eval_filter(filter, mc_max, info, noise, folder) for filter in filters]
    create_general_table(filters, results)


def create_highnoise_general_table():
    mc_max = 10
    noise = "high"
    info = "full"
    folder = "high_general/"
    filters = ["ekf", "amclb", "amcl"]
    results = [eval_filter(filter, mc_max, info, noise, folder) for filter in filters]
    create_general_table(filters, results)


def create_highnoise_lambda_table():
    mc_max = 50
    noise = "high"
    folder = "lambda/"
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
    folder = "particle/"
    row_label = "$N_p$"
    filters = ["medh", "naedh"]
    label_values = [1, 10, 1000, 10000]

    results = [
        [
            eval_filter(filter, mc_max, f"full_p{label_value}", noise, folder)
            for filter in filters
        ]
        for label_value in label_values
    ]

    create_double_row_table(filters, results, row_label, label_values)


def generate_plot_data():
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

    ## evaluation
    # particle
    particle_results_medh = [
        eval_filter(
            "medh", mc_max_particle, f"full_p{particle_num}", noise, folder_particle
        )
        for particle_num in particle_nums
    ]
    # lambda
    lambda_results_medh = [
        eval_filter("medh", mc_max_lambda, f"_l{lambda_num}", noise, folder_lambda)
        for lambda_num in lambda_nums
    ]
    lambda_results_naedh = [
        eval_filter("naedh", mc_max_lambda, f"_l{lambda_num}", noise, folder_lambda)
        for lambda_num in lambda_nums
    ]
    # baseline
    ekf_result = eval_filter("ekf", mc_max_ekf, "full", noise, folder_ekf)
    amclp_result = eval_filter("amcl", mc_max_ekf, "full", noise, folder_ekf)  # amcl+
    amclb_result = eval_filter(
        "amcl", mc_max_amclb, "noselective", noise, folder_amclb
    )  # amcl basic

    ## data extraction
    particle_results_medh_pos = [result["pos_mean"] for result in particle_results_medh]
    particle_results_medh_ori = [result["ori_mean"] for result in particle_results_medh]
    particle_results_medh_comp = [
        result["comp_mean"] for result in particle_results_medh
    ]

    lambda_results_medh_pos = [result["pos_mean"] for result in lambda_results_medh]
    lambda_results_medh_ori = [result["ori_mean"] for result in lambda_results_medh]
    lambda_results_medh_comp = [result["comp_mean"] for result in lambda_results_medh]

    lambda_results_naedh_pos = [result["pos_mean"] for result in lambda_results_naedh]
    lambda_results_naedh_ori = [result["ori_mean"] for result in lambda_results_naedh]
    lambda_results_naedh_comp = [result["comp_mean"] for result in lambda_results_naedh]

    ekf_result_pos = ekf_result["pos_mean"]
    ekf_result_ori = ekf_result["ori_mean"]
    ekf_result_comp = ekf_result["comp_mean"]

    amclp_result_pos = amclp_result["pos_mean"]
    amclp_result_ori = amclp_result["ori_mean"]
    amclp_result_comp = amclp_result["comp_mean"]

    amclb_result_pos = amclb_result["pos_mean"]
    amclb_result_ori = amclb_result["ori_mean"]
    amclb_result_comp = amclb_result["comp_mean"]

    return (
        particle_results_medh_pos,
        particle_results_medh_ori,
        particle_results_medh_comp,
        lambda_results_medh_pos,
        lambda_results_medh_ori,
        lambda_results_medh_comp,
        lambda_results_naedh_pos,
        lambda_results_naedh_ori,
        lambda_results_naedh_comp,
        ekf_result_pos,
        ekf_result_ori,
        ekf_result_comp,
        amclp_result_pos,
        amclp_result_ori,
        amclp_result_comp,
        amclb_result_pos,
        amclb_result_ori,
        amclb_result_comp,
    )


def create_rmse_pos_comptime_plot():
    lambda_nums = [3, 4, 5, 7, 10, 15, 25]

    particle_results_medh_pos = [20.4, 18.78, 15.63, 15.35, 14.43]
    particle_results_medh_comp = [48.74, 48.9, 49.78, 51.33, 62.98]
    lambda_results_medh_pos = [20.51, 19.71, 19.13, 17.47, 15.63, 14.88, 14.77]
    lambda_results_medh_comp = [20.79, 24.96, 28.87, 36.99, 49.78, 61.67, 66.96]
    lambda_results_naedh_pos = [24.28, 23.62, 21.08, 19.46, 17.01, 16.92, 16.08]
    lambda_results_naedh_comp = [21.13, 25.57, 29.64, 37.01, 49.62, 61.0, 65.1]
    ekf_result_pos = 44.61
    ekf_result_comp = 6.23
    amclp_result_pos = 61.11
    amclp_result_comp = 38.29
    amclb_result_pos = 111.13
    amclb_result_comp = 5.62

    ## plotting
    # setup
    plt.rcParams["text.usetex"] = True
    plt.rcParams.update({"font.size": 15, "font.family": "serif"})

    time_lim_main = [0, 75]
    rmse_lim_main = [10, 150]

    # main plot setup
    fig, ax1 = plt.subplots()
    fig.set_figheight(7.5)
    fig.set_figwidth(7)
    ax1.set_yscale("log")
    ax1.grid(which="both")

    ax1.set_xlabel(r"TIME (ms)")
    ax1.set_ylabel(r"RMSE pos. (mm)")
    ax1.set_xlim(time_lim_main)
    ax1.set_ylim(rmse_lim_main)

    # ticks
    ax1.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    minor_locator = ax1.yaxis.get_minor_locator()
    minor_ticks = minor_locator()
    minor_ticks = np.append(minor_ticks, 120)
    updated_minor_locator = FixedLocator(minor_ticks)
    ax1.yaxis.set_minor_locator(updated_minor_locator)

    # markers
    lambda_marker_sizes = [l * 7 + 5 for l in lambda_nums]
    particle_marker_sizes = [20, 50, 70, 90, 110]
    marker_line_width = 2

    inset_marker_size = 30

    # main plots
    medh_lambda_handle = ax1.scatter(
        lambda_results_medh_comp,
        lambda_results_medh_pos,
        marker="s",
        facecolors="none",
        edgecolors="tab:green",
        s=lambda_marker_sizes,
        linewidths=marker_line_width,
        label="MEDH $N_\lambda$",
    )
    medh_particle_handle = ax1.scatter(
        particle_results_medh_comp,
        particle_results_medh_pos,
        facecolors="none",
        edgecolors="tab:purple",
        s=particle_marker_sizes,
        linewidths=marker_line_width,
        label="MEDH $N_p$",
    )
    naedh_lambda_handle = ax1.scatter(
        lambda_results_naedh_comp,
        lambda_results_naedh_pos,
        marker="s",
        facecolors="none",
        edgecolors="tab:blue",
        s=lambda_marker_sizes,
        linewidths=marker_line_width,
        label="NAEDH $N_\lambda$",
    )
    ekf_handle = ax1.scatter(
        ekf_result_comp,
        ekf_result_pos,
        marker="*",
        facecolors="none",
        edgecolors="black",
        s=inset_marker_size * 3,
        linewidths=1.5,
        label="EKF",
    )
    amclp_handle = ax1.scatter(
        amclp_result_comp,
        amclp_result_pos,
        marker="d",
        facecolors="none",
        edgecolors="black",
        s=inset_marker_size * 3,
        linewidths=1.5,
        label=r"$\mathrm{AMCL}+$",
    )
    amclb_handle = ax1.scatter(
        amclb_result_comp,
        amclb_result_pos,
        marker="v",
        facecolors="none",
        edgecolors="black",
        s=inset_marker_size * 3,
        linewidths=1.5,
        label="AMCL",
    )

    ax1.legend(
        handles=[
            medh_particle_handle,
            medh_lambda_handle,
            naedh_lambda_handle,
            ekf_handle,
            amclb_handle,
            amclp_handle,
        ]
    )
    plt.savefig("rmse_pos_time_final.eps")
    plt.show()


def create_rmse_ori_comptime_plot():
    ## evaluation params
    lambda_nums = [3, 4, 5, 7, 10, 15, 25]

    particle_results_medh_ori = [22.88, 22.83, 22.67, 22.42, 21.76]
    particle_results_medh_comp = [48.74, 48.9, 49.78, 51.33, 62.98]
    lambda_results_medh_ori = [25.96, 25.91, 25.7, 24.36, 22.67, 22.25, 22.07]
    lambda_results_medh_comp = [20.79, 24.96, 28.87, 36.99, 49.78, 61.67, 66.96]
    lambda_results_naedh_ori = [25.25, 25.68, 25.62, 24.58, 22.63, 22.28, 22.3]
    lambda_results_naedh_comp = [21.13, 25.57, 29.64, 37.01, 49.62, 61.0, 65.1]
    ekf_result_ori = 28.79
    ekf_result_comp = 6.23
    amclp_result_ori = 110.88
    amclp_result_comp = 38.29
    amclb_result_ori = 148.06
    amclb_result_comp = 5.62

    ## plotting
    # setup
    plt.rcParams["text.usetex"] = True
    plt.rcParams.update({"font.size": 15, "font.family": "serif"})

    rmse_lim_main = [20, 160]
    time_lim_main = [0, 75]

    # main plot setup
    fig, ax1 = plt.subplots()
    fig.set_figheight(7.5)
    fig.set_figwidth(7)
    ax1.set_yscale("log")
    ax1.grid(which="both")

    ax1.set_xlabel(r"TIME (ms)")
    ax1.set_ylabel(r"RMSE ori. (mrad)")
    ax1.set_xlim(time_lim_main)
    ax1.set_ylim(rmse_lim_main)

    # ticks
    ax1.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    minor_locator = ax1.yaxis.get_minor_locator()
    minor_ticks = minor_locator()
    minor_ticks = np.append(minor_ticks, 150)
    updated_minor_locator = FixedLocator(minor_ticks)
    ax1.yaxis.set_minor_locator(updated_minor_locator)

    # markers
    lambda_marker_sizes = [l * 7 + 5 for l in lambda_nums]
    particle_marker_sizes = [20, 50, 70, 90, 110]
    marker_line_width = 2

    inset_marker_size = 30

    # main plots
    medh_lambda_handle = ax1.scatter(
        lambda_results_medh_comp,
        lambda_results_medh_ori,
        marker="s",
        facecolors="none",
        edgecolors="tab:green",
        linewidths=marker_line_width,
        s=lambda_marker_sizes,
        label="MEDH $N_\lambda$",
    )
    medh_particle_handle = ax1.scatter(
        particle_results_medh_comp,
        particle_results_medh_ori,
        facecolors="none",
        edgecolors="tab:purple",
        linewidths=marker_line_width,
        s=particle_marker_sizes,
        label="MEDH $N_p$",
    )
    naedh_lambda_handle = ax1.scatter(
        lambda_results_naedh_comp,
        lambda_results_naedh_ori,
        marker="s",
        facecolors="none",
        edgecolors="tab:blue",
        linewidths=marker_line_width,
        s=lambda_marker_sizes,
        label="NAEDH $N_\lambda$",
    )
    ekf_handle = ax1.scatter(
        ekf_result_comp,
        ekf_result_ori,
        marker="*",
        facecolors="none",
        edgecolors="black",
        s=inset_marker_size * 3,
        linewidths=1.5,
        label="EKF",
    )
    amclp_handle = ax1.scatter(
        amclp_result_comp,
        amclp_result_ori,
        marker="d",
        facecolors="none",
        edgecolors="black",
        s=inset_marker_size * 3,
        linewidths=1.5,
        label=r"$\mathrm{AMCL}+$",
    )
    amclb_handle = ax1.scatter(
        amclb_result_comp,
        amclb_result_ori,
        marker="v",
        facecolors="none",
        edgecolors="black",
        s=inset_marker_size * 3,
        linewidths=1.5,
        label="AMCL",
    )

    ax1.legend(
        handles=[
            medh_particle_handle,
            medh_lambda_handle,
            naedh_lambda_handle,
            ekf_handle,
            amclb_handle,
            amclp_handle,
        ]
    )
    plt.savefig("rmse_ori_time_final.eps")
    plt.show()


# create_rmse_pos_comptime_plot()
create_rmse_ori_comptime_plot()
# create_highnoise_particle_table()
# create_highnoise_general_table()
# create_lownoise_general_table()
# create_highnoise_lambda_table()
