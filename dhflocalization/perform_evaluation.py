from dhflocalization.rawdata import ResultLoader, ConfigImporter, RawDataLoader
from dhflocalization.gridmap import GridMap
from dhflocalization.visualization import TrackPlotter
from dhflocalization.evaluator import metrics, data_association
from dhflocalization.rawdata import optitrack_reader
from dhflocalization.rawdata import YamlWriter

from scipy.spatial.transform import Rotation
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt


def from_data(true_states, filtered_results, do_plot=False, map_filename=None):
    (err_mean_sqare, err_mean_abs, std) = metrics.eval(true_states, filtered_results)

    print(err_mean_sqare)
    print("---")
    print(err_mean_abs)
    print("---")
    print(std)

    if not do_plot or not map_filename:
        return

    ogm = GridMap.load_map_from_config(map_filename)
    track_plotter = TrackPlotter(background_map=ogm)
    track_plotter.plot_results(true_states, filtered_results)


def from_file(results_filename, do_plot=False):
    filtered_results = ResultLoader.load(results_filename)
    meta_data = ConfigImporter.read(results_filename)
    simulation_data = RawDataLoader.load_from_json(meta_data["cfg_simu_data_filename"])

    # stds = calc_empiric_std(filtered_results["medh"]["track"])

    true_states = simulation_data.x_true
    (err_mean_squares, err_mean_abss, err_stds) = metrics.eval(
        true_states, filtered_results
    )

    # update the original config file with the results
    metrics_dict = {
        "RMSE": err_mean_squares,
        "MAE": err_mean_abss,
        "STD": err_stds,
    }
    YamlWriter().updateFile(
        payload=metrics_dict,
        filename=results_filename,
    )

    print(err_mean_squares)
    print("---")
    print(err_mean_abss)
    print("---")
    print(err_stds)

    if not do_plot:
        return

    ogm = GridMap.load_map_from_config(meta_data["cfg_map_config_filename"])
    track_plotter = TrackPlotter(background_map=ogm)
    track_plotter.plot_results(simulation_data.x_true, filtered_results)


def calc_empiric_std(filtered_track):
    stds = np.zeros((filtered_track.timesteps(), 3))
    for timestep, state in enumerate(filtered_track):
        stds[timestep, :] = np.std(state.state_vectors, axis=0)

    return stds


def compare_filtered_with_optitrack(dhf_fn, amcl_fn, optitrack_fn):
    optitrack_data = optitrack_reader.load(optitrack_fn)
    turtlebot_data = optitrack_data.rigid_bodies["turtlebot"]
    optitrack_stamps = [
        time + optitrack_data.start_time_epoch for time in turtlebot_data.times
    ]
    turtlebot_positions = turtlebot_data.positions
    turtlebot_rotations_quat = turtlebot_data.rotations
    turtlebot_rotations = Rotation.from_quat(turtlebot_rotations_quat).as_euler("zxy")

    optitrack_state = np.asarray(
        [
            [pos[0], -pos[2], rot[0]]
            for (pos, rot) in zip(turtlebot_positions, turtlebot_rotations)
        ]
    )

    dhf_filtered_data = RawDataLoader.load_from_json(dhf_fn)
    amcl_filtered_data = RawDataLoader.load_from_json(amcl_fn)

    dhf_stamps = list(dhf_filtered_data.times)
    amcl_stamps = list(amcl_filtered_data.times)

    dhf_state = dhf_filtered_data.x_true
    amcl_state = amcl_filtered_data.x_amcl

    dhf_state[:, 2] = abs(np.arctan2(np.sin(dhf_state[:, 2]), np.cos(dhf_state[:, 2])))
    amcl_state[:, 2] = abs(amcl_state[:, 2])

    optitrack_dhf_indices = data_association.optitrack_with_filter(
        optitrack_stamps, dhf_stamps
    )
    optitrack_amcl_indices = data_association.optitrack_with_filter(
        optitrack_stamps, amcl_stamps
    )

    dhf_results = metrics.eval(
        {"dhf": dhf_state},
        {"true": optitrack_state[optitrack_dhf_indices, :]},
        return_results=True,
    )
    amcl_results = metrics.eval(
        {"amcl": amcl_state},
        {"true": optitrack_state[optitrack_amcl_indices, :]},
        return_results=True,
    )

    print("----")
    print(dhf_results[0], amcl_results[0])
    print("----")
    print(dhf_results[1], amcl_results[1])
    print("----")
    print(dhf_results[2], amcl_results[2])
    print("----")


def eval_dhf(mc_filepaths, simu=False):
    if simu is False:
        dhf_stamps = list(RawDataLoader.load_from_json("real_take01").times)
        optitrack_stamps, optitrack_state = process_optitrack("take01")
        optitrack_dhf_indices = data_association.optitrack_with_filter(
            optitrack_stamps, dhf_stamps
        )
        ground_truth = optitrack_state[optitrack_dhf_indices, :]
    else:
        ground_truth = RawDataLoader.load_from_json("house_true").x_true

    rmse_pos_mc = []
    rmse_ori_mc = []
    comptime_mc = []
    for filepath in mc_filepaths:
        dhf_states = load_from_json(filepath)
        results = metrics.eval(
            {
                "ekf": np.array(dhf_states["ekf"]["state"]),
                "edh": np.array(dhf_states["edh"]["state"]),
                "aedh": np.array(dhf_states["aedh"]["state"]),
                "naedh": np.array(dhf_states["naedh"]["state"]),
                "ledh": np.array(dhf_states["ledh"]["state"]),
                "cledh": np.array(dhf_states["cledh"]["state"]),
            },
            {"true": ground_truth},
            return_results=True,
        )
        comptimes = {
            "ekf": dhf_states["ekf"]["comptime"],
            "edh": dhf_states["edh"]["comptime"],
            "aedh": dhf_states["aedh"]["comptime"],
            "naedh": dhf_states["naedh"]["comptime"],
            "ledh": dhf_states["ledh"]["comptime"],
            "cledh": dhf_states["cledh"]["comptime"],
        }
        rmse_pos_mc.append(results[0]["position"])
        rmse_ori_mc.append(results[0]["orientation"])
        comptime_mc.append(comptimes)

    mc_results = {}
    for filter_type in rmse_pos_mc[0].keys():
        filter_rmse_pos_mc = []
        filter_rmse_ori_mc = []
        filter_comptime_mc = []
        for mc_run in range(len(rmse_pos_mc)):
            filter_rmse_pos_mc.append(rmse_pos_mc[mc_run][filter_type])
            filter_rmse_ori_mc.append(rmse_ori_mc[mc_run][filter_type])
            filter_comptime_mc.append(comptime_mc[mc_run][filter_type])
        mc_results[filter_type] = {}
        mc_results[filter_type]["pos"] = np.array(filter_rmse_pos_mc).mean()
        mc_results[filter_type]["ori"] = np.array(filter_rmse_ori_mc).mean()
        mc_results[filter_type]["comptime"] = np.array(filter_comptime_mc).mean()

    return mc_results


def eval_amcl():
    filenames = ["real/amcl/amcl_take01_" + str(mc + 1) for mc in range(20)]
    optitrack_stamps, optitrack_state = process_optitrack("take01")
    rmse_pos_mc = []
    rmse_ori_mc = []
    comptime_mc = []
    for filename in filenames:
        amcl_stamps = []
        amcl_states = []
        amcl_comptimes = []

        amcl_result = load_from_json(filename)

        for step in amcl_result:
            amcl_stamps.append(step["t"]),
            amcl_states.append(step["amcl"]),
            amcl_comptimes.append(step["comptime"]),

        optitrack_amcl_indices = data_association.optitrack_with_filter(
            optitrack_stamps, amcl_stamps
        )
        amcl_results = metrics.eval(
            {"amcl": np.array(amcl_states)},
            {"true": optitrack_state[optitrack_amcl_indices, :]},
            return_results=True,
        )
        rmse_pos = amcl_results[0]["position"]["amcl"]
        rmse_ori = amcl_results[0]["orientation"]["amcl"]
        comptime_per_step = np.array(amcl_comptimes).mean() / 10**3  # millisec
        rmse_pos_mc.append(rmse_pos)
        rmse_ori_mc.append(rmse_ori)
        comptime_mc.append(comptime_per_step)

    return {
        "pos": np.array(rmse_pos_mc).mean(),
        "ori": np.array(rmse_ori_mc).mean(),
        "comptime": np.array(comptime_mc).mean(),
    }


def process_optitrack(optitrack_fn):
    optitrack_data = optitrack_reader.load(optitrack_fn)
    turtlebot_data = optitrack_data.rigid_bodies["turtlebot"]
    optitrack_stamps = [
        time + optitrack_data.start_time_epoch for time in turtlebot_data.times
    ]

    turtlebot_positions = turtlebot_data.positions
    turtlebot_rotations_quat = turtlebot_data.rotations
    turtlebot_rotations = Rotation.from_quat(turtlebot_rotations_quat).as_euler("zxy")

    optitrack_state = np.asarray(
        [
            [pos[0], -pos[2], rot[2]]
            for (pos, rot) in zip(turtlebot_positions, turtlebot_rotations)
        ]
    )

    return optitrack_stamps, optitrack_state


def load_from_json(result_path):
    relative_path = "resources/results/" + result_path + ".json"
    base_path = Path(__file__).parent
    file_path = (base_path / relative_path).resolve()
    try:
        json_file = open(
            file_path,
        )
    except AttributeError:
        raise ValueError("File not found at {}".format(file_path))

    data = json.load(json_file)
    return data["data"]


def eval_simu_amcl():
    filenames = ["simu/amcl/amcl_simu_take01_" + str(mc + 1) for mc in range(20)]
    ground_truth = RawDataLoader.load_from_json("simu_take01").x_true

    rmse_pos_mc = []
    rmse_ori_mc = []
    comptime_mc = []
    for filename in filenames:
        amcl_stamps = []
        amcl_states = []
        amcl_comptimes = []

        amcl_result = load_from_json(filename)

        for step in amcl_result:
            amcl_stamps.append(step["t"]),
            amcl_states.append(step["amcl"]),
            amcl_comptimes.append(step["comptime"]),

        amcl_results = metrics.eval(
            {"amcl": np.array(amcl_states)},
            {"true": ground_truth[:-1]},
            return_results=True,
        )
        rmse_pos = amcl_results[0]["position"]["amcl"]
        rmse_ori = amcl_results[0]["orientation"]["amcl"]
        comptime_per_step = np.array(amcl_comptimes).mean() / 10**3  # millisec
        rmse_pos_mc.append(rmse_pos)
        rmse_ori_mc.append(rmse_ori)
        comptime_mc.append(comptime_per_step)

    return {
        "pos": np.array(rmse_pos_mc).mean(),
        "ori": np.array(rmse_ori_mc).mean(),
        "comptime": np.array(comptime_mc).mean(),
    }


def eval_simu_edh_naedh():
    runs = []
    for lamb in range(10):
        mc_filenames = [
            "simu/edh-naedh/dhf_take01_ehd-naedh_simugmapping_p100_l"
            + str(lamb + 1)
            + "_"
            + str(mc + 1)
            for mc in range(20)
        ]
        runs.append(eval_dhf(mc_filenames, simu=True))

    rmse_pos_edh = [run["edh"]["pos"] for run in runs]
    rmse_pos_naedh = [run["naedh"]["pos"] for run in runs]

    rmse_ori_edh = [run["edh"]["ori"] for run in runs]
    rmse_ori_naedh = [run["naedh"]["ori"] for run in runs]

    comp_edh = [run["edh"]["comptime"] * 1000 for run in runs]
    comp_naedh = [run["naedh"]["comptime"] * 1000 for run in runs]

    # rmse_ori_edh = [run["edh"]["ori"] for run in runs]
    # rmse_ori_naedh = [run["naedh"]["ori"] for run in runs]

    # comp_edh = [run["edh"]["comptime"] for run in runs]
    # comp_naedh = [run["naedh"]["comptime"] for run in runs]

    plt.rcParams["text.usetex"] = True
    fig, axs = plt.subplots(3)
    axs[0].plot(list(range(1, 11)), rmse_pos_edh, "o")
    axs[0].plot(list(range(1, 11)), rmse_pos_naedh, "x")

    axs[1].plot(list(range(1, 11)), rmse_ori_edh, "o")
    axs[1].plot(list(range(1, 11)), rmse_ori_naedh, "x")
    axs[2].plot(list(range(1, 11)), comp_edh, "o")
    axs[2].plot(list(range(1, 11)), comp_naedh, "x")
    axs[0].legend(["EDH", "NA-EDH"])

    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    axs[2].set_xlabel(r"$N_{\lambda}$")
    axs[0].set_ylabel

    axs[0].set_ylabel(r"$\mathrm{RMSE\ pos.\ (\mathrm{m})}$")
    axs[1].set_ylabel(r"$\mathrm{RMSE\ ori.\ (\mathrm{rad})}$")
    axs[2].set_ylabel(r"$\mathrm{comp. time}\ (\mathrm{ms})$")

    plt.show()
    plt.savefig("simu_edh_naedh.eps")
    print("hi")
    print("hi")


def eval_simu_ledh_cledh():
    cledh_runs = []
    for cluster_num in [20, 5]:  # 20,10,5 clusters
        cledh_mc_filenames = [
            "simu/ledh-cledh/dhf_take01_cledh_szimugmapping_p100_c"
            + str(cluster_num)
            + "_"
            + str(mc + 1)
            for mc in range(5)
        ]
        cledh_runs.append(eval_dhf(cledh_mc_filenames, simu=True))

    # ledh_runs = []
    # ledh_mc_filenames = [
    #     "simu/ledh-cledh/dhf_take01_ledh_simugmapping_p30_" + str(mc + 1)
    #     for mc in range(5)
    # ]
    # ledh_runs.append(eval_dhf(ledh_mc_filenames, simu=True))

    rmse_pos_cledh = [run["cledh"]["pos"] for run in cledh_runs]
    # rmse_pos_ledh = [run["ledh"]["pos"] for run in ledh_runs]

    rmse_ori_cledh = [run["cledh"]["ori"] for run in cledh_runs]
    # rmse_ori_ledh = [run["ledh"]["ori"] for run in ledh_runs]

    comp_cledh = [run["cledh"]["comptime"] * 1000 for run in cledh_runs]
    # comp_ledh = [run["ledh"]["comptime"] * 1000 for run in ledh_runs]

    print(rmse_pos_cledh)
    # comp_ledh = [run["ledh"]["comptime"] for run in ledh_runs]

    # plt.plot([20, 10, 5], rmse_pos_cledh, "x")
    # plt.plot(30, rmse_pos_ledh, "x")
    plt.show()


def eval_ledh_cledh():
    cledh_runs = []
    for cluster_num in [20, 5]:  # 20,10,5 clusters
        cledh_mc_filenames = [
            "real/ledh-cledh/dhf_take01_cledh_sztakigmapping_p100_c"
            + str(cluster_num)
            + "_"
            + str(mc + 1)
            for mc in range(5)
        ]
        cledh_runs.append(eval_dhf(cledh_mc_filenames, simu=False))

    # ledh_runs = []
    # ledh_mc_filenames = [
    #     "real/ledh-cledh/dhf_take01_ledh_sztakigmapping_p30_" + str(mc + 1)
    #     for mc in range(5)
    # ]
    # ledh_runs.append(eval_dhf(ledh_mc_filenames, simu=False))

    rmse_pos_cledh = [run["cledh"]["pos"] for run in cledh_runs]
    # rmse_pos_ledh = [run["ledh"]["pos"] for run in ledh_runs]

    rmse_ori_cledh = [run["cledh"]["ori"] for run in cledh_runs]
    # rmse_ori_ledh = [run["ledh"]["ori"] for run in ledh_runs]

    comp_cledh = [run["cledh"]["comptime"] * 1000 for run in cledh_runs]
    # comp_ledh = [run["ledh"]["comptime"] * 1000 for run in ledh_runs]

    print(rmse_pos_cledh)
    # comp_ledh = [run["ledh"]["comptime"] for run in ledh_runs]

    # plt.plot([20, 10, 5], rmse_pos_cledh, "x")
    # plt.plot(30, rmse_pos_ledh, "x")
    plt.show()


def eval_edh_naedh():
    runs = []
    for lamb in range(10):
        mc_filenames = [
            "real/edh-naedh/dhf_take01_edh-naedh_sztakigmapping_p100_l"
            + str(lamb + 1)
            + "_"
            + str(mc + 1)
            for mc in range(20)
        ]
        runs.append(eval_dhf(mc_filenames, simu=False))

    rmse_pos_edh = [run["edh"]["pos"] for run in runs]
    rmse_pos_naedh = [run["naedh"]["pos"] for run in runs]

    rmse_ori_edh = [run["edh"]["ori"] for run in runs]
    rmse_ori_naedh = [run["naedh"]["ori"] for run in runs]

    comp_edh = [run["edh"]["comptime"] * 1000 for run in runs]
    comp_naedh = [run["naedh"]["comptime"] * 1000 for run in runs]

    # rmse_ori_edh = [run["edh"]["ori"] for run in runs]
    # rmse_ori_naedh = [run["naedh"]["ori"] for run in runs]

    # comp_edh = [run["edh"]["comptime"] for run in runs]
    # comp_naedh = [run["naedh"]["comptime"] for run in runs]

    plt.rcParams["text.usetex"] = True
    fig, axs = plt.subplots(3)
    axs[0].plot(list(range(1, 11)), rmse_pos_edh, "o")
    axs[0].plot(list(range(1, 11)), rmse_pos_naedh, "x")

    axs[1].plot(list(range(1, 11)), rmse_ori_edh, "o")
    axs[1].plot(list(range(1, 11)), rmse_ori_naedh, "x")
    axs[2].plot(list(range(1, 11)), comp_edh, "o")
    axs[2].plot(list(range(1, 11)), comp_naedh, "x")
    axs[0].legend(["EDH", "NA-EDH"])

    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[2].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    axs[2].set_xlabel(r"$N_{\lambda}$")
    axs[0].set_ylabel

    axs[0].set_ylabel(r"$\mathrm{RMSE\ pos.\ (\mathrm{m})}$")
    axs[1].set_ylabel(r"$\mathrm{RMSE\ ori.\ (\mathrm{rad})}$")
    axs[2].set_ylabel(r"$\mathrm{comp. time}\ (\mathrm{ms})$")

    plt.show()
    plt.savefig("real_edh_naedh.eps")


if __name__ == "__main__":
    results_filename = "23-04-26T170729"
    from_file(results_filename, do_plot=True)
