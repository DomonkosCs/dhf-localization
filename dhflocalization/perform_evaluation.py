from dhflocalization.rawdata import resultLoader, ConfigImporter, RawDataLoader
from dhflocalization.gridmap import GridMap
from dhflocalization.gridmap import PgmProcesser
from dhflocalization.visualization import TrackPlotter
from dhflocalization.evaluator import metrics as metrics
from dhflocalization.rawdata import optitrack_reader

import matplotlib.pyplot as plt


def main(results_filename):
    # load results from the pickle file
    results = resultLoader.load(results_filename)
    config = ConfigImporter.importData(results_filename)
    # (err_mean_sqare, err_mean_abs, std) = metrics.eval(
    #     filtered_states=results[0],
    #     reference_states=results[1],
    #     export_filename=results_filename,
    #     return_results=True,
    # )

    # print(err_mean_sqare)
    # print(err_mean_abs)
    # print(std)

    map_fn = config["cfg_map_filename"]

    ogm = GridMap.load_grid_map_from_array(
        PgmProcesser.read_pgm(map_fn), 0.05, 10, 10.05
    )
    track_plotter = TrackPlotter(background_map=ogm)
    track_plotter.plot_tracks(results[0], results[1])


def compare_filtered_with_optitrack(dhf_fn, amcl_fn, optitrack_fn):
    optitrack_data = optitrack_reader.load(optitrack_fn)
    turtlebot_data = optitrack_data.rigid_bodies["turtlebot"]
    dhf_filtered_data = RawDataLoader.load_from_json(dhf_fn)
    amcl_filtered_data = RawDataLoader.load_from_json(amcl_fn)

    optitrack_times = [
        time + optitrack_data.start_time_epoch for time in turtlebot_data.times
    ]
    dhf_filtered_times = list(dhf_filtered_data.times)
    amcl_filtered_times = list(amcl_filtered_data.times)

    dhf_filter_pos_x = [pos[0] for pos in list(dhf_filtered_data.x_true)]
    optitrack_pos_x = [pos[0] for pos in turtlebot_data.positions]
    dhf_filter_pos_y = [pos[1] for pos in list(dhf_filtered_data.x_true)]
    optitrack_pos_y = [-pos[2] for pos in turtlebot_data.positions]
    amcl_filter_pos_x = [pos[0] for pos in list(amcl_filtered_data.x_amcl)]
    amcl_filter_pos_y = [pos[1] for pos in list(amcl_filtered_data.x_amcl)]

    plt.scatter(dhf_filter_pos_x, dhf_filter_pos_y)
    plt.scatter(amcl_filter_pos_x, amcl_filter_pos_y)
    plt.scatter(optitrack_pos_x, optitrack_pos_y)
    plt.legend(["dhf", "amcl", "opti"])

    plt.figure()
    plt.plot(dhf_filtered_times, dhf_filter_pos_x)
    plt.plot(amcl_filtered_times, amcl_filter_pos_x)
    plt.plot(optitrack_times, optitrack_pos_x)
    plt.legend()
    plt.figure()
    plt.plot(dhf_filtered_times, dhf_filter_pos_y)
    plt.plot(amcl_filtered_times, amcl_filter_pos_y)
    plt.plot(optitrack_times, optitrack_pos_y)
    plt.legend()

    plt.show()

    print("hi")


if __name__ == "__main__":
    compare_filtered_with_optitrack("real_3_corr", "amcl", "take01")
    # results_filename = "22-09-30T084254"
    # main(results_filename)
