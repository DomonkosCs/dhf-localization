from dhflocalization.rawdata import resultLoader, ConfigImporter, RawDataLoader
from dhflocalization.gridmap import GridMap
from dhflocalization.gridmap import PgmProcesser
from dhflocalization.visualization import TrackPlotter
from dhflocalization.evaluator import metrics, data_association
from dhflocalization.rawdata import optitrack_reader

from scipy.spatial.transform import Rotation
import numpy as np
import math


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
    optitrack_stamps = [
        time + optitrack_data.start_time_epoch for time in turtlebot_data.times
    ]
    turtlebot_positions = turtlebot_data.positions
    turtlebot_rotations_quat = turtlebot_data.rotations
    turtlebot_rotations = Rotation.from_quat(turtlebot_rotations_quat).as_euler("zxz")

    optitrack_state = np.asarray(
        [
            [pos[0], -pos[2], rot[1]]
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


if __name__ == "__main__":
    compare_filtered_with_optitrack("edh_take01", "amcl_take01", "take01")
    # results_filename = "22-09-30T084254"
    # main(results_filename)
