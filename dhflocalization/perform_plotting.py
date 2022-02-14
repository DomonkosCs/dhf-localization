from rawdata import resultLoader
import numpy as np
from gridmap import GridMap
from gridmap import PgmProcesser
from visualization import TrackPlotter
import matplotlib.pyplot as plt


def main(results_filename):
    def calcPoseFromStateArray(filtered_states, reference_states):
        filtered_poses = {}
        for key, value in filtered_states.items():
            filtered_poses[key] = np.array([state.pose for state in value])
        reference_poses = {}
        for key, value in reference_states.items():
            reference_poses[key] = np.array([state.pose for state in value])

        return filtered_poses, reference_poses

    def calcErrorMetrics(filtered_poses, reference_poses):
        def normalize_angle(angle):
            return np.arctan2(np.sin(angle), np.cos(angle))

        def calc_angle_diff(a, b):
            a = normalize_angle(a)
            b = normalize_angle(b)
            d1 = a - b
            d2 = 2 * np.pi - abs(d1)
            if d1 > 0:
                d2 *= -1.0
            if abs(d1) < abs(d2):
                return d1
            else:
                return d2

        def angle_set_diff(set_1, set_2):
            return [calc_angle_diff(a, b) for a, b in zip(set_1, set_2)]

        err_mean_sqare = {"position": {}, "orientation": {}}
        for key, value in filtered_poses.items():
            value = np.array(value)
            err_mean_sqare["position"][key] = np.sqrt(
                np.mean(
                    np.linalg.norm(
                        reference_poses["true"][:, :-1] - value[:, :-1], axis=1
                    )
                    ** 2
                )
            )
            err_mean_sqare["orientation"][key] = np.sqrt(
                np.mean(
                    np.array(angle_set_diff(reference_poses["true"][:, 2], value[:, 2]))
                    ** 2
                )
            )

        err_mean_abs = {"position": {}, "orientation": {}}
        for key, value in filtered_poses.items():
            err_mean_abs["position"][key] = np.mean(
                np.linalg.norm(reference_poses["true"][:, :-1] - value[:, :-1], axis=1)
            )
            err_mean_abs["orientation"][key] = np.mean(
                np.abs(
                    np.array(angle_set_diff(reference_poses["true"][:, 2], value[:, 2]))
                )
            )

        std = {"position": {}, "orientation": {}}
        for key, value in filtered_poses.items():
            std["position"][key] = np.std(
                np.linalg.norm(reference_poses["true"][:, :-1] - value[:, :-1], axis=1)
            )
            std["orientation"][key] = np.std(
                np.array(angle_set_diff(reference_poses["true"][:, 2], value[:, 2]))
            )

        return err_mean_sqare, err_mean_abs, std

    results = resultLoader.load(results_filename)
    filtered_poses, reference_poses = calcPoseFromStateArray(results[0], results[1])

    err_mean_sqare, err_mean_abs, std = calcErrorMetrics(
        filtered_poses, reference_poses
    )
    print(err_mean_sqare)
    print(err_mean_abs)
    print(std)

    map_fn = "tb3_house_lessnoisy"

    ogm = GridMap.load_grid_map_from_array(
        PgmProcesser.read_pgm(map_fn), 0.05, 10, 10.05
    )
    track_plotter = TrackPlotter()
    track_plotter.background_map = ogm
    track_plotter.plot_tracks(results[0], results[1])


if __name__ == "__main__":
    results_filename = "22-02-14T101103"
    main(results_filename)
