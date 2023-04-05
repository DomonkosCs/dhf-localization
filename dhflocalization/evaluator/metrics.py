import numpy as np
from ..rawdata import YamlWriter
from dhflocalization.utils import angle_set_diff, calc_angle_diff

def calc_nees(true_states, filtered_track):
    covars, states = map(
        list,
        zip(*[(timestep.covar, timestep.state_vector) for timestep in filtered_track]),
    )
    timesteps = len(covars)
    dims = len(states[0])
    nees_track = np.zeros(timesteps)
    for t in range(len(covars)):
        covar = covars[t]
        state = states[t]
        true = true_states[t]

        diff = np.array(
            [state[0] - true[0], state[1] - true[1], calc_angle_diff(state[2], true[2])]
        )
        nees = diff[np.newaxis] @ np.linalg.inv(covar) @ diff[:, np.newaxis]
        nees_track[t] = nees

    nees_avg = 1 / dims * nees_track.mean()
    return nees_avg


def calc_error_metrics(true_states, filtered_track):
    err_mean_sqare = {}
    err_mean_abs = {}
    err_std = {}

    filtered_states = filtered_track.to_np_array()

    true_xy = true_states[:, :-1]
    true_angle = true_states[:, 2]

    filtered_xy = filtered_states[:, :-1]
    filtered_angle = filtered_states[:, 2]

    err_xy_norm = np.linalg.norm(true_xy - filtered_xy, axis=1)
    err_angle = np.array(angle_set_diff(true_angle, filtered_angle))

    # MSE
    err_mean_sqare["pos"] = float(np.sqrt(np.mean(err_xy_norm**2)))
    err_mean_sqare["ori"] = float(np.sqrt(np.mean(err_angle**2)))

    # MAE
    err_mean_abs["pos"] = float(np.mean(err_xy_norm))
    err_mean_abs["ori"] = float(np.mean(np.abs(err_angle)))

    # STD of error
    err_std["pos"] = float(np.std(err_xy_norm))
    err_std["ori"] = float(np.std(err_angle))

    return err_mean_sqare, err_mean_abs, err_std

def eval(
    true_states,
    filtered_results,
    export_filename=None,
    return_results=False,
):

    err_mean_squares = {}
    err_mean_abss = {}
    err_stds = {}
    for algo, result in filtered_results.items():
        err_mean_square, err_mean_abs, err_std = calc_error_metrics(
            true_states, result["track"]
        )
        err_mean_squares[algo] = err_mean_square
        err_mean_abss[algo] = err_mean_abs
        err_stds[algo] = err_std
        

    if export_filename:
        metrics_dict = {
            "RMSE": err_mean_squares,
            "MAE": err_mean_abss,
            "STD": err_stds,
        }
        YamlWriter().updateFile(
            payload=metrics_dict,
            filename=export_filename,
        )
    if return_results:
        return err_mean_squares, err_mean_abss, err_stds
