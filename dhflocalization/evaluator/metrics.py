import numpy as np
from ..rawdata import YamlWriter
from dhflocalization.utils import angle_set_diff, calc_angle_diff

def calc_nees(true_track,filtered_track):
    covars,states = map(list,zip(*[(timestep.covar,timestep.state_vector) for timestep in filtered_track]))
    timesteps = len(covars)
    dims = len(states[0])
    nees_track = np.zeros(timesteps)
    for t in range(len(covars)):
        covar = covars[t]
        state = states[t]
        true = true_track[t]

        diff = np.array([state[0]-true[0],state[1]-true[1],calc_angle_diff(state[2],true[2])])
        nees = diff[np.newaxis] @ np.linalg.inv(covar) @ diff[:,np.newaxis]
        nees_track[t] = nees


    nees_avg = 1/dims * nees_track.mean()
    return nees_avg

def calc_error_metrics(filtered_poses, reference_poses):

    err_mean_sqare = {"position": {}, "orientation": {}}
    err_mean_abs = {"position": {}, "orientation": {}}
    err_std = {"position": {}, "orientation": {}}

    for algo, value in filtered_poses.items():
        states = np.array(value["state"])

        true_xy = reference_poses["true"][:, :-1]
        true_angle = reference_poses["true"][:, 2]

        filtered_xy = states[:, :-1]
        filtered_angle = states[:, 2]

        err_xy_norm = np.linalg.norm(true_xy - filtered_xy, axis=1)
        err_angle = np.array(angle_set_diff(true_angle, filtered_angle))

        # MSE
        err_mean_sqare["position"][algo] = float(np.sqrt(np.mean(err_xy_norm**2)))
        err_mean_sqare["orientation"][algo] = float(np.sqrt(np.mean(err_angle**2)))

        # MAE
        err_mean_abs["position"][algo] = float(np.mean(err_xy_norm))
        err_mean_abs["orientation"][algo] = float(np.mean(np.abs(err_angle)))

        # STD of error
        err_std["position"][algo] = float(np.std(err_xy_norm))
        err_std["orientation"][algo] = float(np.std(err_angle))

    return err_mean_sqare, err_mean_abs, err_std


def eval(
    filtered_states,
    reference_states,
    export_filename=None,
    return_results=False,
):

    err_mean_sqare, err_mean_abs, err_std = calc_error_metrics(
        filtered_states, reference_states
    )
    if export_filename:
        metrics_dict = {
            "RMSE": err_mean_sqare,
            "MAE": err_mean_abs,
            "STD": err_std,
        }
        YamlWriter().updateFile(
            payload=metrics_dict,
            filename=export_filename,
        )
    if return_results:
        return err_mean_sqare, err_mean_abs, err_std
