import numpy as np
from ..rawdata import YamlWriter


def _normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def _calc_angle_diff(a, b):
    a = _normalize_angle(a)
    b = _normalize_angle(b)
    d1 = a - b
    d2 = 2 * np.pi - abs(d1)
    if d1 > 0:
        d2 *= -1.0
    if abs(d1) < abs(d2):
        return d1
    else:
        return d2


def _angle_set_diff(set_1, set_2):
    return [_calc_angle_diff(a, b) for a, b in zip(set_1, set_2)]


def _calc_error_metrics(filtered_poses, reference_poses):

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
        err_angle = np.array(_angle_set_diff(true_angle, filtered_angle))

        # MSE
        err_mean_sqare["position"][algo] = float(np.sqrt(np.mean(err_xy_norm ** 2)))
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

    err_mean_sqare, err_mean_abs, err_std = _calc_error_metrics(
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
