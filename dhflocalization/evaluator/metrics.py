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
    for key, value in filtered_poses.items():
        value = np.array(value)
        err_mean_sqare["position"][key] = float(
            np.sqrt(
                np.mean(
                    np.linalg.norm(
                        reference_poses["true"][:, :-1] - value[:, :-1], axis=1
                    )
                    ** 2
                )
            )
        )
        err_mean_sqare["orientation"][key] = float(
            np.sqrt(
                np.mean(
                    np.array(
                        _angle_set_diff(reference_poses["true"][:, 2], value[:, 2])
                    )
                    ** 2
                )
            )
        )

    err_mean_abs = {"position": {}, "orientation": {}}
    for key, value in filtered_poses.items():
        err_mean_abs["position"][key] = float(
            np.mean(
                np.linalg.norm(reference_poses["true"][:, :-1] - value[:, :-1], axis=1)
            )
        )
        err_mean_abs["orientation"][key] = float(
            np.mean(
                np.abs(
                    np.array(
                        _angle_set_diff(reference_poses["true"][:, 2], value[:, 2])
                    )
                )
            )
        )

    std = {"position": {}, "orientation": {}}
    for key, value in filtered_poses.items():
        std["position"][key] = float(
            np.std(
                np.linalg.norm(reference_poses["true"][:, :-1] - value[:, :-1], axis=1)
            )
        )
        std["orientation"][key] = float(
            np.std(
                np.array(_angle_set_diff(reference_poses["true"][:, 2], value[:, 2]))
            )
        )

    return err_mean_sqare, err_mean_abs, std


def eval(
    filtered_states, reference_states, export_filename=None, return_results=False,
):

    err_mean_sqare, err_mean_abs, std = _calc_error_metrics(
        filtered_states, reference_states
    )
    if export_filename:
        metrics_dict = {
            "RMSE": err_mean_sqare,
            "MAE": err_mean_abs,
            "STD": std,
        }
        YamlWriter().updateFile(
            payload=metrics_dict, filename=export_filename,
        )
    if return_results:
        return err_mean_sqare, err_mean_abs, std
