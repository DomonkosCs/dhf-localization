import numpy as np
from rawdata import YamlWriter


class MetricEvaluator:
    def __init__(self) -> None:
        pass

    def _normalize_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _calc_angle_diff(self, a, b):
        a = self._normalize_angle(a)
        b = self._normalize_angle(b)
        d1 = a - b
        d2 = 2 * np.pi - abs(d1)
        if d1 > 0:
            d2 *= -1.0
        if abs(d1) < abs(d2):
            return d1
        else:
            return d2

    def _angle_set_diff(self, set_1, set_2):
        return [self._calc_angle_diff(a, b) for a, b in zip(set_1, set_2)]

    def _calc_pose_from_state_array(self, filtered_states, reference_states):
        filtered_poses = {}
        for key, value in filtered_states.items():
            filtered_poses[key] = np.array([state.pose for state in value])
        reference_poses = {}
        for key, value in reference_states.items():
            reference_poses[key] = np.array([state.pose for state in value])

        return filtered_poses, reference_poses

    def _calc_error_metrics(self, filtered_poses, reference_poses):

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
                            self._angle_set_diff(
                                reference_poses["true"][:, 2], value[:, 2]
                            )
                        )
                        ** 2
                    )
                )
            )

        err_mean_abs = {"position": {}, "orientation": {}}
        for key, value in filtered_poses.items():
            err_mean_abs["position"][key] = float(
                np.mean(
                    np.linalg.norm(
                        reference_poses["true"][:, :-1] - value[:, :-1], axis=1
                    )
                )
            )
            err_mean_abs["orientation"][key] = float(
                np.mean(
                    np.abs(
                        np.array(
                            self._angle_set_diff(
                                reference_poses["true"][:, 2], value[:, 2]
                            )
                        )
                    )
                )
            )

        std = {"position": {}, "orientation": {}}
        for key, value in filtered_poses.items():
            std["position"][key] = float(
                np.std(
                    np.linalg.norm(
                        reference_poses["true"][:, :-1] - value[:, :-1], axis=1
                    )
                )
            )
            std["orientation"][key] = float(
                np.std(
                    np.array(
                        self._angle_set_diff(reference_poses["true"][:, 2], value[:, 2])
                    )
                )
            )

        return err_mean_sqare, err_mean_abs, std

    def eval(
        self,
        filtered_states,
        reference_states,
        export_filename=None,
        return_results=False,
    ):
        filtered_poses, reference_poses = self._calc_pose_from_state_array(
            filtered_states, reference_states
        )

        err_mean_sqare, err_mean_abs, std = self._calc_error_metrics(
            filtered_poses, reference_poses
        )
        if export_filename:
            metrics_dict = {
                "RMSE": err_mean_sqare,
                "MAE": err_mean_abs,
                "STD": std,
            }
            YamlWriter().updateFile(
                payload=metrics_dict,
                filename=export_filename,
            )
        if return_results:
            return err_mean_sqare, err_mean_abs, std
