from kinematics import MotionModel
from measurement import Measurement
from customtypes import StateHypothesis
import numpy as np


class EKF:
    def __init__(self, motion_model: MotionModel, measurement_model: Measurement):
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.filtered_states = []
        self.propagated_state = None

    def init_state(self, mean: np.ndarray, covar: np.ndarray) -> None:
        state = StateHypothesis(pose=mean, covar=covar)
        self.filtered_states.append(state)

    def propagate(self, control_input, return_state=False) -> StateHypothesis:
        propagated_state = self.motion_model.propagate(
            self.filtered_states[-1], control_input
        )
        self.propagated_state = propagated_state

        if return_state:
            return propagated_state

    def update(self, measurement) -> StateHypothesis:
        state = self.propagated_state

        (
            cd,
            grad_cd_x,
            grad_cd_z,
            _,
        ) = self.measurement_model.processDetection(state, measurement)

        measurement_covar = self.measurement_model.range_noise_std**2 * np.eye(
            grad_cd_z.shape[0]
        )

        K = (
            state.covar
            @ grad_cd_x
            / (
                grad_cd_x.T @ state.covar @ grad_cd_x
                + grad_cd_z.T @ measurement_covar @ grad_cd_z
            )
        )

        updated_pose = state.pose - K * cd
        updated_covar = (np.eye(3) - K @ grad_cd_x.T) @ state.covar
        updated_state = StateHypothesis(pose=updated_pose, covar=updated_covar)

        self.filtered_states.append(updated_state)
