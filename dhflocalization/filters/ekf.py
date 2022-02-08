from kinematics import MotionModel
from measurement import Measurement
from state.state import StateHypothesis
import numpy as np


class EKF:
    def __init__(self, motion_model: MotionModel, measurement_model: Measurement):
        self.motion_model = motion_model
        self.measurement_model = measurement_model

    def propagate(self, state: StateHypothesis, control_input) -> StateHypothesis:
        predicted_state = self.motion_model.propagate(
            state, control_input)

        return predicted_state

    def update(self, state: StateHypothesis, measurement) -> StateHypothesis:

        cd, grad_cd_x, grad_cd_z, ray_endpoints = self.measurement_model.processDetection(
            state, measurement)
        updated_state = StateHypothesis()

        measurement_covar = self.measurement_model.range_noise_std**2 * \
            np.eye(grad_cd_z.shape[0])

        K = state.covar @ grad_cd_x / \
            (grad_cd_x.T @ state.covar @ grad_cd_x +
             grad_cd_z.T @ measurement_covar @ grad_cd_z)
        updated_state.pose = state.pose - K * cd

        updated_state.covar = (np.eye(3) - K @ grad_cd_x.T) @ state.covar
        return updated_state
