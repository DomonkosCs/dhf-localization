from ..kinematics import MotionModel
from ..measurement import MeasurementModel
from ..customtypes import StateHypothesis
import numpy as np


class EKF:
    def __init__(self, motion_model: MotionModel, measurement_model: MeasurementModel):
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.filtered_states = []
        self.propagated_state = None

    def init_state(self, mean: np.ndarray, covar: np.ndarray) -> None:
        state = StateHypothesis(pose=mean, covar=covar)
        self.filtered_states.append(state)

    def propagate(self, prior, control_input) -> StateHypothesis:
        return self.motion_model.propagate(prior, control_input)

    def update(self, prior, measurement) -> StateHypothesis:

        (
            cd,
            grad_cd_x,
            grad_cd_z,
            _,
        ) = self.measurement_model.process_detection(prior.state_vector, measurement)

        measurement_covar = self.measurement_model.range_noise_std**2 * np.eye(
            grad_cd_z.shape[0]
        )

        K = (
            prior.covar
            @ grad_cd_x
            / (
                grad_cd_x.T @ prior.covar @ grad_cd_x
                + grad_cd_z.T @ measurement_covar @ grad_cd_z
            )
        )

        posterior_mean = prior.state_vector - K.flatten() * cd
        posterior_covar = (np.eye(3) - K @ grad_cd_x.T) @ prior.covar
        posterior = StateHypothesis(state_vector=posterior_mean, covar=posterior_covar)

        return posterior
