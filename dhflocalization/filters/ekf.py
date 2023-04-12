from ..measurement import MeasurementModel
from ..customtypes import StateHypothesis
from ..utils import calc_angle_diff
import numpy as np
import time


class EKF:
    def __init__(self, measurement_model: MeasurementModel):
        self.measurement_model = measurement_model

    def update(self, prior, measurement) -> StateHypothesis:
        start = time.time()

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

        # cf. Markovic, I., Cesic, J., & Petrovic, I. (2017). On wrapping the Kalman filter and estimating with the SO(2) group
        # however, it literally makes no difference
        correction = K.flatten() * cd
        posterior_mean = np.zeros(3)
        posterior_mean[:2] = prior.state_vector[:2] - correction[:2]
        posterior_mean[2] = calc_angle_diff(prior.state_vector[2], correction[2])
        posterior_covar = (np.eye(3) - K @ grad_cd_x.T) @ prior.covar
        posterior = StateHypothesis(state_vector=posterior_mean, covar=posterior_covar)

        end = time.time()
        comptime = end - start
        return posterior, comptime
