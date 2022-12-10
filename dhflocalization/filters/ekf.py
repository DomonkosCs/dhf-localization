from ..measurement import MeasurementModel
from ..customtypes import StateHypothesis
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

        posterior_mean = prior.state_vector - K.flatten() * cd
        posterior_covar = (np.eye(3) - K @ grad_cd_x.T) @ prior.covar
        posterior = StateHypothesis(state_vector=posterior_mean, covar=posterior_covar)

        end = time.time()
        comptime = end - start
        return posterior, comptime
