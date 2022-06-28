from kinematics import MotionModel
from measurement import MeasurementModel
from typing import Optional
import numpy as np
import time
from customtypes import StateHypothesis


class CEDH:
    def __init__(
        self,
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        particle_num,
    ) -> None:
        self.motion_model: MotionModel = motion_model
        self.measurement_model: MeasurementModel = measurement_model
        self.particle_num = particle_num
        self.filtered_states: list[StateHypothesis] = []
        self.propagated_state: Optional[StateHypothesis] = None
        self.run_time = 0

    def init_particles_from_gaussian(
        self, init_mean: np.ndarray, init_covar: np.ndarray, return_state=False
    ) -> StateHypothesis:
        init_state = StateHypothesis.init_particles_from_gaussian(
            self.particle_num, init_mean, init_covar
        )
        self.filtered_states.append(init_state)
        if return_state:
            return init_state

    def propagate(self, control_input):
        particle_poses = self.motion_model.propagate_particles(
            self.filtered_states[-1].particles, control_input
        )
        self.propagated_state = StateHypothesis.create_from_particles(particle_poses)

    def update(self, ekf_covar: np.ndarray, measurement) -> None:

        start_time = time.time()
        num_of_rays = len(measurement)
        measurement_covar = self.measurement_model.range_noise_std**2 * np.eye(
            num_of_rays
        )

        particle_poses = self.propagated_state.particles
        particle_poses_mean = np.mean(particle_poses, axis=0)

        cd, grad_cd_x, grad_cd_z, _ = self.measurement_model.process_detection(
            StateHypothesis(particle_poses_mean), measurement
        )
        y = -cd + grad_cd_x.T @ particle_poses_mean

        M = ekf_covar @ (grad_cd_x @ grad_cd_x.T)
        p = grad_cd_x.T @ ekf_covar @ grad_cd_x
        w = (
            ekf_covar
            @ grad_cd_x
            * np.linalg.inv(grad_cd_z.T @ measurement_covar @ grad_cd_z)
            * y
        )
        r = grad_cd_z.T @ measurement_covar @ grad_cd_z
        mu = np.trace(M)

        fi = np.eye(3) + M / mu * (np.sqrt(r / (p + r)) - 1)
        fib0 = 2 / 3 * w / p * (p + r) - 2 / 3 * w / p * r * np.sqrt(r / (p + r))
        fib1 = M / p @ particle_poses_mean[:, np.newaxis] * (np.sqrt(r / (p + r)) - 1)
        fib2 = 2 * w / p * (r - p / 2) - 2 / p * w * r * np.sqrt(r / (r + p))
        fib3 = 2 / p * M @ particle_poses_mean[:, np.newaxis] * (p / 2 + r) / (
            p + r
        ) - 2 / p * M @ particle_poses_mean[:, np.newaxis] * r / np.sqrt(r * (p + r))
        fib4 = 1 / 3 * w / p * (p**2 - 4 * p * r - 8 * r**2) / (
            p + r
        ) + 1 / 3 * w / p * 8 * r**2 / np.sqrt(r) / np.sqrt(p + r)
        particle_poses = fi @ particle_poses.T + 1 * (fib0 + fib1 + fib2 + fib3 + fib4)
        particle_poses = particle_poses.T

        updated_state = StateHypothesis.create_from_particles(particle_poses)
        self.filtered_states.append(updated_state)

        time_elapsed = time.time() - start_time
        self.run_time = self.run_time + time_elapsed

    def update_multistep(self, ekf_covar: np.ndarray, measurement, stepnum):
        start_time = time.time()
        num_of_rays = len(measurement)
        measurement_covar = self.measurement_model.range_noise_std**2 * np.eye(
            num_of_rays
        )

        particle_poses = self.propagated_state.particles
        particle_poses_mean_0 = np.mean(particle_poses, axis=0)

        lambdas = np.linspace(0, 1, stepnum + 1)
        for i in range(stepnum):
            particle_poses_mean = np.mean(particle_poses, axis=0)
            cd, grad_cd_x, grad_cd_z, _ = self.measurement_model.process_detection(
                StateHypothesis(particle_poses_mean), measurement
            )
            y = -cd + grad_cd_x.T @ particle_poses_mean
            M = ekf_covar @ (grad_cd_x @ grad_cd_x.T)
            p = grad_cd_x.T @ ekf_covar @ grad_cd_x
            w = (
                ekf_covar
                @ grad_cd_x
                * np.linalg.inv(grad_cd_z.T @ measurement_covar @ grad_cd_z)
                * y
            )
            r = grad_cd_z.T @ measurement_covar @ grad_cd_z

            lam_0 = lambdas[i]
            lam_1 = lambdas[i + 1]

            kl0 = lam_0 * p + r
            kl1 = lam_1 * p + r

            fi = np.eye(3) + M / p * (np.sqrt(kl0 / kl1) - 1)
            fib0 = 2 / 3 * w / p * (kl1 - kl0 ** (3 / 2) * kl1 ** (-1 / 2))
            fib1 = (
                M
                / p
                @ particle_poses_mean_0[:, np.newaxis]
                * (kl1 ** (-1 / 2) * kl0 ** (1 / 2) - 1)
            )
            fib2 = (
                w
                / p
                * (
                    2 * r
                    - p * lam_1
                    - kl1 ** (-1 / 2) * kl0 ** (1 / 2) * (2 * r - p * lam_0)
                )
            )
            fib3 = (
                M
                @ particle_poses_mean_0[:, np.newaxis]
                / p
                * (
                    (p * lam_1 + 2 * r) * kl1 ** (-1)
                    - (p * lam_0 + 2 * r) * kl0 ** (-1 / 2) * kl1 ** (-1 / 2)
                )
            )
            fib4 = (
                1
                / 3
                / p
                * w
                * kl1 ** (-1 / 2)
                * (
                    (p**2 * lam_1**2 - 4 * p * r * lam_1 - 8 * r**2)
                    * ((p * lam_1 + r) ** (-1 / 2))
                    - (p**2 * lam_0**2 - 4 * p * r * lam_0 - 8 * r**2)
                    * ((p * lam_0 + r) ** (-1 / 2))
                )
            )

            particle_poses = fi @ particle_poses.T + 1 * (
                fib0 + fib1 + fib2 + fib3 + fib4
            )
            particle_poses = particle_poses.T

        updated_state = StateHypothesis.create_from_particles(particle_poses)
        self.filtered_states.append(updated_state)

        time_elapsed = time.time() - start_time
        self.run_time = self.run_time + time_elapsed

    def get_runtime(self):
        return self.run_time
 