from ..customtypes import ParticleState
from ..kinematics import MotionModel
from ..measurement import MeasurementModel
import numpy as np
import time


class EDH:
    def __init__(
        self,
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        particle_num,
        lambda_num,
    ) -> None:
        self.motion_model: MotionModel = motion_model
        self.measurement_model: MeasurementModel = measurement_model
        self.lambda_num = lambda_num
        self.particle_num = particle_num
        lambdas, self.d_lambda = np.linspace(
            0, 1, self.lambda_num, endpoint=False, retstep=True
        )
        self.lambdas = lambdas + self.d_lambda
        self.run_time = 0

    def propagate(self, prior, control_input):
        return self.motion_model.propagate_particles(prior, control_input)

    def update(self, prediction, ekf_prediction, measurement) -> None:

        start_time = time.time()
        num_of_rays = len(measurement)
        measurement_covar = self.measurement_model.range_noise_std ** 2 * np.eye(
            num_of_rays
        )

        ekf_covar = ekf_prediction.covar
        particle_poses = prediction.state_vectors
        particle_poses_mean = prediction.mean()
        particle_poses_mean_0 = particle_poses_mean
        for lamb in self.lambdas:
            cd, grad_cd_x, grad_cd_z, _ = self.measurement_model.process_detection(
                particle_poses_mean, measurement
            )

            y = -cd + grad_cd_x.T @ particle_poses_mean
            B = (
                -0.5
                * ekf_covar
                @ grad_cd_x
                / (
                    lamb * grad_cd_x.T @ ekf_covar @ grad_cd_x
                    + grad_cd_z.T @ measurement_covar @ grad_cd_z
                )
                @ grad_cd_x.T
            )

            b = (np.eye(3) + 2 * lamb * B) @ (
                (np.eye(3) + lamb * B)
                @ ekf_covar
                @ grad_cd_x
                / (grad_cd_z.T @ measurement_covar @ grad_cd_z)
                * y
                + B @ np.array([particle_poses_mean_0]).T
            )

            particle_poses = particle_poses + self.d_lambda * (
                (B @ particle_poses.T).T + b.T
            )
            particle_poses_mean = np.mean(particle_poses, axis=0)

        posterior = ParticleState(state_vectors=particle_poses)

        time_elapsed = time.time() - start_time
        self.run_time = self.run_time + time_elapsed

        return posterior

    def get_runtime(self):
        return self.run_time
