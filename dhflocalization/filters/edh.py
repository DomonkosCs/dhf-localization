from kinematics import MotionModel
from measurement import Measurement
import numpy as np

from customtypes import StateHypothesis


class EDH:
    def __init__(
        self, motion_model: MotionModel, measurement_model: Measurement, particle_num
    ) -> None:
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.lambda_num = 10
        self.particle_num = particle_num
        lambdas, self.d_lambda = np.linspace(
            0, 1, self.lambda_num, endpoint=False, retstep=True
        )
        self.lambdas = lambdas + self.d_lambda
        self.filtered_states = []
        self.propagated_state = None

    def init_particles_from_gaussian(self, init_mean, init_covar, return_state=False):
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

    def update(self, ekf_covar, measurement):

        num_of_rays = len(measurement)
        measurement_covar = self.measurement_model.range_noise_std**2 * np.eye(
            num_of_rays
        )

        particle_poses = self.propagated_state.particles
        particle_poses_mean = np.mean(particle_poses, axis=0)
        particle_poses_mean_0 = particle_poses_mean
        for lamb in self.lambdas:
            cd, grad_cd_x, grad_cd_z, _ = self.measurement_model.processDetection(
                StateHypothesis(particle_poses_mean), measurement
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
                np.array([B @ particle_state for particle_state in particle_poses])
                + b.T
            )
            particle_poses_mean = np.mean(particle_poses, axis=0)

        updated_state = StateHypothesis.create_from_particles(particle_poses)
        self.filtered_states.append(updated_state)
