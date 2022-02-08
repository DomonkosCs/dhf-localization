from matplotlib.pyplot import axis
from kinematics import MotionModel
from measurement import Measurement
import numpy as np

from state import StateHypothesis


class EDH:
    def __init__(self, motion_model: MotionModel, measurement_model: Measurement) -> None:
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.lambda_num = 10
        lambdas, self.d_lambda = np.linspace(
            0, 1, self.lambda_num, endpoint=False, retstep=True)
        self.lambdas = lambdas+self.d_lambda

    def propagate(self, state: StateHypothesis, control_input):
        particle_poses = self.motion_model.propagate_particles(
            state.particles, control_input)

        return StateHypothesis.init_from_particle_poses(particle_poses)

    def update(self, state: StateHypothesis, ekf_covar, measurement):

        num_of_rays = len(measurement)
        measurement_covar = self.measurement_model.range_noise_std**2 * \
            np.eye(num_of_rays)

        particle_poses = state.particles
        particle_poses_mean = np.mean(particle_poses, axis=0)
        particle_poses_mean_0 = particle_poses_mean
        for l in self.lambdas:
            cd, grad_cd_x, grad_cd_z, _ = self.measurement_model.processDetection(
                StateHypothesis(particle_poses_mean), measurement)

            y = - cd + grad_cd_x.T @ particle_poses_mean
            B = -0.5*ekf_covar @ grad_cd_x / \
                (l*grad_cd_x.T @ ekf_covar @ grad_cd_x +
                 grad_cd_z.T @ measurement_covar @ grad_cd_z) @ grad_cd_x.T

            b = (np.eye(3) + 2*l*B) @ \
                ((np.eye(3) + l*B) @ ekf_covar @ grad_cd_x /
                 (grad_cd_z.T @ measurement_covar @ grad_cd_z) * y + B @ np.array([particle_poses_mean_0]).T)

            particle_poses = particle_poses + self.d_lambda * (np.array(
                [B @ particle_state for particle_state in particle_poses]) + b.T)
            particle_poses_mean = np.mean(particle_poses, axis=0)

        return StateHypothesis.init_from_particle_poses(particle_poses)
