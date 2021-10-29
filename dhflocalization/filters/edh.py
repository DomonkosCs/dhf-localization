from matplotlib.pyplot import axis
from kinematics.motionmodel import MotionModel
from measurement.measurement import Measurement
import numpy as np

from state.state import StateHypothesis


class EDH:
    def __init__(self, motion_model: MotionModel, measurement_model: Measurement) -> None:
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.d_lambda = 0.1
        self.lambdas = np.linspace(self.d_lambda, 1, 10)

    def propagate(self, state: StateHypothesis, control_input):
        particle_poses = self.motion_model.propagate_particles(
            state.particles, control_input)

        return StateHypothesis.init_from_particle_poses(particle_poses)

    def update(self, state: StateHypothesis, ekf_covar, measurement):

        num_of_rays = len(measurement)
        measurement_covar = self.measurement_model.range_noise_std**2 * \
            np.eye(num_of_rays)

        particle_poses = state.particles
        for l in self.lambdas:
            particle_poses_mean = np.mean(particle_poses, axis=0)
            cd, grad_cd_x, grad_cd_z, _ = self.measurement_model.processDetection(
                StateHypothesis(particle_poses_mean), measurement)

            y = - cd + grad_cd_x.T @ particle_poses_mean

            B = -0.5*ekf_covar @ grad_cd_x / \
                (l*grad_cd_x.T @ ekf_covar @ grad_cd_x +
                 grad_cd_z.T @ measurement_covar @ grad_cd_z) @ grad_cd_x.T

            b = (np.eye(3) + 2*l*B) @ \
                ((np.eye(3) + l*B) @ ekf_covar @ grad_cd_x /
                 (grad_cd_z.T @ measurement_covar @ grad_cd_z) * y + B @ np.array([particle_poses_mean]).T)

            particle_poses = particle_poses + self.d_lambda * (np.array(
                [B @ particle_state for particle_state in particle_poses]) + b.T)
            particle_poses[particle_poses[:, 2] > np.pi, 2] -= 2*np.pi
            particle_poses[particle_poses[:, 2] < -np.pi, 2] += 2*np.pi
        return StateHypothesis.init_from_particle_poses(particle_poses)
