from matplotlib.pyplot import axis
from kinematics.motionmodel import MotionModel
from measurement.measurement import Measurement
from state.particle import ParticleSet
import numpy as np

from state.state import StateHypothesis


class EDH:
    def __init__(self, motion_model: MotionModel, measurement_model: Measurement) -> None:
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.d_lambda = 0.1
        self.lambdas = np.linspace(self.d_lambda, 1, 10)

    def propagate(self, particle_set: ParticleSet, control_input):
        particle_poses = self.motion_model.propagate_particles(
            particle_set.particle_poses, control_input)

        return ParticleSet(particle_poses)

    def update(self, particle_set: ParticleSet, ekf_covar, measurement):

        num_of_rays = len(measurement)
        measurement_covar = self.measurement_model.range_noise_std**2 * \
            np.eye(num_of_rays)

        particle_poses = particle_set.particle_poses
        for l in self.lambdas:
            particle_poses_mean = np.mean(particle_poses, axis=0)
            cd, grad_cd_x, grad_cd_z, _ = self.measurement_model.processDetection(
                StateHypothesis(np.array([particle_poses_mean]).T), measurement)

            y = - cd + grad_cd_x.T @ particle_poses_mean

            B = -0.5*ekf_covar @ grad_cd_x / \
                (l*grad_cd_x.T @ ekf_covar @ grad_cd_x +
                 grad_cd_z.T @ measurement_covar @ grad_cd_z) @ grad_cd_x.T

            b = (np.eye(3) + 2*l*B) @ \
                ((np.eye(3) + l*B) @ ekf_covar @ grad_cd_x /
                 (grad_cd_z.T @ measurement_covar @ grad_cd_z) * y + B @ np.array([particle_poses_mean]).T)

            particle_poses = particle_poses + self.d_lambda * (np.array(
                [B @ particle_state for particle_state in particle_poses]) + b.T)

        return ParticleSet(particle_poses)
