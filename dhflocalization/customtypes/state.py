from selectors import PollSelector
from importlib_metadata import NullFinder
import numpy as np


class StateHypothesis:
    def __init__(self, pose: np.ndarray, covar: np.ndarray = None, particles=None):

        if pose.shape == (3, 1):
            self.pose = pose
        elif pose.ndim == 1:
            self.pose = np.array([pose]).T
        else:
            self.pose = pose.T

        self.covar = np.array(covar) if covar is not None else None
        self.particles = particles

    @classmethod
    def init_particles_from_gaussian(cls, particle_num, gaussian_mean, gaussian_covar):
        particles = np.random.multivariate_normal(
            gaussian_mean, gaussian_covar, particle_num
        )

        return cls.create_from_particles(particles=particles)

    @classmethod
    def create_from_particles(cls, particles):

        particle_mean = np.mean(particles, axis=0)
        particle_covar = np.cov(particles, rowvar=False)

        return cls(pose=particle_mean, covar=particle_covar, particles=particles)
