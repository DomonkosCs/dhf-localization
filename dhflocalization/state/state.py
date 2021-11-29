

import numpy as np


class StateHypothesis:
    def __init__(self, pose=[], covar=[], particles=None):
        self.pose = np.array([pose]).T
        self.covar = np.array(covar)
        self.particles = particles

    @classmethod
    def init_from_particle_prior(cls, particle_num, particle_mean, particle_covar, use_init_covar=False):
        particles = np.random.multivariate_normal(
            particle_mean, particle_covar, particle_num)
        return cls.init_from_particle_poses(particles, particle_covar, use_init_covar)

    @classmethod
    def init_from_particle_poses(cls, particles, init_particle_covar=None, use_init_covar=False):
        if use_init_covar:
            particle_covar = init_particle_covar
        else:
            particle_covar = np.cov(particles, rowvar=False)
        particle_mean = np.mean(particles, axis=0)
        return cls(particle_mean, particle_covar, particles)

    def remove_particles(self):
        self.particles = None
