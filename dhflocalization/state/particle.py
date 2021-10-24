import numpy as np

from state.state import StateHypothesis


class ParticleSet:
    def __init__(self, particle_poses):
        self.particle_poses = particle_poses
        particle_covar = np.cov(self.particle_poses, rowvar=False)
        particle_mean = np.mean(self.particle_poses, axis=0)
        self.mean_state = StateHypothesis(
            np.array([particle_mean]).T, particle_covar)

    @classmethod
    def init_from_prior(cls, particle_num, particle_mean, particle_covar):
        particle_poses = np.random.multivariate_normal(
            particle_mean, particle_covar, particle_num)
        return cls(particle_poses)
