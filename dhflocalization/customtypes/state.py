import numpy as np


class StateHypothesis:
    def __init__(self, state_vector: np.ndarray, covar: np.ndarray):
        self.state_vector = state_vector
        self.covar = covar


class ParticleState:
    def __init__(self, state_vectors=None):
        self.state_vectors = state_vectors

    def mean(self):
        return np.average(self.state_vectors, axis=0)

    @classmethod
    def init_from_gaussian(cls, mean, covar, particle_num, rng=np.random.default_rng()):
        state_vectors = rng.multivariate_normal(mean, covar, particle_num)
        return cls(state_vectors=state_vectors)
