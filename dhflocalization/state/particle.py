import numpy as np

from state.state import StateHypothesis


class ParticleSet:
    def __init__(self, particle_num, particle_mean, particle_covar) -> None:
        particle_poses = np.random.multivariate_normal(
            particle_mean, particle_covar, particle_num)
        self.particles = [StateHypothesis(np.asmatrix(particle_pose).T, None)
                          for particle_pose in particle_poses]

    def __iter__(self):
        for particle in self.particles:
            yield particle

    @classmethod
    def fromlist(cls, particle_list):
        obj = cls.__new__(cls)
        # super(MyClass, obj).__init__()
        obj.particles = particle_list
        return obj

    def mean(self):
        cumsum = np.matrix([0.0, 0.0, 0.0]).T
        for particle in self.particles:
            cumsum += particle.pose
        return cumsum/len(self.particles)
