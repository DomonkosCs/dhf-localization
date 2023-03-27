from ..customtypes import ParticleState
import numpy as np


class EDH:
    def __init__(self, updater, init_mean, init_covar, rng=None):
        self.updater = updater
        self.prior = self._init_particles(init_mean, init_covar, rng)

        self.last_particle_posterior = self.prior
        self.filtered_track = []
        self.filtered_track.append(self.last_particle_posterior.mean())

        self.comptimes = []

    def _init_particles(self, init_mean, init_covar, rng):
        return ParticleState.init_from_gaussian(
            init_mean,
            init_covar,
            self.updater.particle_num,
            rng=rng,
        )

    def update(self, prediction, prediction_covar, measurement):
        posterior, comptime = self.updater.update(
            prediction, prediction_covar, measurement
        )

        self.filtered_track.append(posterior.mean())
        self.comptimes.append(comptime)

        self.last_particle_posterior = posterior

    def get_results(self):
        return {
            self.updater.key: {
                "state": np.asarray(self.filtered_track),
                "comptime": np.array(self.comptimes).mean(),
            }
        }
