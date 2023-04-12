from ..customtypes import ParticleState
import numpy as np
import time

from scipy.spatial.distance import pdist, squareform
from scipy.linalg import sqrtm
import kmedoids


class LEDHUpdater:
    # localized EDH
    def __init__(self, measurement_model, lambda_num, particle_num):
        self.key = "ledh"

        self.measurement_model = measurement_model
        self.particle_num = particle_num

        lambdas, self.d_lambda = np.linspace(
            0, 1, lambda_num, endpoint=False, retstep=True
        )
        self.lambdas = lambdas + self.d_lambda

    def update(self, prediction, prediction_covar, measurement):
        start = time.time()

        num_of_rays = len(measurement)
        measurement_covar = self.measurement_model.range_noise_std**2 * np.eye(
            num_of_rays
        )

        particle_poses = prediction.state_vectors
        particle_poses_mean_0 = prediction.mean()

        for lamb in self.lambdas:
            for i in range(self.particle_num):
                # linearize measurement model about the current particle
                cd, grad_cd_x, grad_cd_z, _ = self.measurement_model.process_detection(
                    particle_poses[i, :], measurement
                )

                # transform measurement
                y = -cd + grad_cd_x.T @ particle_poses[i, :]

                B = (
                    -0.5
                    * prediction_covar
                    @ grad_cd_x
                    / (
                        lamb * grad_cd_x.T @ prediction_covar @ grad_cd_x
                        + grad_cd_z.T @ measurement_covar @ grad_cd_z
                    )
                    @ grad_cd_x.T
                )

                b = (np.eye(3) + 2 * lamb * B) @ (
                    (np.eye(3) + lamb * B)
                    @ prediction_covar
                    @ grad_cd_x
                    / (grad_cd_z.T @ measurement_covar @ grad_cd_z)
                    * y
                    + B @ np.array([particle_poses_mean_0]).T
                )

                flow_vector = B @ particle_poses[i, :, np.newaxis] + b
                particle_poses[i, :] = (
                    particle_poses[i, :] + self.d_lambda * flow_vector.T
                )

        posterior = ParticleState(state_vectors=particle_poses)
        end = time.time()
        comptime = end - start
        return posterior, comptime


class MEDHUpdater:
    # mean EDH (original)
    def __init__(self, measurement_model, lambda_num, particle_num):
        self.key = "medh"

        self.measurement_model = measurement_model
        self.particle_num = particle_num

        lambdas, self.d_lambda = np.linspace(
            0, 1, lambda_num, endpoint=False, retstep=True
        )
        self.lambdas = lambdas + self.d_lambda

    def update(self, prediction, prediction_covar, measurement):
        start = time.time()

        num_of_rays = len(measurement)
        measurement_covar = self.measurement_model.range_noise_std**2 * np.eye(
            num_of_rays
        )

        particle_poses = prediction.state_vectors
        particle_poses_mean = prediction.mean()
        particle_poses_mean_0 = particle_poses_mean

        for lamb in self.lambdas:
            # linearize measurement model about the mean
            cd, grad_cd_x, grad_cd_z, _ = self.measurement_model.process_detection(
                particle_poses_mean, measurement
            )

            # transform measurement
            y = -cd + grad_cd_x.T @ particle_poses_mean

            B = (
                -0.5
                * prediction_covar
                @ grad_cd_x
                / (
                    lamb * grad_cd_x.T @ prediction_covar @ grad_cd_x
                    + grad_cd_z.T @ measurement_covar @ grad_cd_z
                )
                @ grad_cd_x.T
            )

            b = (np.eye(3) + 2 * lamb * B) @ (
                (np.eye(3) + lamb * B)
                @ prediction_covar
                @ grad_cd_x
                / (grad_cd_z.T @ measurement_covar @ grad_cd_z)
                * y
                + B @ np.array([particle_poses_mean_0]).T
            )

            # update particles
            particle_poses = particle_poses + self.d_lambda * (
                (B @ particle_poses.T).T + b.T
            )

            # recalculate linearization point
            particle_poses_mean = np.mean(particle_poses, axis=0)

        posterior = ParticleState(state_vectors=particle_poses)
        end = time.time()
        comptime = end - start
        return posterior, comptime


class CLEDHUpdater:
    # clustered EDH
    def __init__(self, measurement_model, lambda_num, particle_num, cluster_num):
        self.key = "cledh"

        self.measurement_model = measurement_model
        self.particle_num = particle_num
        self.cluster_num = cluster_num

        lambdas, self.d_lambda = np.linspace(
            0, 1, lambda_num, endpoint=False, retstep=True
        )
        self.lambdas = lambdas + self.d_lambda

    def update(self, prediction, prediction_covar, measurement):
        start = time.time()
        num_of_rays = len(measurement)
        measurement_covar = self.measurement_model.range_noise_std**2 * np.eye(
            num_of_rays
        )

        particle_poses = prediction.state_vectors
        particle_poses_mean = prediction.mean()
        particle_poses_mean_0 = particle_poses_mean

        # cluster particles
        flow_vectors = np.zeros_like(particle_poses)
        for i in range(self.particle_num):
            # linearize measurement model about the current particle
            cd, grad_cd_x, grad_cd_z, _ = self.measurement_model.process_detection(
                particle_poses[i, :], measurement
            )

            # transform measurement
            y = -cd + grad_cd_x.T @ particle_poses[i, :]

            B = (
                -0.5
                * prediction_covar
                @ grad_cd_x
                / (grad_cd_z.T @ measurement_covar @ grad_cd_z)
                @ grad_cd_x.T
            )

            b = (np.eye(3)) @ (
                (np.eye(3))
                @ prediction_covar
                @ grad_cd_x
                / (grad_cd_z.T @ measurement_covar @ grad_cd_z)
                * y
                + B @ np.array([particle_poses_mean_0]).T
            )

            flow_vectors[i, :] = (B @ particle_poses[i, :, np.newaxis] + b).T

        labels, medoid_idxs = self._pam_clustering(
            flow_vectors, particle_poses, prediction_covar, self.cluster_num
        )
        medoid_states = particle_poses[medoid_idxs, :]
        for lamb in self.lambdas:
            for i in range(self.cluster_num):
                # linearize measurement model about the current medoid
                cd, grad_cd_x, grad_cd_z, _ = self.measurement_model.process_detection(
                    medoid_states[i, :], measurement
                )

                # transform measurement
                y = -cd + grad_cd_x.T @ medoid_states[i, :]

                B = (
                    -0.5
                    * prediction_covar
                    @ grad_cd_x
                    / (
                        lamb * grad_cd_x.T @ prediction_covar @ grad_cd_x
                        + grad_cd_z.T @ measurement_covar @ grad_cd_z
                    )
                    @ grad_cd_x.T
                )

                b = (np.eye(3) + 2 * lamb * B) @ (
                    (np.eye(3) + lamb * B)
                    @ prediction_covar
                    @ grad_cd_x
                    / (grad_cd_z.T @ measurement_covar @ grad_cd_z)
                    * y
                    + B @ np.array([particle_poses_mean_0]).T
                )

                # update the medoid
                medoid_flow_vector = B @ medoid_states[i, :, np.newaxis] + b
                medoid_states[i, :] = (
                    medoid_states[i, :] + self.d_lambda * medoid_flow_vector.T
                )

                flow_vector = B @ particle_poses[labels == i, :].T + b
                particle_poses[labels == i, :] = (
                    particle_poses[labels == i, :] + self.d_lambda * flow_vector.T
                )

            # recalculate linearization point
            particle_poses_mean = np.mean(particle_poses, axis=0)

        posterior = ParticleState(state_vectors=particle_poses)
        end = time.time()
        comptime = end - start
        return posterior, comptime

    def _pam_clustering(self, flow_vectors, particles, covariance, cluster_num):
        inv_cov = np.linalg.inv(covariance)
        mahal_flow_vectors = pdist(
            np.subtract(flow_vectors, flow_vectors.mean(axis=1)[:, np.newaxis])
            @ sqrtm(inv_cov),
            "correlation",
        )

        mahal_particles = pdist(particles, "mahalanobis", VI=inv_cov)
        mahal_particles = (
            2 * (mahal_particles - min(mahal_particles)) / mahal_particles.ptp()
        )

        alfa = 0.5
        dist = squareform(alfa * mahal_particles + (1 - alfa) * mahal_flow_vectors)

        clusters = kmedoids.fasterpam(dist, cluster_num)
        labels = clusters.labels
        medoids = clusters.medoids
        return labels, medoids


class AEDHUpdater:
    # analytic EDH
    def __init__(self, measurement_model, particle_num):
        self.key = "aedh"

        self.measurement_model = measurement_model
        self.particle_num = particle_num

    def update(self, prediction, prediction_covar, measurement):
        start = time.time()

        num_of_rays = len(measurement)
        measurement_covar = self.measurement_model.range_noise_std**2 * np.eye(
            num_of_rays
        )

        particle_poses = prediction.state_vectors
        particle_poses_mean = prediction.mean()

        # linearize measurement
        cd, grad_cd_x, grad_cd_z, _ = self.measurement_model.process_detection(
            particle_poses_mean, measurement
        )

        # transform measurement
        y = -cd + grad_cd_x.T @ particle_poses_mean

        # calculate analytic flow parameters
        M = prediction_covar @ (grad_cd_x @ grad_cd_x.T)
        p = grad_cd_x.T @ prediction_covar @ grad_cd_x
        w = (
            prediction_covar
            @ grad_cd_x
            * np.linalg.inv(
                grad_cd_z.T @ measurement_covar @ grad_cd_z
            )  # TODO only a division
            * y
        )
        r = grad_cd_z.T @ measurement_covar @ grad_cd_z
        mu = np.trace(M)

        fi = np.eye(3) + M / mu * (np.sqrt(r / (p + r)) - 1)
        fib0 = 2 / 3 * w / p * (p + r) - 2 / 3 * w / p * r * np.sqrt(r / (p + r))
        fib1 = M / p @ particle_poses_mean[:, np.newaxis] * (np.sqrt(r / (p + r)) - 1)
        fib2 = 2 * w / p * (r - p / 2) - 2 / p * w * r * np.sqrt(r / (r + p))
        fib3 = 2 / p * M @ particle_poses_mean[:, np.newaxis] * (p / 2 + r) / (
            p + r
        ) - 2 / p * M @ particle_poses_mean[:, np.newaxis] * r / np.sqrt(r * (p + r))
        fib4 = 1 / 3 * w / p * (p**2 - 4 * p * r - 8 * r**2) / (
            p + r
        ) + 1 / 3 * w / p * 8 * r**2 / np.sqrt(r) / np.sqrt(p + r)

        # update particles
        particle_poses = fi @ particle_poses.T + (fib0 + fib1 + fib2 + fib3 + fib4)
        particle_poses = particle_poses.T

        posterior = ParticleState(state_vectors=particle_poses)
        end = time.time()
        comptime = end - start
        return posterior, comptime


class NAEDHUpdater:
    # N-step analytic EDH
    def __init__(self, measurement_model, step_num, particle_num):
        self.key = "naedh"

        self.measurement_model = measurement_model
        self.particle_num = particle_num
        self.step_num = step_num

    def update(self, prediction, prediction_covar, measurement):
        start = time.time()
        num_of_rays = len(measurement)
        measurement_covar = self.measurement_model.range_noise_std**2 * np.eye(
            num_of_rays
        )

        particle_poses = prediction.state_vectors
        particle_poses_mean = prediction.mean()
        particle_poses_mean_0 = particle_poses_mean

        steps = np.linspace(0, 1, self.step_num + 1)
        for i in range(self.step_num):
            # linearize about the mean
            cd, grad_cd_x, grad_cd_z, _ = self.measurement_model.process_detection(
                particle_poses_mean, measurement
            )

            # transform measurement
            y = -cd + grad_cd_x.T @ particle_poses_mean

            # calculate analytic flow parameters
            M = prediction_covar @ (grad_cd_x @ grad_cd_x.T)
            p = grad_cd_x.T @ prediction_covar @ grad_cd_x
            r = grad_cd_z.T @ measurement_covar @ grad_cd_z
            w = prediction_covar @ grad_cd_x * np.linalg.inv(r) * y

            lam_0 = steps[i]
            lam_1 = steps[i + 1]

            kl0 = lam_0 * p + r
            kl1 = lam_1 * p + r

            fi = np.eye(3) + M / p * (np.sqrt(kl0 / kl1) - 1)
            fib2 = (
                w
                / p
                * (
                    -1 / 3 * kl1
                    + 3 * r
                    - kl1 ** (-1 / 2) * kl0 ** (1 / 2) * (3 * r - 1 / 3 * kl0)
                )
            )
            fib3 = (
                M
                @ particle_poses_mean_0[:, np.newaxis]
                * r
                / p
                * (kl1 ** (-1) - kl0 ** (-1 / 2) * kl1 ** (-1 / 2))
            )
            fib4 = (
                1
                / 3
                / p
                * w
                * kl1 ** (-1 / 2)
                * (
                    (p**2 * lam_1**2 - 4 * p * r * lam_1 - 8 * r**2)
                    * ((p * lam_1 + r) ** (-1 / 2))
                    - (p**2 * lam_0**2 - 4 * p * r * lam_0 - 8 * r**2)
                    * ((p * lam_0 + r) ** (-1 / 2))
                )
            )

            # update particles
            particle_poses = fi @ particle_poses.T + (fib2 + fib3 + fib4)
            particle_poses = particle_poses.T

            # recalculate linearization point
            particle_poses_mean = np.mean(particle_poses, axis=0)

        posterior = ParticleState(state_vectors=particle_poses)
        end = time.time()
        comptime = end - start
        return posterior, comptime
