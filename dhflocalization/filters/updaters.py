from ..customtypes import ParticleState
import numpy as np
import time


class MEDHUpdater:
    # mean EDH (original)
    def __init__(self, measurement_model, lambda_num, lambda_type, particle_num):
        self.key = "medh"

        self.measurement_model = measurement_model
        self.particle_num = particle_num

        self.d_lambdas = self.init_lambdas(lambda_num, type=lambda_type)

    def init_lambdas(self, lambda_num, type):
        # returns a list of every delta_lambda
        if type == "lin":
            return [1 / lambda_num] * lambda_num
        elif type == "exp":
            d_lambdas = []
            b = 1.3  # base to use

            d_lambda_0 = (b - 1) / (b**lambda_num - 1)
            for i in range(lambda_num):
                d_lambda = d_lambda_0 * b**i
                d_lambdas.append(d_lambda)

            return d_lambdas
        else:
            print("Error: lambda division type should be either 'lin' or 'exp'!")

    def update(self, prediction, prediction_covar, measurement):
        start = time.time()

        particle_poses = prediction.state_vectors
        particle_poses_mean = prediction.mean()
        particle_poses_mean_0 = particle_poses_mean

        lamb = 0
        for d_lamb in self.d_lambdas:
            lamb += d_lamb
            # linearize measurement model about the mean
            cd, grad_cd_x, grad_cd_z, _ = self.measurement_model.process_detection(
                particle_poses_mean, measurement
            )
            num_of_rays = len(grad_cd_z)
            measurement_covar = self.measurement_model.range_noise_std**2 * np.eye(
                num_of_rays
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
            particle_poses = particle_poses + d_lamb * ((B @ particle_poses.T).T + b.T)

            # recalculate linearization point
            particle_poses_mean = np.mean(particle_poses, axis=0)

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

        particle_poses = prediction.state_vectors
        particle_poses_mean = prediction.mean()
        particle_poses_mean_0 = particle_poses_mean

        steps = np.linspace(0, 1, self.step_num + 1)
        for i in range(self.step_num):
            # linearize about the mean
            cd, grad_cd_x, grad_cd_z, _ = self.measurement_model.process_detection(
                particle_poses_mean, measurement
            )
            num_of_rays = len(grad_cd_z)
            measurement_covar = self.measurement_model.range_noise_std**2 * np.eye(
                num_of_rays
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
