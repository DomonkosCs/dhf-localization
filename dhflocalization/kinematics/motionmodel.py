from abc import ABC, abstractmethod
import numpy as np

from ..customtypes import StateHypothesis, ParticleState


class MotionModel(ABC):
    @abstractmethod
    def propagate(self, state):
        raise NotImplementedError


class OdometryMotionModel(MotionModel):
    def __init__(self, alfas, rng=np.random.default_rng()):
        self.alfa_1 = alfas[0]
        self.alfa_2 = alfas[1]
        self.alfa_3 = alfas[2]
        self.alfa_4 = alfas[3]
        self.rng = rng
        pass

    def propagate_particles(self, prior, control_input):
        """
        states: n by 3 numpy array
        """

        delta_rot_1, delta_trans, delta_rot_2 = self.transform_control_input(
            control_input
        )

        control_covar = self.calc_control_noise_covar(
            delta_rot_1, delta_trans, delta_rot_2
        )

        # sample the transformed control inputs for each state individually
        control_covar = np.zeros((3, 3))  #! No noise
        delta_hat_rot_1 = self.calc_angle_diff(
            delta_rot_1,
            np.sqrt(control_covar[0, 0])
            * self.rng.standard_normal(prior.state_vectors.shape[0]),
        )
        delta_hat_trans = self.calc_angle_diff(
            delta_trans,
            np.sqrt(control_covar[1, 1])
            * self.rng.standard_normal(prior.state_vectors.shape[0]),
        )
        delta_hat_rot_2 = self.calc_angle_diff(
            delta_rot_2,
            np.sqrt(control_covar[2, 2])
            * self.rng.standard_normal(prior.state_vectors.shape[0]),
        )

        state_angles = prior.state_vectors[:, 2]

        diff_x = np.multiply(delta_hat_trans, np.cos(state_angles + delta_hat_rot_1))
        diff_y = np.multiply(delta_hat_trans, np.sin(state_angles + delta_hat_rot_1))
        diff_fi = delta_hat_rot_1 + delta_hat_rot_2

        diff_array = np.array([diff_x, diff_y, diff_fi]).T
        predicted_states = prior.state_vectors + diff_array

        return ParticleState(state_vectors=predicted_states)

    def transform_control_input(self, control_input):
        state_odom_prev = control_input[0]
        state_odom_now = control_input[1]

        # if np.linalg.norm(state_odom_now[1] - state_odom_prev[1]) < 0.01:
        #     delta_rot_1 = 0
        # else:
        delta_rot_1 = self.calc_angle_diff(
            np.arctan2(
                state_odom_now[1] - state_odom_prev[1],
                state_odom_now[0] - state_odom_prev[0],
            ),
            state_odom_prev[2],
        )

        delta_trans = np.sqrt(
            (state_odom_now[0] - state_odom_prev[0]) ** 2
            + (state_odom_now[1] - state_odom_prev[1]) ** 2
        )

        delta_rot_2 = self.calc_angle_diff(
            self.calc_angle_diff(state_odom_now[2], state_odom_prev[2]), delta_rot_1
        )

        return delta_rot_1, delta_trans, delta_rot_2

    def propagate(self, prior, control_input):
        delta_rot_1, delta_trans, delta_rot_2 = self.transform_control_input(
            control_input
        )
        control_covar = self.calc_control_noise_covar(
            delta_rot_1, delta_trans, delta_rot_2
        )

        fi = prior.state_vector[2]
        predicted_state = prior.state_vector + np.array(
            [
                delta_trans * np.cos(fi + delta_rot_1),
                delta_trans * np.sin(fi + delta_rot_1),
                delta_rot_1 + delta_rot_2,
            ]
        )

        jacobi_state, jacobi_input = self.calc_jacobians(delta_rot_1, delta_trans, fi)
        predicted_covar = (
            jacobi_state @ prior.covar @ jacobi_state.T
            + jacobi_input @ control_covar @ jacobi_input.T
        )

        return StateHypothesis(state_vector=predicted_state, covar=predicted_covar)

    def calc_jacobians(self, delta_rot_1, delta_trans, fi):

        J_state_13 = -delta_trans * np.sin(fi + delta_rot_1)
        J_state_23 = delta_trans * np.cos(fi + delta_rot_1)
        J_state = np.array([[1, 0, J_state_13], [0, 1, J_state_23], [0, 0, 1]])

        J_input_11 = -delta_trans * np.sin(fi + delta_rot_1)
        J_input_21 = delta_trans * np.cos(fi + delta_rot_1)
        J_input_12 = np.cos(fi + delta_rot_1)
        J_input_22 = np.sin(fi + delta_rot_1)

        J_input = np.array(
            [[J_input_11, J_input_12, 0], [J_input_21, J_input_22, 0], [1, 0, 1]]
        )

        return [J_state, J_input]

    def calc_control_noise_covar(self, delta_rot_1, delta_trans, delta_rot_2):

        # We want to treat backward and forward motion symmetrically for the
        # noise model to be applied below.  The standard model seems to assume
        # forward motion.
        # From https://github.com/ros-planning/navigation/blob/noetic-devel/amcl/src/amcl/sensors/amcl_odom.cpp
        delta_rot_1_noise = min(
            abs(self.calc_angle_diff(delta_rot_1, 0)),
            abs(self.calc_angle_diff(delta_rot_1, np.pi)),
        )
        delta_rot_2_noise = min(
            abs(self.calc_angle_diff(delta_rot_2, 0)),
            abs(self.calc_angle_diff(delta_rot_2, np.pi)),
        )

        control_var_11 = (
            self.alfa_1 * delta_rot_1_noise**2 + self.alfa_2 * delta_trans**2
        )
        control_var_22 = self.alfa_3 * delta_trans**2 + self.alfa_4 * (
            delta_rot_1_noise**2 + delta_rot_2_noise**2
        )
        control_var_33 = (
            self.alfa_1 * delta_rot_2_noise**2 + self.alfa_2 * delta_trans**2
        )

        return np.diag([control_var_11, control_var_22, control_var_33])

    def normalize_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def calc_angle_diff(self, a, b):
        a = self.normalize_angle(a)
        b = self.normalize_angle(b)
        d1 = a - b
        d2 = 2 * np.pi - abs(d1)

        if type(d1) is np.ndarray:
            return_arr = d2
            d2[d1 > 0] *= -1.0
            return_arr[abs(d1) < abs(d2)] = d1[abs(d1) < abs(d2)]
            return return_arr
        else:
            if d1 > 0:
                d2 *= -1.0
            if abs(d1) < abs(d2):
                return d1
            else:
                return d2
