
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from state.state import StateHypothesis


class MotionModel(ABC):
    @abstractmethod
    def propagate(self, state):
        raise NotImplementedError


class VelocityMotionModel(MotionModel):
    def __init__(self):
        pass

    def propagate(self, state, control_input):
        dt = state.timestep
        dw = 1e-11/dt
        x = state.state[0]
        y = state.state[1]
        fi = state.state[2]
        v = control_input[0]
        w = control_input[1] + dw  # avoid divison by 0

        x_next = x - v/w*np.sin(fi) + v/w*np.sin(fi + w*dt)
        y_next = y + v/w*np.cos(fi) - v/w*np.cos(fi + w*dt)
        fi_next = fi + w*dt

        return np.array([x_next, y_next, fi_next])

    def calcJacobian(self, state, control_input):
        dt = state.timestep
        dw = 1e-11/dt
        x = state.state[0]
        y = state.state[1]
        fi = state.state[2]
        v = control_input[0]
        w = control_input[1] + dw  # avoid divison by 0

        df1_dfi = - v/w*np.cos(fi) + v/w*np.cos(fi + w*dt)
        df2_dfi = - v/w*np.sin(fi) + v/w*np.sin(fi + w*dt)

        jacob_x = np.array([[1, 0, df1_dfi], [0, 1, df2_dfi], [0, 0, 1]])

        df1_dv = -1/w*np.sin(fi) + 1/w*np.sin(fi+w*dt)
        df1_dw = v*w**(-2)*np.sin(fi) + v/w*dt * \
            np.cos(fi + w*dt) - v*w**(-2)*np.cos(fi + w*dt)

        df2_dv = 1/w*np.cos(fi) - 1/w*np.cos(fi+w*dt)
        df2_dw = -v*w**(-2)*np.cos(fi) + v/w*dt * \
            np.sin(fi + w*dt) + v*w**(-2)*np.cos(fi + w*dt)


class OdometryMotionModel(MotionModel):
    def __init__(self, alfas):
        self.alfa_1 = alfas[0]
        self.alfa_2 = alfas[1]
        self.alfa_3 = alfas[2]
        self.alfa_4 = alfas[3]
        pass

    def propagate(self, state: StateHypothesis, control_input):

        state_odom_prev = control_input[0]
        state_odom_now = control_input[1]
        fi = state.pose[2, 0]

        delta_rot_1 = np.arctan2(
            state_odom_now[1] - state_odom_prev[1],
            state_odom_now[0] - state_odom_prev[0]) - state_odom_prev[2]

        delta_trans = np.sqrt(
            (state_odom_now[0] - state_odom_prev[0])**2
            + (state_odom_now[1] - state_odom_prev[1])**2)

        delta_rot_2 = state_odom_now[2] - state_odom_prev[2] - delta_rot_1

        prop_pose = state.pose + np.matrix([delta_trans*np.cos(fi + delta_rot_1),
                                            delta_trans *
                                            np.sin(fi + delta_rot_1),
                                            delta_rot_1 + delta_rot_2]).T

        jacobi_state, jacobi_input = self.calcJacobians(
            delta_rot_1, delta_trans, fi)
        control_covar = self.calcControlNoiseCovar(
            delta_rot_1, delta_rot_2, delta_trans)

        prop_covar = jacobi_state*state.covar*jacobi_state.T
        + jacobi_input * control_covar * jacobi_input.T

        return prop_pose, prop_covar

    def calcJacobians(self, delta_rot_1, delta_trans, fi):

        J_state_13 = -delta_trans*np.sin(fi + delta_rot_1)
        J_state_23 = delta_trans*np.cos(fi + delta_rot_1)
        J_state = np.matrix(
            [[1, 0, J_state_13], [0, 1, J_state_23], [0, 0, 0]])

        J_input_11 = -delta_trans*np.sin(fi + delta_rot_1)
        J_input_21 = delta_trans*np.cos(fi + delta_rot_1)
        J_input_12 = np.cos(fi + delta_rot_1)
        J_input_22 = np.sin(fi + delta_rot_1)

        J_input = np.matrix([[J_input_11, J_input_12, 0],
                             [J_input_21, J_input_22, 0],
                             [1, 0, 1]])

        return [J_state, J_input]

    def calcControlNoiseCovar(self, delta_rot_1, delta_rot_2, delta_trans):

        control_covar_11 = self.alfa_1 * \
            abs(delta_rot_1) + self.alfa_2 * delta_trans
        control_covar_22 = self.alfa_3 * delta_trans + \
            self.alfa_4*(abs(delta_rot_1) + abs(delta_rot_2))
        control_covar_33 = self.alfa_1 * \
            abs(delta_rot_2) + self.alfa_2 * delta_trans

        return np.diag([control_covar_11, control_covar_22, control_covar_33])
