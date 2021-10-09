
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from state.state import RobotState


class MotionModel(ABC):
    @abstractmethod
    def propagate(self, state):
        raise NotImplementedError


class VelocityMotionModel(MotionModel):
    def __init__(self):
        pass

    def propagate(self, state: RobotState, control_input):
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

    def calcJacobian(self, state: RobotState, control_input):
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
