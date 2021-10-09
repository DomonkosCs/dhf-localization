

import numpy as np
from kinematics.motionmodel import MotionModel


class RobotState:
    def __init__(self, timestep, init_state: np.array, motion_model: MotionModel):
        self.timestamp = 0
        self.timestep = timestep
        self.state = init_state
        self.motion_model = motion_model

    def next_step(self, control_input):
        self.state = self.motion_model.propagate(
            self, control_input)
        self.timestamp += self.timestep
