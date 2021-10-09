

from gridmap.grid_map import GridMap
from measurement.sensor import Detection
from state.state import RobotState
import numpy as np


class Measurement:
    def __init__(self, detection: Detection, state: RobotState, ogm: GridMap):
        self.detection = detection
        self.state = state
        self.ogm = ogm
        self.processDetection()

    def processDetection(self):
        x_o = np.zeros_like(self.detection.z)
        z = self.detection.z
        ogm = self.ogm

        r_cos = np.multiply(z[:, 1], np.cos(
            z[:, 0]+self.state.state[2]))
        r_sin = np.multiply(z[:, 1], np.sin(
            z[:, 0]+self.state.state[2]))
        x_o[:, 0] = r_cos + self.state.state[0]
        x_o[:, 1] = r_sin + self.state.state[1]
        self.x_o = x_o

        df_d_x = ogm.calc_distance_function_derivate_interp(
            x_o[:, 0], x_o[:, 1], 1, 0)
        df_d_y = ogm.calc_distance_function_derivate_interp(
            x_o[:, 0], x_o[:, 1], 0, 1)
        cd_d_x = df_d_x.mean()
        cd_d_y = df_d_y.mean()
        cd_d_fi = (np.multiply(cd_d_x, -r_sin) +
                   np.multiply(cd_d_y, r_cos)).mean()
        self.grad_cd_x = [cd_d_x, cd_d_y, cd_d_fi]  # grad_hx

        self.grad_cd_z = np.multiply((np.multiply(df_d_x, np.cos(z[:, 0]+self.state.state[2])) +
                                      np.multiply(df_d_y, np.sin(z[:, 0]+self.state.state[2]))), 1/z.shape[0])
