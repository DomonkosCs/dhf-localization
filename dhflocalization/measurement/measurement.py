

from gridmap.grid_map import GridMap
from measurement.sensor import Detection
from state.state import StateHypothesis
import numpy as np


class Measurement:
    def __init__(self, ogm: GridMap, range_noise_std):
        self.ogm = ogm
        self.range_noise_std = range_noise_std

    def processDetection(self, state: StateHypothesis, measurement):
        # measurement = [(fi_0,r_0),...,(fi_i,r_i)]

        ranges = [ray[1] for ray in measurement]
        angles = [ray[0] for ray in measurement]

        x_o = np.zeros([len(ranges), 2])
        ogm = self.ogm

        r_cos = np.multiply(ranges, np.cos(
            angles+state.pose[2]))
        r_sin = np.multiply(ranges, np.sin(
            angles+state.pose[2]))
        x_o[:, 0] = r_cos + state.pose[0, 0]
        x_o[:, 1] = r_sin + state.pose[1, 0]

        df = [ogm.calc_distance_transform_xy_pos(x, y) for x, y in x_o]
        cd = np.asarray(df).mean()
        df_d_x = ogm.calc_distance_function_derivate_interp(
            x_o[:, 0], x_o[:, 1], 1, 0)
        df_d_y = ogm.calc_distance_function_derivate_interp(
            x_o[:, 0], x_o[:, 1], 0, 1)
        cd_d_x = df_d_x.mean()
        cd_d_y = df_d_y.mean()
        cd_d_fi = (np.multiply(df_d_x, -r_sin) +
                   np.multiply(df_d_y, r_cos)).mean()
        grad_cd_x = np.array([[cd_d_x, cd_d_y, cd_d_fi]]).T  # grad_hx

        grad_cd_z = np.array([(df_d_x * np.cos(angles+state.pose[2]) +
                               df_d_y * np.sin(angles+state.pose[2])) * 1/len(angles)]).T

        return cd, grad_cd_x, grad_cd_z, x_o
