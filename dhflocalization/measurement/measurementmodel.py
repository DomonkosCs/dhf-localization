from ..gridmap import GridMap
import numpy as np


class MeasurementModel:
    def __init__(
        self,
        ogm: GridMap,
        range_noise_std,
        robot_sensor_dx=0,
        robot_sensor_dy=0,
        robot_sensor_dyaw=0,
    ):
        self.ogm = ogm
        self.range_noise_std = range_noise_std
        self.robot_sensor_tr = np.array(
            [robot_sensor_dx, robot_sensor_dy, robot_sensor_dyaw]
        )

    def process_detection(self, state_vector, measurement):

        ranges = [ray[1] for ray in measurement]
        angles = [ray[0] for ray in measurement]

        x_o = np.zeros([len(ranges), 2])
        ogm = self.ogm

        # Transform readings from sensor frame to global frame.
        # First, transform to the robot's frame, and then to the global.
        angle_global = angles + self.robot_sensor_tr[2] + state_vector[2]
        r_cos = np.multiply(ranges, np.cos(angle_global))
        r_sin = np.multiply(ranges, np.sin(angle_global))
        x_o[:, 0] = r_cos + self.robot_sensor_tr[0] + state_vector[0]
        x_o[:, 1] = r_sin + self.robot_sensor_tr[1] + state_vector[1]

        df = ogm.calc_distance_transform_xy_pos(x_o)

        cd = np.mean(df)
        df_d_x, df_d_y = ogm.calc_distance_function_derivate_interp(x_o)
        cd_d_x = df_d_x.mean()
        cd_d_y = df_d_y.mean()
        cd_d_fi = (np.multiply(df_d_x, -r_sin) + np.multiply(df_d_y, r_cos)).mean()
        grad_cd_x = np.array([[cd_d_x, cd_d_y, cd_d_fi]]).T  # grad_hx

        grad_cd_z = np.array(
            [
                (df_d_x * np.cos(angle_global) + df_d_y * np.sin(angle_global))
                * 1
                / len(angles)
            ]
        ).T

        return cd, grad_cd_x, grad_cd_z, x_o
