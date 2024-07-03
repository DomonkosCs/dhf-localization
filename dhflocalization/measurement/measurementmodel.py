from ..gridmap import GridMap
import numpy as np
import matplotlib.pyplot as plt


class MeasurementModel:
    def __init__(
        self,
        ogm: GridMap,
        range_noise_std,
        robot_sensor_dx=0,
        robot_sensor_dy=0,
        outlier_threshold=3,
    ):
        self.ogm = ogm
        self.range_noise_std = range_noise_std
        self.robot_sensor_tr = np.array([robot_sensor_dx, robot_sensor_dy])
        self.outlier_threshold = outlier_threshold

    def _filter_outliers(self, df):
        # Using MAD and robust z-score to filter outliers

        median_df = np.median(df)
        mad_df = np.median(np.abs(df - median_df))

        # Calculate the modified z-scores of df
        m_z_scores_df = 0.6745 * (df - median_df) / mad_df

        # Get boolean array where True indicates the value is not an outlier
        not_outliers = np.abs(m_z_scores_df) < self.outlier_threshold
        return not_outliers

    def process_detection(self, state_vector, measurement):

        ranges = [ray[1] for ray in measurement]
        angles = [ray[0] for ray in measurement]

        x_o = np.zeros([len(ranges), 2])
        ogm = self.ogm

        # Transform readings from sensor frame to global frame.
        # First, transform to the robot's frame, and then to the global.
        angle_global = angles + state_vector[2]
        r_cos = np.multiply(ranges, np.cos(angle_global))
        r_sin = np.multiply(ranges, np.sin(angle_global))
        x_o[:, 0] = (
            r_cos
            + np.cos(state_vector[2]) * self.robot_sensor_tr[0]
            - np.sin(state_vector[2]) * self.robot_sensor_tr[1]
            + state_vector[0]
        )
        x_o[:, 1] = (
            r_sin
            + np.sin(state_vector[2]) * self.robot_sensor_tr[0]
            + np.cos(state_vector[2]) * self.robot_sensor_tr[1]
            + state_vector[1]
        )

        df = ogm.calc_distance_transform_xy_pos(x_o)

        not_outliers = self._filter_outliers(df)

        # Check outliers
        # for i in range(len(x_o)):
        #     plt.annotate(str(round(df[i], 4)), (x_o[i, 0], x_o[i, 1]))
        # plt.scatter(x_o[not_outliers, 0], x_o[not_outliers, 1], s=2)

        # ogm.plot_grid_map()
        # plt.scatter(x_o[:, 0], x_o[:, 1], s=2)

        # Exclude outliers
        df_no_outliers = df[not_outliers]
        x_o_no_outliers = x_o[not_outliers]
        cd_no_outliers = np.mean(df_no_outliers)
        r_sin_no_outliers = r_sin[not_outliers]
        r_cos_no_outliers = r_cos[not_outliers]
        angle_global_no_outliers = np.array(angles)[not_outliers] + state_vector[2]

        df_d_x, df_d_y = ogm.calc_distance_function_derivate_interp(x_o_no_outliers)
        cd_d_x = df_d_x.mean()
        cd_d_y = df_d_y.mean()
        cd_d_fi = (
            np.multiply(df_d_x, -r_sin_no_outliers)
            + np.multiply(df_d_y, r_cos_no_outliers)
        ).mean()
        grad_cd_x = np.array([[cd_d_x, cd_d_y, cd_d_fi]]).T  # grad_hx

        grad_cd_z = np.array(
            [
                (
                    df_d_x * np.cos(angle_global_no_outliers)
                    + df_d_y * np.sin(angle_global_no_outliers)
                )
                * 1
                / len(df_d_x)
            ]
        ).T

        return cd_no_outliers, grad_cd_x, grad_cd_z, x_o_no_outliers
