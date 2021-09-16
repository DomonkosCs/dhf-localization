#!/usr/bin/env python

# %%
import matplotlib
from measurement.measurement import Measurement
from measurement.sensor import Detection, Sensor
from state.state import RobotState
from gridmap.grid_map import GridMap
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
%matplotlib widget
# %%


def transform_to_global(measurements, robot_state):
    return np.array([robot_state[0] + measurements[:, 1]*np.cos(measurements[:, 0]+robot_state[2]),
                     robot_state[1] + measurements[:, 1] *
                     np.sin(measurements[:, 0]+robot_state[2])
                     ]).T
# %%


test = GridMap.load_grid_map_from_csv('map_test.csv', 1, 0, 0)
test.plot_grid_map()
fig = plt.figure()
test.plot_distance_transform(fig)
test.plot_distance_transform_interp(fig)
grid_data = np.reshape(np.array(test.data), (test.height, test.width))
edt = ndimage.distance_transform_edt(1 - grid_data)
edt_interp = RectBivariateSpline(
    np.arange(test.width), np.arange(test.height), edt)

# %%
ogm = GridMap.load_grid_map_from_csv('map_test.csv', 1, 0, 0)
sensor = Sensor(0, 2, 10, 0)
state = RobotState(5, 5, 0, 0)
detection = Detection(sensor=sensor, timestamp=0)
m = Measurement(detection=detection, state=state, ogm=ogm)

# %%
