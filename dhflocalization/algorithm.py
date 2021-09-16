#!/usr/bin/env python

# %%
import matplotlib
from numpy.lib import stride_tricks
from measurement.measurement import Measurement
from measurement.sensor import Detection, Sensor
from state.state import RobotState
from gridmap.grid_map import GridMap
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# %%


def transform_to_global(measurements, robot_state):
    return np.array([robot_state[0] + measurements[:, 1]*np.cos(measurements[:, 0]+robot_state[2]),
                     robot_state[1] + measurements[:, 1] *
                     np.sin(measurements[:, 0]+robot_state[2])
                     ]).T
# %%


# test = GridMap.load_grid_map_from_csv('map_test.csv', 1, 0, 0)
# test.plot_grid_map()
# fig = plt.figure()
# test.plot_distance_transform(fig)
# test.plot_distance_transform_interp(fig)
# grid_data = np.reshape(np.array(test.data), (test.height, test.width))
# edt = ndimage.distance_transform_edt(1 - grid_data)
# edt_interp = RectBivariateSpline(
#     np.arange(test.width), np.arange(test.height), edt)

# %%
ogm = GridMap.load_grid_map_from_csv('map_test.csv', 1, 0, 0)
sensor = Sensor(0, 2, 10, 0)
state = RobotState(0, -0.75, -2.25, 0)
detection = Detection(sensor=sensor, timestamp=0)
m = Measurement(detection=detection, state=state, ogm=ogm)

# %%


def getHelpers(state, direction, ogm):
    tileX, tileY = ogm.get_xy_index_from_xy_pos(state.x, state.y)
    dirX = direction[0]
    dirY = direction[1]
    dTileX = 1 if (dirX > 0) else -1
    dtX = ((tileX + (0.5*dTileX + 0.5) -
           ogm.get_xy_index_from_xy_pos(0, 0)[0]) * ogm.resolution - state.x) / dirX

    dTileY = 1 if (dirY > 0) else -1
    # dtY = ((tileY + (0.5*dTileY + 0.5)) *
    #        ogm.resolution - state.y + ogm.height/2)
    dtY = ((tileY + (0.5*dTileY + 0.5) -
           ogm.get_xy_index_from_xy_pos(0, 0)[1]) * ogm.resolution - state.y) / dirY

    return tileX, tileY, dTileX, dTileY, dtX, dtY,  dTileX * ogm.resolution / dirX, dTileY * ogm.resolution / dirY


def raycast(ogm, direction, state):
    tileX, tileY, dTileX, dTileY, dtX, dtY, ddtX, ddtY = getHelpers(
        state, direction, ogm)

    t = 0
    while ((tileX > 0) & (tileX <= ogm.width) & (tileY > 0) & (tileY <= ogm.height)):
        if (dtX < dtY):
            tileX = tileX + dTileX
            dt = dtX
            t = t + dt
            dtX = dtX + ddtX - dt
            dtY = dtY - dt
        else:
            tileY = tileY + dTileY
            dt = dtY
            t = t + dt
            dtX = dtX - dt
            dtY = dtY + ddtY - dt

        if (ogm.get_value_from_xy_index(tileX, tileY)):
            break

    return t, tileX, tileY


state = RobotState(0, -1, 0, 0)
t, tileX, tileY = raycast(
    ogm, [np.cos(np.deg2rad(134.9)), np.sin(np.deg2rad(134.9))], state)
print(t, tileX, tileY)
# %%
