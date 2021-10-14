#!/usr/bin/env python

# %%
from os import stat
import matplotlib
from gridmap.processpgm import PgmProcesser
from kinematics.motionmodel import OdometryMotionModel, VelocityMotionModel
from measurement.measurement import Measurement
from measurement.sensor import Detection, Sensor
from rawdata.loadsimudata import RawDataLoader
from state.state import StateHypothesis
from filters.ekf import EKF
from gridmap.grid_map import GridMap
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
from matplotlib import cm
import cProfile
import pstats
import re

from visualization.plotter import Plotter
%matplotlib widget
# %%


map_fn = '/Users/domonkoscsuzdi/Desktop/Research/Localization/code/dhflocalization/resources/tb3_house_true.pgm'
simu_data_fn = '/Users/domonkoscsuzdi/Desktop/Research/Localization/code/dhflocalization/resources/topicexport_10hz.json'

map = GridMap.load_grid_map_from_array(
    PgmProcesser.read_pgm(map_fn), 0.05, 10, 10.05)

# map.plot_grid_map()
odom_mot_model = OdometryMotionModel([0.1, 0.1, 0.1, 0.1])
measurement_model = Measurement(map, 0.03)
ekf = EKF(odom_mot_model, measurement_model)
states = []

init_covar = np.matrix([[0.5**2, 0, 0], [0, 0.5**2, 0], [0, 0, 0.3**2]])
init_state = StateHypothesis(
    np.matrix([-3.5, 1.5, 0]).T, init_covar, 0)
states.append(init_state)
x_odom, measurement, x_true = RawDataLoader.loadFromJson(simu_data_fn)
plotter = Plotter()
plotter.background_map = map

for i in range(1, len(x_odom), 1):
    state = ekf.propagate(states[-1], [x_odom[i-1], x_odom[i]])
    state, ray_endpoints = ekf.update(state, measurement[i])
    ray_endpoints_pixel = (
        ray_endpoints - [map.left_lower_x, map.left_lower_y])/0.05
    # plotter.ax.plot(ray_endpoints_pixel[:, 0],
    #                 ray_endpoints_pixel[:, 1], ".", zorder=2)
    # plotter.ax.quiver((state.pose[0, 0] - map.left_lower_x) /
    #                   0.05, (state.pose[1, 0] - map.left_lower_y)/0.05, np.cos(state.pose[2, 0]), np.sin(state.pose[2, 0]), scale=100, zorder=3)
    states.append(state)

plotter.plot_ground_truths(
    states, [0, 1], truths_label="Filtered", linestyle="dotted")
plotter.plot_ground_truths([StateHypothesis(np.asmatrix(odom_pose).T+np.matrix(
    [-3, 1, 0]).T, None, 0) for odom_pose in x_odom], [0, 1], truths_label="Odom", linestyle="--")
plotter.plot_ground_truths([StateHypothesis(np.asmatrix(odom_pose).T, None, 0)
                           for odom_pose in x_true], [0, 1], truths_label="True", linestyle="-")


# %%

""" %load_ext snakeviz
%snakeviz - -new-tab foo()
 """
# %%
