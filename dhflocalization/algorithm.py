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
measurement_model = Measurement(map, 0.1)
ekf = EKF(odom_mot_model, measurement_model)
states = []

init_state = StateHypothesis(
    np.matrix([-3, 1, 0]).T, 0.1*npm.eye(3), 0)
states.append(init_state)
x_odom, measurement = RawDataLoader.loadFromJson(simu_data_fn)
for i in range(1, len(x_odom), 1):
    state = ekf.propagate(states[-1], [x_odom[i-1], x_odom[i]])
    state = ekf.update(state, measurement[i])
    states.append(state)

plotter = Plotter()
plotter.background_map = map
plotter.plot_ground_truths(states, [0, 1], truths_label="Filtered")
plotter.plot_ground_truths([StateHypothesis(np.asmatrix(odom_pose).T+np.matrix(
    [-3, 1, 0]).T, None, 0) for odom_pose in x_odom], [0, 1], truths_label="Odom", linestyle="-")

# %%

%load_ext snakeviz
%snakeviz - -new-tab foo()

# %%
