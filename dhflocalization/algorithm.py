#!/usr/bin/env python

# %%
from gridmap.processpgm import PgmProcesser
from kinematics.motionmodel import OdometryMotionModel, VelocityMotionModel
from measurement.measurement import Measurement
from measurement.sensor import Detection, Sensor
from state.state import StateHypothesis
from filters.ekf import EKF
from gridmap.grid_map import GridMap
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
from matplotlib import cm
%matplotlib widget
# %%
ogm = GridMap.load_grid_map_from_csv('map_test.csv', 1, 0, 0)
vel_mot_model = VelocityMotionModel()
sensor = Sensor(0, 20, 400, 0)
state = RobotState(0.01, [5, 5, 0], vel_mot_model)
detection = Detection(state, ogm, sensor, timestamp=0)
m = Measurement(detection=detection, state=state, ogm=ogm)
# %%
fn = '/Users/domonkoscsuzdi/Desktop/Research/Localization/code/dhflocalization/resources/tb3_house_true.pgm'
map = GridMap.load_grid_map_from_array(PgmProcesser.read_pgm(fn), 0.05, 0, 0)
map.plot_grid_map()
# %%
fn = '/Users/domonkoscsuzdi/Desktop/Research/Localization/code/dhflocalization/resources/tb3_house_true.pgm'
map = GridMap.load_grid_map_from_array(PgmProcesser.read_pgm(fn), 0.05, 0, 0)

odom_mot_model = OdometryMotionModel([0.01, 0.01, 0.01, 0.01])
measurement_model = Measurement(map, 0.01)
ekf = EKF(odom_mot_model, measurement_model)
init_state = StateHypothesis(np.matrix([0, 0, 0]).T, 0.01*npm.eye(3), 0)

states = []
states.append(init_state)
ranges = [1, 1.1, 1.3, 0.9, 0.9, 1.1, 2.3]
angles = np.linspace(0, 2*np.pi, len(ranges), endpoint=False)
measurement = [(angle, range) for angle, range in zip(angles, ranges)]
for i in range(1):
    state = ekf.propagate(states[-1], [[0, 0, 0], [0.1, 0.2, 0.01]])
    state = ekf.update(state, measurement)
    states.append(state)

# %%
