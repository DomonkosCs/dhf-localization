#!/usr/bin/env python

# %%
import matplotlib
from numpy.lib import stride_tricks
from gridmap.processpgm import PgmProcesser
from kinematics.motionmodel import VelocityMotionModel
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
ogm = GridMap.load_grid_map_from_csv('map_test.csv', 1, 0, 0)
vel_mot_model = VelocityMotionModel()
sensor = Sensor(0, 20, 400, 0)
state = RobotState(0.01, [5, 5, 0], vel_mot_model)
detection = Detection(state, ogm, sensor, timestamp=0)
m = Measurement(detection=detection, state=state, ogm=ogm)
# %%
state.next_step([10, 0])
print(state.state)
state.next_step([0, 100*np.pi/2])
state.next_step([10, 0])
print(state.state)
state.next_step([0, 100*np.pi/2])
state.next_step([10, 0])
print(state.state)
state.next_step([0, 100*np.pi/2])
state.next_step([10, 0])
print(state.state)
state.next_step([0, 100*np.pi/2])
state.next_step([10, 0])
print(state.state)
state.next_step([0, 100*np.pi/2])
state.next_step([10, 0])
print(state.state)
state.next_step([0, 100*np.pi/2])
state.next_step([10, 0])
print(state.state)
state.next_step([0, 100*np.pi/2])
state.next_step([10, 0])
print(state.state)
state.next_step([0, 100*np.pi/2])
state.next_step([10, 0])
print(state.state)
state.next_step([0, 100*np.pi/2])
state.next_step([10, 0])
print(state.state)
state.next_step([0, 100*np.pi/2])
state.next_step([10, 0])
print(state.state)
state.next_step([0, 100*np.pi/2])
state.next_step([10, 0])
print(state.state)
state.next_step([0, 100*np.pi/2])
state.next_step([10, 0])
print(state.state)
state.next_step([0, 100*np.pi/2])
state.next_step([10, 0])
print(state.state)
# %%
fn = '/Users/domonkoscsuzdi/Desktop/Research/Localization/code/dhflocalization/resources/tb3_house_true.pgm'
map = GridMap.load_grid_map_from_array(PgmProcesser.read_pgm(fn), 0.05, 0, 0)
map.plot_grid_map()
# %%
