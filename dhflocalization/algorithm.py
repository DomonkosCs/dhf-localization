#!/usr/bin/env python

# %%
from visualization.plotter import Plotter
from os import stat
import matplotlib
from filters.edh import EDH
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
import matplotlib.pyplot as plt
from matplotlib import cm
import cProfile
import pstats
import re
%matplotlib widget
# %%
map_fn = '/Users/domonkoscsuzdi/Desktop/Research/Localization/code/dhflocalization/resources/tb3_house_true.pgm'
simu_data_fn = '/Users/domonkoscsuzdi/Desktop/Research/Localization/code/dhflocalization/resources/topicexport_2.json'

ogm = GridMap.load_grid_map_from_array(
    PgmProcesser.read_pgm(map_fn), 0.05, 10, 10.05)

plotter = Plotter()
plotter.background_map = ogm

motion_model = OdometryMotionModel([0.1, 0.1, 0.1, 0.1])
measurement_model = Measurement(ogm, 0.03)
ekf = EKF(motion_model, measurement_model)
edh = EDH(motion_model, measurement_model)
x_odom, measurement, x_true = RawDataLoader.loadFromJson(simu_data_fn)

particle_num = 500

# dhf_particle_sets = []
edh_states = []
ekf_filtered_states = []

# Draw from the prior
init_particle_mean = [-3, 1, 0]
init_particle_covar = [[0.2**2, 0, 0], [0, 0.2**2, 0], [0, 0, 0.02**2]]
# init_particle_set = ParticleSet.init_from_prior(
#     particle_num, init_particle_mean, init_particle_covar)
init_state = StateHypothesis.init_from_particle_prior(
    particle_num, init_particle_mean, init_particle_covar)
edh_states.append(init_state)
# dhf_particle_sets.append(init_particle_set)
ekf_filtered_states.append(init_state)


for i in range(1, len(x_odom), 1):
    edh_state = edh.propagate(
        edh_states[-1], [x_odom[i-1], x_odom[i]])
    ekf_state = ekf.propagate(
        ekf_filtered_states[-1], [x_odom[i-1], x_odom[i]])

    edh_state = edh.update(
        edh_state, ekf_state.covar, measurement[i])
    ekf_state = ekf.update(ekf_state, measurement[i])

    ekf_filtered_states.append(ekf_state)
    edh_states.append(edh_state)

plotter.plot_ground_truths(
    edh_states, [0, 1], truths_label="Filtered", linestyle="dotted")

plotter.plot_ground_truths(
    ekf_filtered_states, [0, 1], truths_label="Filtered", linestyle="dotted")

plotter.plot_ground_truths([StateHypothesis(np.array(odom_pose)+np.array([-3, 1, 0]))
                           for odom_pose in x_odom], [0, 1], truths_label="Odom", linestyle="--")
plotter.plot_ground_truths([StateHypothesis(true_pose)
                           for true_pose in x_true], [0, 1], truths_label="True", linestyle="-")
# %%
dhf_poses = [
    dhf_filtered_state.pose for dhf_filtered_state in dhf_filtered_states]

dhf_poses = np.array(dhf_poses).squeeze()[:, :-1]

ekf_poses = [
    ekf_filtered_state.pose for ekf_filtered_state in ekf_filtered_states]

ekf_poses = np.array(ekf_poses).squeeze()[:, :-1]

true_poses = np.array(x_true)[:, :-1]

print(np.sqrt(np.mean(np.linalg.norm(true_poses-dhf_poses, axis=1)**2)))
print(np.sqrt(np.mean(np.linalg.norm(true_poses-ekf_poses, axis=1)**2)))


# ! TODO Szöget normalizálni.
# %%
%load_ext snakeviz
%snakeviz - -new-tab foo()
