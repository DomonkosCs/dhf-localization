#!/usr/bin/env python

# %%
from os import stat
import matplotlib
from filters.edh import EDH
from gridmap.processpgm import PgmProcesser
from kinematics.motionmodel import OdometryMotionModel, VelocityMotionModel
from measurement.measurement import Measurement
from measurement.sensor import Detection, Sensor
from rawdata.loadsimudata import RawDataLoader
from state.particle import ParticleSet
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

from visualization.plotter import Plotter
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
# d_lambda = 0.1
# lambdas = np.linspace(d_lambda, 1, 10)
dhf_particle_sets = []
ekf_filtered_states = []

# Draw from the prior
init_particle_mean = [-3, 1, 0]
init_particle_covar = [[0.2**2, 0, 0], [0, 0.2**2, 0], [0, 0, 0.02**2]]
init_particle_set = ParticleSet.init_from_prior(
    particle_num, init_particle_mean, init_particle_covar)

dhf_particle_sets.append(init_particle_set)
ekf_filtered_states.append(init_particle_set.mean_state)


for i in range(1, len(x_odom), 1):
    particle_set = edh.propagate(
        dhf_particle_sets[-1], [x_odom[i-1], x_odom[i]])
    ekf_state = ekf.propagate(
        ekf_filtered_states[-1], [x_odom[i-1], x_odom[i]])

    particle_set = edh.update(
        particle_set, ekf_state.covar, measurement[i])
    ekf_state = ekf.update(ekf_state, measurement[i])

    ekf_filtered_states.append(ekf_state)
    dhf_particle_sets.append(particle_set)


dhf_filtered_states = [
    particle_set.mean_state for particle_set in dhf_particle_sets]

plotter.plot_ground_truths(
    dhf_filtered_states, [0, 1], truths_label="Filtered", linestyle="dotted")

plotter.plot_ground_truths(
    ekf_filtered_states, [0, 1], truths_label="Filtered", linestyle="dotted")


plotter.plot_ground_truths([StateHypothesis(np.asmatrix(odom_pose).T+np.matrix(
    [-3, 1, 0]).T, None) for odom_pose in x_odom], [0, 1], truths_label="Odom", linestyle="--")
plotter.plot_ground_truths([StateHypothesis(np.asmatrix(true_pose).T, None)
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
