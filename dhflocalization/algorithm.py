#!/usr/bin/env python

# %%
from os import stat
import matplotlib
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
# %matplotlib widget
# %%
map_fn = '/Users/domonkoscsuzdi/Desktop/Research/Localization/code/dhflocalization/resources/tb3_house_true.pgm'
simu_data_fn = '/Users/domonkoscsuzdi/Desktop/Research/Localization/code/dhflocalization/resources/topicexport.json'

ogm = GridMap.load_grid_map_from_array(
    PgmProcesser.read_pgm(map_fn), 0.05, 10, 10.05)

plotter = Plotter()
plotter.background_map = ogm

# map.plot_grid_map()
motion_model = OdometryMotionModel([0.1, 0.1, 0.1, 0.1])
measurement_model = Measurement(ogm, 0.03)
ekf = EKF(motion_model, measurement_model)
x_odom, measurement, x_true = RawDataLoader.loadFromJson(simu_data_fn)

particle_num = 500
d_lambda = 0.1
lambdas = np.linspace(d_lambda, 1, 10)
dhf_filtered_states = []
ekf_filtered_states = []
# Draw from the prior
init_particle_mean = [-3.5, 1.5, 0]
init_particle_covar = [[0.5**2, 0, 0], [0, 0.5**2, 0], [0, 0, 0.05**2]]
init_particle_poses = np.random.multivariate_normal(
    init_particle_mean, init_particle_covar, particle_num)

init_ekf_covar = np.cov(init_particle_poses, rowvar=False)
init_ekf_mean = np.mean(init_particle_poses, axis=0)
init_state = StateHypothesis(
    np.array([init_ekf_mean]).T, np.array(init_ekf_covar))

dhf_filtered_states.append(init_state)
ekf_filtered_states.append(init_state)
particle_poses = init_particle_poses


for i in range(1, len(x_odom), 1):
    particle_poses = motion_model.propagate_particles(
        particle_poses, [x_odom[i-1], x_odom[i]])
    particle_poses_mean = np.mean(particle_poses, axis=0)
    ekf_state = ekf.propagate(
        ekf_filtered_states[-1], [x_odom[i-1], x_odom[i]])

    measurement_covar = measurement_model.range_noise_std**2 * \
        np.eye(len(measurement[i]))

    for l in lambdas:
        cd, grad_cd_x, grad_cd_z, _ = measurement_model.processDetection(
            StateHypothesis(np.array([particle_poses_mean]).T, None), measurement[i])

        y = - cd + grad_cd_x.T @ particle_poses_mean
        B = -0.5*ekf_state.covar @ grad_cd_x / \
            (l*grad_cd_x.T @ ekf_state.covar @ grad_cd_x +
             grad_cd_z.T @ measurement_covar @ grad_cd_z) @ grad_cd_x.T

        b = (np.eye(3) + 2*l*B) @ \
            ((np.eye(3) + l*B) @ ekf_state.covar @ grad_cd_x /
             (grad_cd_z.T @ measurement_covar @ grad_cd_z) * y + B @ np.array([particle_poses_mean]).T)

        particle_poses = particle_poses + d_lambda * (np.array(
            [B @ particle_state for particle_state in particle_poses]) + b.T)
        particle_poses_mean = np.mean(particle_poses, axis=0)

    ekf_state, _ = ekf.update(ekf_state, measurement[i])

    ekf_filtered_states.append(ekf_state)
    dhf_filtered_states.append(StateHypothesis(
        np.array([particle_poses_mean]).T, None))

plotter.plot_ground_truths(
    dhf_filtered_states, [0, 1], truths_label="Filtered", linestyle="dotted")

plotter.plot_ground_truths(
    ekf_filtered_states, [0, 1], truths_label="Filtered", linestyle="dotted")


plotter.plot_ground_truths([StateHypothesis(np.asmatrix(odom_pose).T+np.matrix(
    [-3, 1, 0]).T, None, 0) for odom_pose in x_odom], [0, 1], truths_label="Odom", linestyle="--")
plotter.plot_ground_truths([StateHypothesis(np.asmatrix(odom_pose).T, None, 0)
                           for odom_pose in x_true], [0, 1], truths_label="True", linestyle="-")
# %%
# %load_ext snakeviz
# %snakeviz - -new-tab foo()
