#!/usr/bin/env python

# %%
import re
import pstats
import cProfile
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline
from gridmap.grid_map import GridMap
from filters.ekf import EKF
from state.state import StateHypothesis
from rawdata.loadsimudata import RawDataLoader
from measurement.sensor import Detection, Sensor
from measurement.measurement import Measurement
from kinematics.motionmodel import OdometryMotionModel, VelocityMotionModel
from gridmap.processpgm import PgmProcesser
from filters.edh import EDH
from visualization.plotter import Plotter
from os import stat
import matplotlib
matplotlib.rcParams['text.usetex'] = True
%matplotlib widget
# %%


def foo():
    map_fn = '/Users/domonkoscsuzdi/Desktop/Research/Localization/code/dhflocalization/resources/tb3_house_true.pgm'
    simu_data_fn = '/Users/domonkoscsuzdi/Desktop/Research/Localization/code/dhflocalization/resources/5hz_005.json'

    ogm = GridMap.load_grid_map_from_array(
        PgmProcesser.read_pgm(map_fn), 0.05, 10, 10.05)

    motion_model = OdometryMotionModel([0.1, 0.1, 0.1, 0.1])
    measurement_model = Measurement(ogm, 0.01)
    ekf = EKF(motion_model, measurement_model)
    edh = EDH(motion_model, measurement_model)
    x_odom, measurement, x_true, x_amcl = RawDataLoader.loadFromJson(
        simu_data_fn)

    particle_num = 10

    edh_filtered_states = []
    ekf_filtered_states = []

    # Draw from the prior
    init_particle_mean = [-3.0, 1.0, 0]
    init_particle_covar = [[0.1**2, 0, 0], [0, 0.1**2, 0], [0, 0, 0.05**2]]
    init_state = StateHypothesis.init_from_particle_prior(
        particle_num, init_particle_mean, init_particle_covar)
    edh_filtered_states.append(init_state)
    ekf_filtered_states.append(init_state)

    for i in range(1, len(x_odom), 1):
        edh_state = edh.propagate(
            edh_filtered_states[-1], [x_odom[i-1], x_odom[i]])
        ekf_state = ekf.propagate(
            ekf_filtered_states[-1], [x_odom[i-1], x_odom[i]])

        edh_state = edh.update(
            edh_state, ekf_state.covar, measurement[i])
        ekf_state = ekf.update(ekf_state, measurement[i])
        ekf_filtered_states.append(ekf_state)
        edh_filtered_states.append(edh_state)

    edh_poses = [
        edh_state.pose for edh_state in edh_filtered_states]

    edh_poses = np.array(edh_poses).squeeze()

    ekf_poses = [
        ekf_filtered_state.pose for ekf_filtered_state in ekf_filtered_states]

    ekf_poses = np.array(ekf_poses).squeeze()

    true_poses = np.array(x_true)
    amcl_poses = np.array(x_amcl)

    amcl_filtered_states = [StateHypothesis(
        amcl_pose) for amcl_pose in amcl_poses]
    filter_states = {"edh": edh_filtered_states,
                     "ekf": ekf_filtered_states, "amcl": amcl_filtered_states}
# %%


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def calc_angle_diff(a, b):
    a = normalize_angle(a)
    b = normalize_angle(b)
    d1 = a-b
    d2 = 2*np.pi - abs(d1)
    if(d1 > 0):
        d2 *= -1.0
    if(abs(d1) < abs(d2)):
        return d1
    else:
        return d2


def angle_set_diff(set_1, set_2):
    return [calc_angle_diff(a, b) for a, b in zip(set_1, set_2)]


r_mean_sqare = {
    'ekf_rms': np.sqrt(np.mean(np.linalg.norm(
        true_poses[:, :-1]-ekf_poses[:, :-1], axis=1)**2)),
    'amcl_rms': np.sqrt(np.mean(np.linalg.norm(
        true_poses[:, :-1]-amcl_poses[:, :-1], axis=1)**2)),
    'edh_rms':
    np.sqrt(np.mean(np.linalg.norm(
        true_poses[:, :-1]-edh_poses[:, :-1], axis=1)**2))
}
mean_abs = {
    'ekf_ma': np.mean(np.linalg.norm(
        true_poses[:, :-1]-ekf_poses[:, :-1], axis=1)),
    'amcl_ma': np.mean(np.linalg.norm(
        true_poses[:, :-1]-amcl_poses[:, :-1], axis=1)),
    'edh_ma':
    np.mean(np.linalg.norm(
        true_poses[:, :-1]-edh_poses[:, :-1], axis=1))
}

std = {
    'ekf_std': np.std(np.linalg.norm(
        true_poses[:, :-1]-ekf_poses[:, :-1], axis=1)),
    'amcl_std': np.std(np.linalg.norm(
        true_poses[:, :-1]-amcl_poses[:, :-1], axis=1)),
    'edh_std': np.std(np.linalg.norm(
        true_poses[:, :-1]-edh_poses[:, :-1], axis=1))
}
print(r_mean_sqare)
print(mean_abs)
print(std)

r_mean_sqare_th = {
    'ekf_rms': np.sqrt(np.mean(np.array(angle_set_diff(true_poses[:, 2], ekf_poses[:, 2]))**2)),
    'amcl_rms': np.sqrt(np.mean(np.array(angle_set_diff(true_poses[:, 2], amcl_poses[:, 2]))**2)),
    'edh_rms': np.sqrt(np.mean(np.array(angle_set_diff(true_poses[:, 2], edh_poses[:, 2]))**2))
}
mean_abs_th = {
    'ekf_rms': np.mean(np.abs(np.array(angle_set_diff(true_poses[:, 2], ekf_poses[:, 2])))),
    'amcl_rms': np.mean(np.abs(np.array(angle_set_diff(true_poses[:, 2], amcl_poses[:, 2])))),
    'edh_rms': np.mean(np.abs(np.array(angle_set_diff(true_poses[:, 2], edh_poses[:, 2]))))
}

std_th = {
    'ekf_std': np.std(np.array(angle_set_diff(true_poses[:, 2], ekf_poses[:, 2]))),
    'amcl_std': np.std(np.array(angle_set_diff(true_poses[:, 2], amcl_poses[:, 2]))),
    'edh_std': np.std(np.array(angle_set_diff(true_poses[:, 2], edh_poses[:, 2])))
}
print('THETA')
print(r_mean_sqare_th)
print(mean_abs_th)
print(std_th)
# %%
filter = "amcl"

plotter = Plotter()
plotter.background_map = ogm


plotter.plot_ground_truths([StateHypothesis(np.array(odom_pose)+np.array([-3, 1, 0]))
                           for odom_pose in x_odom], [0, 1], truths_label="Odom", linestyle=":")
plotter.plot_ground_truths([StateHypothesis(true_pose)
                           for true_pose in x_true], [0, 1], truths_label="True", linestyle="-")
plotter.plot_tracks(
    filter_states[filter], [0, 1], linestyle='--', marker=None, color="black", track_label="AMCL")
plotter.ax.set_xlabel(r'$x\,(\mathrm{cell})$')
plotter.ax.set_ylabel(r'$y\,(\mathrm{cell})$')
# %%


time = np.linspace(0, 569.987-6.821, true_poses[:, 0].shape[0])
bottom_lim = -0.25
top_lim = 0.25
plt.figure(figsize=(10, 5))
ax = plt.gca()
plt.subplot(311)
ax = plt.gca()
ax.grid(1)
ax.set_ylim(bottom_lim, top_lim)
ax.set_ylabel(r'$\Delta x\,(\mathrm{m})$')
plt.plot(time, true_poses[:, 0]-ekf_poses[:, 0],
         label="EKF", color="black", linewidth=0.6)
plt.legend()
plt.subplot(312)
ax = plt.gca()
ax.grid(1)
ax.set_ylim(bottom_lim, top_lim)
ax.set_ylabel(r'$\Delta x\,(\mathrm{m})$')
plt.plot(time, true_poses[:, 0]-amcl_poses[:, 0],
         label="AMCL", color="black", linewidth=0.6)
plt.legend()

plt.subplot(313)
plt.plot(time, true_poses[:, 0]-edh_poses[:, 0],
         label="EDH", color="black", linewidth=0.6)
ax = plt.gca()
plt.legend()

ax.grid(1)
ax.set_ylim(bottom_lim, top_lim)
ax.set_xlabel(r'$t\,(\mathrm{s})$')
ax.set_ylabel(r'$\Delta x\,(\mathrm{m})$')
# %%
bottom_lim = -0.25
top_lim = 0.25
plt.figure(figsize=(10, 5))
ax = plt.gca()
plt.subplot(311)
ax = plt.gca()
ax.grid(1)
ax.set_ylim(bottom_lim, top_lim)
ax.set_ylabel(r'$\Delta y\,(\mathrm{m})$')
plt.plot(time, true_poses[:, 1]-ekf_poses[:, 1],
         label="EKF", color="black", linewidth=0.6)
plt.legend()
plt.subplot(312)
ax = plt.gca()
ax.grid(1)
ax.set_ylim(bottom_lim, top_lim)
ax.set_ylabel(r'$\Delta y\,(\mathrm{m})$')
plt.plot(time, true_poses[:, 1]-amcl_poses[:, 1],
         label="AMCL", color="black", linewidth=0.6)
plt.legend()

plt.subplot(313)
plt.plot(time, true_poses[:, 1]-edh_poses[:, 1],
         label="EDH", color="black", linewidth=0.6)
ax = plt.gca()
plt.legend()

ax.grid(1)
ax.set_ylim(bottom_lim, top_lim)
ax.set_xlabel(r'$t\,(\mathrm{s})$')
ax.set_ylabel(r'$\Delta y\,(\mathrm{m})$')
# %%
bottom_lim = -0.15
top_lim = 0.15
plt.figure(figsize=(10, 5))
ax = plt.gca()
plt.subplot(311)
ax = plt.gca()
ax.grid(1)
ax.set_ylim(bottom_lim, top_lim)
ax.set_ylabel(r'$\Delta \theta\,(\mathrm{rad})$')
plt.plot(time, angle_set_diff(true_poses[:, 2], ekf_poses[:, 2]),
         label="EKF", color="black", linewidth=0.6)
plt.legend()
plt.subplot(312)
ax = plt.gca()
ax.grid(1)
ax.set_ylim(bottom_lim, top_lim)
ax.set_ylabel(r'$\Delta \theta\,(\mathrm{rad})$')
plt.plot(time, angle_set_diff(true_poses[:, 2], amcl_poses[:, 2]),
         label="AMCL", color="black", linewidth=0.6)
plt.legend()

plt.subplot(313)
plt.plot(time, angle_set_diff(true_poses[:, 2], edh_poses[:, 2]),
         label="EDH", color="black", linewidth=0.6)
ax = plt.gca()
plt.legend()

ax.grid(1)
ax.set_ylim(bottom_lim, top_lim)
ax.set_xlabel(r'$t\,(\mathrm{s})$')
ax.set_ylabel(r'$\Delta \theta\,(\mathrm{rad})$')
# %%
%load_ext snakeviz
%snakeviz - -new-tab foo()
