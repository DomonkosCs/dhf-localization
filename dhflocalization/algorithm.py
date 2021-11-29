#!/usr/bin/env python

# %%
import matplotlib
from visualization.plotter import Plotter
from filters.edh import EDH
from gridmap.processpgm import PgmProcesser
from kinematics.motionmodel import OdometryMotionModel
from measurement.measurement import Measurement
from rawdata.loadsimudata import RawDataLoader
from state.state import StateHypothesis
from rawdata.resultdatahandler import resultExporter, resultLoader
from filters.ekf import EKF
from gridmap.grid_map import GridMap
import numpy as np
import matplotlib.pyplot as plt
""" import cProfile
import pstats
import re """
%load_ext autoreload
%autoreload 2

matplotlib.rcParams['text.usetex'] = True
%matplotlib widget
# %%


map_fn = '/Users/domonkoscsuzdi/Desktop/Research/Localization/code/dhflocalization/resources/tb3_house_true.pgm'
simu_data_fn = '/Users/domonkoscsuzdi/Desktop/Research/Localization/code/dhflocalization/resources/5hz_000001.json'

ogm = GridMap.load_grid_map_from_array(
    PgmProcesser.read_pgm(map_fn), 0.05, 10, 10.05)

motion_model = OdometryMotionModel([0.1, 0.1, 0.1, 0.1])
measurement_model = Measurement(ogm, 0.01)
ekf = EKF(motion_model, measurement_model)
edh = EDH(motion_model, measurement_model)
x_odom, measurement, x_true, x_amcl = RawDataLoader.loadFromJson(
    simu_data_fn)

particle_num = 1

edh_filtered_states = []
ekf_filtered_states = []

# Draw from the prior
init_particle_mean = [-3.0, 1.0, 0]
init_particle_covar = [[0.1**2, 0, 0], [0, 0.1**2, 0], [0, 0, 0.05**2]]
init_state = StateHypothesis.init_from_particle_prior(
    particle_num, init_particle_mean, init_particle_covar, use_init_covar=True)
edh_filtered_states.append(init_state)
ekf_filtered_states.append(init_state)
# %%
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

amcl_filtered_states = [StateHypothesis(
    amcl_pose) for amcl_pose in np.array(x_amcl)]

odom_states = [StateHypothesis(np.array(odom_pose)+np.array([-3, 1, 0]))
               for odom_pose in x_odom]

true_states = [StateHypothesis(true_pose)
               for true_pose in x_true]

filtered_states = {"edh": edh_filtered_states,
                   "ekf": ekf_filtered_states, "amcl": amcl_filtered_states}
reference_states = {"odom": odom_states, "true": true_states}

resultExporter().save(filtered_states, reference_states)
# %%

results = resultLoader().load('11-29-17_05')


def calcPoseFromStateArray(filtered_states, reference_states):
    filtered_poses = {}
    for key, value in filtered_states.items():
        filtered_poses[key] = \
            np.array([state.pose for state in value])
    reference_poses = {}
    for key, value in reference_states.items():
        reference_poses[key] = \
            np.array([state.pose for state in value])

    return filtered_poses, reference_poses


filtered_poses, reference_poses = calcPoseFromStateArray(
    results[0], results[1])

# %%


def calcErrorMetrics(filtered_poses, reference_poses):
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

    err_mean_sqare = {"position": {}, "orientation": {}}
    for key, value in filtered_poses.items():
        value = np.array(value)
        err_mean_sqare["position"][key] = np.sqrt(np.mean(np.linalg.norm(
            reference_poses["true"][:, :-1]-value[:, :-1], axis=1)**2))
        err_mean_sqare["orientation"][key] = np.sqrt(
            np.mean(np.array(angle_set_diff(reference_poses["true"][:, 2], value[:, 2]))**2))

    err_mean_abs = {"position": {}, "orientation": {}}
    for key, value in filtered_poses.items():
        err_mean_abs["position"][key] = np.mean(np.linalg.norm(
            reference_poses["true"][:, :-1]-value[:, :-1], axis=1))
        err_mean_abs["orientation"][key] = np.mean(
            np.abs(np.array(angle_set_diff(reference_poses["true"][:, 2], value[:, 2]))))

    std = {"position": {}, "orientation": {}}
    for key, value in filtered_poses.items():
        std["position"][key] = np.std(np.linalg.norm(
            reference_poses["true"][:, :-1]-value[:, :-1], axis=1))
        std["orientation"][key] = np.std(
            np.array(angle_set_diff(reference_poses["true"][:, 2], value[:, 2])))
    return err_mean_sqare, err_mean_abs, std


err_mean_sqare, err_mean_abs, std = calcErrorMetrics(
    filtered_poses, reference_poses)
print(err_mean_sqare)
print(err_mean_abs)
print(std)
# %%
map_fn = '/Users/domonkoscsuzdi/Desktop/Research/Localization/code/dhflocalization/resources/tb3_house_true.pgm'

ogm = GridMap.load_grid_map_from_array(
    PgmProcesser.read_pgm(map_fn), 0.05, 10, 10.05)
plotter = Plotter()
plotter.background_map = ogm

plotter.plot_tracks(filtered_states['edh'], [0, 1], particle=True)

# %%

# ['odom','truth']
# ['ekf','edh']


def plotTracks(filtered_names, reference_names):

    map_fn = '/Users/domonkoscsuzdi/Desktop/Research/Localization/code/dhflocalization/resources/tb3_house_true.pgm'

    ogm = GridMap.load_grid_map_from_array(
        PgmProcesser.read_pgm(map_fn), 0.05, 10, 10.05)
    plotter = Plotter()
    plotter.background_map = ogm

    plotter.ax.set_xlabel(r'$x\,(\mathrm{cell})$')
    plotter.ax.set_ylabel(r'$y\,(\mathrm{cell})$')

    for name in reference_names:
        plotter.plot_ground_truths(reference_states[name], [
            0, 1], truths_label=name, linestyle=":" if name == 'odom' else "-")

    for name in filtered_names:
        plotter.plot_tracks(
            filtered_states[name], [0, 1], marker=None, linestyle='--', track_label=name)


plotTracks(['true'], ['ekf', 'edh', 'amcl'])
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
