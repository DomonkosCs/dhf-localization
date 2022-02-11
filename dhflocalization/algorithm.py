#!/usr/bin/env python
# %%
# import matplotlib
# from visualization import Plotter
from filters import EDH
from gridmap import PgmProcesser
from kinematics import OdometryMotionModel
from measurement import Measurement
from rawdata import RawDataLoader, ConfigExporter

from customtypes import StateHypothesis

from rawdata import resultExporter
from filters import EKF
from gridmap import GridMap
import numpy as np

# import matplotlib.pyplot as plt

if __name__ == "__main__":

    print("Starting state estimation...")

    # Exports every variable starting with cfg_ to a config YAML file.
    config_exporter = ConfigExporter()

    cfg_random_seed = 2021
    np.random.seed(cfg_random_seed)

    cfg_map_filename = "tb3_house_lessnoisy"
    cfg_simu_data_filename = "5hz_005"

    simulation_data = RawDataLoader.loadFromJson(cfg_simu_data_filename)
    ogm = GridMap.load_grid_map_from_array(
        PgmProcesser.read_pgm(cfg_map_filename),
        resolution=0.05,
        center_x=10,
        center_y=10.05,
    )

    cfg_odometry_alpha_1 = 0.1
    cfg_odometry_alpha_2 = 0.1
    cfg_odometry_alpha_3 = 0.1
    cfg_odometry_alpha_4 = 0.1

    motion_model = OdometryMotionModel(
        [
            cfg_odometry_alpha_1,
            cfg_odometry_alpha_2,
            cfg_odometry_alpha_3,
            cfg_odometry_alpha_4,
        ]
    )

    cfg_measurement_range_noise_std = 0.01

    measurement_model = Measurement(ogm, cfg_measurement_range_noise_std)

    cfg_edh_particle_number = 1
    cfg_edh_lambda_number = 10
    cfg_init_gaussian_mean = np.array([-3.0, 1.0, 0])
    cfg_init_gaussian_covar = np.array(
        [[0.1**2, 0, 0], [0, 0.1**2, 0], [0, 0, 0.05**2]]
    )

    ekf = EKF(motion_model, measurement_model)
    edh = EDH(
        motion_model=motion_model,
        measurement_model=measurement_model,
        particle_num=cfg_edh_particle_number,
        lambda_num=cfg_edh_lambda_number,
    )
    edh.init_particles_from_gaussian(
        cfg_init_gaussian_mean, cfg_init_gaussian_covar, return_state=False
    )

    # Another option is to set the return_state flag on edh.init_particles_from_gaussian,
    # and use returned state to initialize ekf.
    ekf.init_state(mean=cfg_init_gaussian_mean, covar=cfg_init_gaussian_covar)

    for i in range(1, simulation_data.simulation_steps, 1):
        control_input = [simulation_data.x_odom[i - 1], simulation_data.x_odom[i]]
        measurement = simulation_data.measurement[i]

        edh.propagate(control_input)
        ekf_propagated_state = ekf.propagate(control_input, return_state=True)

        edh.update(ekf_propagated_state.covar, measurement)
        ekf.update(measurement)

    amcl_filtered_states = [
        StateHypothesis(amcl_pose) for amcl_pose in simulation_data.x_amcl
    ]

    odom_states = [
        StateHypothesis(odom_pose + np.array([-3, 1, 0]))
        for odom_pose in simulation_data.x_odom
    ]

    true_states = [StateHypothesis(true_pose) for true_pose in simulation_data.x_true]

    filtered_states = {
        # "edh": edh.filtered_states,
        "ekf": ekf.filtered_states,
        "amcl": amcl_filtered_states,
    }
    reference_states = {"odom": odom_states, "true": true_states}

    print("Calcuations completed, saving results...")
    cfg_result_filename = resultExporter().save(filtered_states, reference_states)
    config_exporter.export(locals(), cfg_result_filename)

# #%%


# # %%
# from rawdata import resultExporter, resultLoader

# results = resultLoader().load("02-11-12_02")


# def calcPoseFromStateArray(filtered_states, reference_states):
#     filtered_poses = {}
#     for key, value in filtered_states.items():
#         filtered_poses[key] = np.array([state.pose for state in value])
#     reference_poses = {}
#     for key, value in reference_states.items():
#         reference_poses[key] = np.array([state.pose for state in value])

#     return filtered_poses, reference_poses


# filtered_poses, reference_poses = calcPoseFromStateArray(results[0], results[1])

# # %%


# def calcErrorMetrics(filtered_poses, reference_poses):
#     def normalize_angle(angle):
#         return np.arctan2(np.sin(angle), np.cos(angle))

#     def calc_angle_diff(a, b):
#         a = normalize_angle(a)
#         b = normalize_angle(b)
#         d1 = a - b
#         d2 = 2 * np.pi - abs(d1)
#         if d1 > 0:
#             d2 *= -1.0
#         if abs(d1) < abs(d2):
#             return d1
#         else:
#             return d2

#     def angle_set_diff(set_1, set_2):
#         return [calc_angle_diff(a, b) for a, b in zip(set_1, set_2)]

#     err_mean_sqare = {"position": {}, "orientation": {}}
#     for key, value in filtered_poses.items():
#         value = np.array(value)
#         err_mean_sqare["position"][key] = np.sqrt(
#             np.mean(
#                 np.linalg.norm(reference_poses["true"][:, :-1] - value[:, :-1], axis=1)
#                 ** 2
#             )
#         )
#         err_mean_sqare["orientation"][key] = np.sqrt(
#             np.mean(
#                 np.array(angle_set_diff(reference_poses["true"][:, 2], value[:, 2]))
#                 ** 2
#             )
#         )

#     err_mean_abs = {"position": {}, "orientation": {}}
#     for key, value in filtered_poses.items():
#         err_mean_abs["position"][key] = np.mean(
#             np.linalg.norm(reference_poses["true"][:, :-1] - value[:, :-1], axis=1)
#         )
#         err_mean_abs["orientation"][key] = np.mean(
#             np.abs(np.array(angle_set_diff(reference_poses["true"][:, 2], value[:, 2])))
#         )

#     std = {"position": {}, "orientation": {}}
#     for key, value in filtered_poses.items():
#         std["position"][key] = np.std(
#             np.linalg.norm(reference_poses["true"][:, :-1] - value[:, :-1], axis=1)
#         )
#         std["orientation"][key] = np.std(
#             np.array(angle_set_diff(reference_poses["true"][:, 2], value[:, 2]))
#         )
#     return err_mean_sqare, err_mean_abs, std


# err_mean_sqare, err_mean_abs, std = calcErrorMetrics(filtered_poses, reference_poses)
# print(err_mean_sqare)
# print(err_mean_abs)
# print(std)
# # %%
# map_fn = '/Users/domonkoscsuzdi/Desktop/Research/Localization/code/dhflocalization/resources/tb3_house_true.pgm'

# ogm = GridMap.load_grid_map_from_array(
#     PgmProcesser.read_pgm(map_fn), 0.05, 10, 10.05)
# plotter = Plotter()
# plotter.background_map = ogm

# plotter.plot_tracks(filtered_states['edh'], [0, 1], particle=False)

# # %%

# # ['odom','truth']
# # ['ekf','edh']


# def plotTracks(reference_names, filtered_names):

#     map_fn = \
#         '/Users/domonkoscsuzdi/Desktop/Research/Localization/code/dhflocalization/resources/tb3_house_true.pgm'

#     ogm = GridMap.load_grid_map_from_array(
#         PgmProcesser.read_pgm(map_fn), 0.05, 10, 10.05)
#     plotter = Plotter()
#     plotter.background_map = ogm

#     plotter.ax.set_xlabel(r'$x\,(\mathrm{cell})$')
#     plotter.ax.set_ylabel(r'$y\,(\mathrm{cell})$')

#     for name in reference_names:
#         plotter.plot_ground_truths(reference_states[name], [
#             0, 1], truths_label=name, linestyle=":" if name == 'odom' else "-")

#     for name in filtered_names:
#         plotter.plot_tracks(
#             filtered_states[name], [0, 1], marker=None, linestyle='--', track_label=name)


# plotTracks(['true'], ['edh', 'ekf'])
# # %%

# time = np.linspace(0, 569.987-6.821, reference_poses['true'][:, 0].shape[0])
# bottom_lim = -0.25
# top_lim = 0.25
# plt.figure(figsize=(10, 5))
# ax = plt.gca()
# plt.subplot(311)
# ax = plt.gca()
# ax.grid(1)
# ax.set_ylim(bottom_lim, top_lim)
# ax.set_ylabel(r'$\Delta x\,(\mathrm{m})$')
# plt.plot(time, reference_poses['true'][:, 0]-filtered_poses['ekf'][:, 0],
#          label="EKF", color="black", linewidth=0.6)
# plt.legend()
# plt.subplot(312)
# ax = plt.gca()
# ax.grid(1)
# ax.set_ylim(bottom_lim, top_lim)
# ax.set_ylabel(r'$\Delta x\,(\mathrm{m})$')
# plt.plot(time, reference_poses['true'][:, 0]-filtered_poses['amcl'][:, 0],
#          label="AMCL", color="black", linewidth=0.6)
# plt.legend()

# plt.subplot(313)
# plt.plot(time, reference_poses['true'][:, 0]-filtered_poses['edh'][:, 0],
#          label="EDH", color="black", linewidth=0.6)
# ax = plt.gca()
# plt.legend()

# ax.grid(1)
# ax.set_ylim(bottom_lim, top_lim)
# ax.set_xlabel(r'$t\,(\mathrm{s})$')
# ax.set_ylabel(r'$\Delta x\,(\mathrm{m})$')
# # %%
# bottom_lim = -0.25
# top_lim = 0.25
# plt.figure(figsize=(10, 5))
# ax = plt.gca()
# plt.subplot(311)
# ax = plt.gca()
# ax.grid(1)
# ax.set_ylim(bottom_lim, top_lim)
# ax.set_ylabel(r'$\Delta y\,(\mathrm{m})$')
# plt.plot(time, reference_poses['true'][:, 1]-filtered_poses['ekf'][:, 1],
#          label="EKF", color="black", linewidth=0.6)
# plt.legend()
# plt.subplot(312)
# ax = plt.gca()
# ax.grid(1)
# ax.set_ylim(bottom_lim, top_lim)
# ax.set_ylabel(r'$\Delta y\,(\mathrm{m})$')
# plt.plot(time, reference_poses['true'][:, 1]-filtered_poses['amcl'][:, 1],
#          label="AMCL", color="black", linewidth=0.6)
# plt.legend()

# plt.subplot(313)
# plt.plot(time, reference_poses['true'][:, 1]-filtered_poses['edh'][:, 1],
#          label="EDH", color="black", linewidth=0.6)
# ax = plt.gca()
# plt.legend()

# ax.grid(1)
# ax.set_ylim(bottom_lim, top_lim)
# ax.set_xlabel(r'$t\,(\mathrm{s})$')
# ax.set_ylabel(r'$\Delta y\,(\mathrm{m})$')
# # %%


# def angle_set_diff(set_1, set_2):
#     return [calc_angle_diff(a, b) for a, b in zip(set_1, set_2)]


# def normalize_angle(angle):
#     return np.arctan2(np.sin(angle), np.cos(angle))


# def calc_angle_diff(a, b):
#     a = normalize_angle(a)
#     b = normalize_angle(b)
#     d1 = a-b
#     d2 = 2*np.pi - abs(d1)
#     if(d1 > 0):
#         d2 *= -1.0
#     if(abs(d1) < abs(d2)):
#         return d1
#     else:
#         return d2


# bottom_lim = -0.15
# top_lim = 0.15
# plt.figure(figsize=(10, 5))
# ax = plt.gca()
# plt.subplot(311)
# ax = plt.gca()
# ax.grid(1)
# ax.set_ylim(bottom_lim, top_lim)
# ax.set_ylabel(r'$\Delta \theta\,(\mathrm{rad})$')
# plt.plot(time, angle_set_diff(reference_poses['true'][:, 2], filtered_poses['ekf'][:, 2]),
#          label="EKF", color="black", linewidth=0.6)
# plt.legend()
# plt.subplot(312)
# ax = plt.gca()
# ax.grid(1)
# ax.set_ylim(bottom_lim, top_lim)
# ax.set_ylabel(r'$\Delta \theta\,(\mathrm{rad})$')
# plt.plot(time, angle_set_diff(reference_poses['true'][:, 2], filtered_poses['amcl'][:, 2]),
#          label="AMCL", color="black", linewidth=0.6)
# plt.legend()

# plt.subplot(313)
# plt.plot(time, angle_set_diff(reference_poses['true'][:, 2], filtered_poses['edh'][:, 2]),
#          label="EDH", color="black", linewidth=0.6)
# ax = plt.gca()
# plt.legend()

# ax.grid(1)
# ax.set_ylim(bottom_lim, top_lim)
# ax.set_xlabel(r'$t\,(\mathrm{s})$')
# ax.set_ylabel(r'$\Delta \theta\,(\mathrm{rad})$')
# # %%
# %load_ext snakeviz
# %snakeviz - -new-tab foo()

# %%
