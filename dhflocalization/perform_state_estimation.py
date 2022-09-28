#!/usr/bin/env python
# %%
# import matplotlib
# from visualization import Plotter
from dhflocalization.filters import EDH, CEDH
from dhflocalization.gridmap import PgmProcesser
from dhflocalization.kinematics import OdometryMotionModel
from dhflocalization.measurement import MeasurementModel, MeasurementProcessor
from dhflocalization.rawdata import RawDataLoader, ConfigExporter

from dhflocalization.customtypes import StateHypothesis, ParticleState

from dhflocalization.rawdata import resultExporter
from dhflocalization.filters import EKF
from dhflocalization.gridmap import GridMap
import numpy as np

from dhflocalization import perform_evaluation as evaluate

# import matplotlib.pyplot as plt


def main():
    print("Starting state estimation...")

    # Exports every variable starting with cfg_ to a config YAML file.
    config_exporter = ConfigExporter()

    cfg_random_seed = 2021
    np.random.seed(cfg_random_seed)

    cfg_map_filename = "tb3_house_lessnoisy"
    cfg_map_resolution = 0.05  # m/cell

    cfg_simu_data_filename = "5hz_0001"

    do_plotting = True

    simulation_data = RawDataLoader.load_from_json(cfg_simu_data_filename)
    ogm = GridMap.load_grid_map_from_array(
        PgmProcesser.read_pgm(cfg_map_filename),
        resolution=cfg_map_resolution,
        center_x=10,
        center_y=10.05,
    )

    cfg_max_ray_number = 500
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
    measurement_model = MeasurementModel(ogm, cfg_measurement_range_noise_std)

    cfg_edh_particle_number = 1000
    cfg_edh_lambda_number = 10
    cfg_init_gaussian_mean = np.array([-3.0, 1.0, 0])
    cfg_init_gaussian_covar = np.array(
        [[0.1**2, 0, 0], [0, 0.1**2, 0], [0, 0, 0.05**2]]
    )

    measurement_processer = MeasurementProcessor(max_ray_number=cfg_max_ray_number)
    ekf = EKF(motion_model, measurement_model)
    edh = EDH(
        motion_model=motion_model,
        measurement_model=measurement_model,
        particle_num=cfg_edh_particle_number,
        lambda_num=cfg_edh_lambda_number,
    )

    prior = ParticleState.init_from_gaussian(
        cfg_init_gaussian_mean, cfg_init_gaussian_covar, cfg_edh_particle_number
    )
    ekf_prior = StateHypothesis(
        state_vector=cfg_init_gaussian_mean, covar=cfg_init_gaussian_covar
    )

    ekf_track = []
    edh_track = []
    for i in range(1, simulation_data.simulation_steps, 1):
        control_input = [simulation_data.x_odom[i - 1], simulation_data.x_odom[i]]
        measurement = measurement_processer.filter_measurements(
            simulation_data.measurement[i]
        )
        prediction = edh.propagate(prior, control_input)
        ekf_prediction = ekf.propagate(ekf_prior, control_input)

        posterior = edh.update(prediction, ekf_prediction, measurement)
        ekf_posterior = ekf.update(ekf_prediction, measurement)

        ekf_track.append(ekf_posterior.state_vector)
        edh_track.append(posterior.mean())
        prior = posterior
        ekf_prior = ekf_posterior

    amcl_filtered_states = [
        StateHypothesis(amcl_pose) for amcl_pose in simulation_data.x_amcl
    ]

    odom_states = [
        StateHypothesis(odom_pose + np.array([-3, 1, 0]))
        for odom_pose in simulation_data.x_odom
    ]

    true_states = [StateHypothesis(true_pose) for true_pose in simulation_data.x_true]

    filtered_states = {
        "edh": edh.filtered_states,
        "ekf": ekf.filtered_states,
        "amcl": amcl_filtered_states,
    }
    reference_states = {"odom": odom_states, "true": true_states}

    # TODO move to results
    cfg_avg_ray_number = measurement_processer.get_avg_ray_number()
    cfg_edh_runtime = edh.get_runtime()
    print("Calcuations completed, saving results...")
    cfg_result_filename = resultExporter().save(filtered_states, reference_states)
    config_exporter.export(locals(), cfg_result_filename)

    if do_plotting:
        evaluate.main(cfg_result_filename)


if __name__ == "__main__":
    main()
