#!/usr/bin/env python
# %%
# import matplotlib
# from visualization import Plotter
from dhflocalization.filters import EDH
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
    rng = np.random.default_rng(cfg_random_seed)

    cfg_map_filename = "sztaki_true"
    cfg_map_resolution = 0.05  # m/cell

    cfg_simu_data_filename = "real"

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
        ],
        rng=rng,
    )

    cfg_measurement_range_noise_std = 0.11
    measurement_model = MeasurementModel(ogm, cfg_measurement_range_noise_std)

    cfg_edh_particle_number = 100
    cfg_edh_lambda_number = 10
    # cfg_init_gaussian_mean = np.array([-3.0, 1.0, 0])
    cfg_init_gaussian_mean = np.array([0, 0, 0])
    cfg_init_gaussian_covar = np.array(
        [[0.51**2, 0, 0], [0, 0.51**2, 0], [0, 0, 0.15**2]]
    )

    measurement_processer = MeasurementProcessor(max_ray_number=cfg_max_ray_number)
    ekf = EKF(measurement_model)
    edh = EDH(
        measurement_model=measurement_model,
        particle_num=cfg_edh_particle_number,
        lambda_num=cfg_edh_lambda_number,
    )

    prior = ParticleState.init_from_gaussian(
        cfg_init_gaussian_mean,
        cfg_init_gaussian_covar,
        cfg_edh_particle_number,
        rng=rng,
    )
    ekf_prior = StateHypothesis(
        state_vector=cfg_init_gaussian_mean, covar=cfg_init_gaussian_covar
    )

    ekf_track = []
    edh_track = []
    ekf_track.append(ekf_prior.state_vector)
    edh_track.append(prior.mean())

    for i in range(1, simulation_data.simulation_steps, 1):
        control_input = [simulation_data.x_odom[i - 1], simulation_data.x_odom[i]]
        measurement = measurement_processer.filter_measurements(
            simulation_data.measurement[i]
        )

        # propagate particles and perform ekf prediction
        prediction = motion_model.propagate_particles(prior, control_input)
        ekf_prediction = motion_model.propagate(ekf_prior, control_input)

        posterior = edh.update_mean_flow(prediction, ekf_prediction, measurement)
        ekf_posterior = ekf.update(ekf_prediction, measurement)

        ekf_track.append(ekf_posterior.state_vector)
        edh_track.append(posterior.mean())
        ekf_prior = ekf_posterior
        prior = posterior

    filtered_states = {
        "edh": np.asarray(edh_track),
        "ekf": np.asarray(ekf_track),
        # "amcl": simulation_data.x_amcl,
    }

    reference_states = {
        "odom": simulation_data.x_odom,
        # "odom": simulation_data.x_odom + np.array([-3, 1, 0]),
        "true": simulation_data.x_true,
    }

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
