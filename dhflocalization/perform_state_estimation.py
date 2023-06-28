#!/usr/bin/env python
# %%
import numpy as np

from dhflocalization.filters import EDH
from dhflocalization.filters.updaters import (
    MEDHUpdater,
    NAEDHUpdater,
)
from dhflocalization.kinematics import OdometryMotionModel
from dhflocalization.measurement import MeasurementModel, MeasurementProcessor
from dhflocalization.rawdata import RawDataLoader, ConfigExporter
from dhflocalization.customtypes import StateHypothesis, Track
from dhflocalization.rawdata import ResultExporter
from dhflocalization.filters import EKF
from dhflocalization.gridmap import GridMap
from dhflocalization import perform_evaluation


def main():
    print("Starting state estimation...")

    save_data = False
    do_evaluation = True
    do_plot = False

    cfg_random_seed = 4302948723190478  # 2021
    rng = np.random.default_rng(cfg_random_seed)

    # from ./resources/maps/
    cfg_map_config_filename = "gt_map_01_table"

    # from ./resources/simulations/
    cfg_simu_data_filename = "5hz_o1e-4_l1e-2_filtered"

    simulation_data = RawDataLoader.load_from_json(cfg_simu_data_filename)

    ogm = GridMap.load_map_from_config(cfg_map_config_filename)

    cfg_max_ray_number = 360
    odom_alpha = 0.1
    cfg_odometry_alpha_1 = odom_alpha
    cfg_odometry_alpha_2 = odom_alpha
    cfg_odometry_alpha_3 = odom_alpha
    cfg_odometry_alpha_4 = odom_alpha

    motion_model = OdometryMotionModel(
        [
            cfg_odometry_alpha_1,
            cfg_odometry_alpha_2,
            cfg_odometry_alpha_3,
            cfg_odometry_alpha_4,
        ],
        rng=rng,
    )

    cfg_measurement_range_noise_std = 0.02
    robot_sensor_dx = -0.032  # x offset between the origin of the robot and the lidar
    measurement_model = MeasurementModel(
        ogm, cfg_measurement_range_noise_std, robot_sensor_dx
    )

    # parameters for the MEDH,NAEDH,EKF filters
    cfg_medh_particle_number = 100
    cfg_naedh_particle_number = 100

    cfg_medh_lambda_number = 10
    cfg_naedh_step_number = 10

    cfg_init_gaussian_mean = np.array([-3.0, 1.0, 0.0])
    cfg_init_gaussian_covar = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.025]])

    particle_init_variables = [cfg_init_gaussian_mean, cfg_init_gaussian_covar, rng]

    measurement_processer = MeasurementProcessor(max_ray_number=cfg_max_ray_number)

    # init ekf
    ekf = EKF(measurement_model)
    ekf_prior = StateHypothesis(
        state_vector=cfg_init_gaussian_mean, covar=cfg_init_gaussian_covar
    )
    ekf_track = Track(ekf_prior)
    ekf_comptimes = []

    # init MEDH and NAEDH
    medh_updater = MEDHUpdater(
        measurement_model, cfg_medh_lambda_number, "exp", cfg_medh_particle_number
    )
    medh = EDH(medh_updater, *particle_init_variables)
    naedh_updater = NAEDHUpdater(
        measurement_model, cfg_naedh_step_number, cfg_naedh_particle_number
    )
    naedh = EDH(naedh_updater, *particle_init_variables)

    # which variant to use: either [naedh], [medh] or [naedh,medh]
    edh_variants = [naedh]

    for i in range(1, simulation_data.simulation_steps, 1):
        print("{}/{}".format(i, simulation_data.simulation_steps))
        control_input = [simulation_data.x_odom[i - 1], simulation_data.x_odom[i]]
        measurement = measurement_processer.filter_measurements(
            simulation_data.measurement[i]
        )

        # propagate particles and perform ekf prediction
        ekf_prediction = motion_model.propagate(ekf_prior, control_input)

        for filter in edh_variants:
            prior = filter.last_particle_posterior
            prediction = motion_model.propagate_particles(prior, control_input)

            prediction_covar = ekf_prediction.covar
            filter.update(prediction, prediction_covar, measurement)

        ekf_posterior, ekf_update_comptime = ekf.update(ekf_prediction, measurement)

        ekf_track.append(ekf_posterior)

        ekf_comptimes.append(ekf_update_comptime)
        ekf_prior = ekf_posterior

    filtered_results = {
        "ekf": {
            "track": ekf_track,
            "comptime": np.array(ekf_comptimes).mean(),
        },
    }

    for filter in edh_variants:
        filtered_results.update(filter.get_results())

    print("Calculations completed")

    cfg_avg_ray_number = measurement_processer.get_avg_ray_number()

    if save_data:
        print("Saving results")
        cfg_result_filename = ResultExporter().save(filtered_results)
        # Exports every variable starting with cfg_ to a config YAML file.
        ConfigExporter().export(locals(), cfg_result_filename)

    if do_evaluation:
        perform_evaluation.from_data(
            simulation_data.x_true, filtered_results, do_plot, cfg_map_config_filename
        )


if __name__ == "__main__":
    main()
