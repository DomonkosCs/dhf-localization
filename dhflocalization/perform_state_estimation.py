#!/usr/bin/env python
# %%
import numpy as np

from dhflocalization.filters import EDH
from dhflocalization.filters.updaters import (
    LEDHUpdater,
    MEDHUpdater,
    AEDHUpdater,
    NAEDHUpdater,
    CLEDHUpdater,
)
from dhflocalization.kinematics import OdometryMotionModel
from dhflocalization.measurement import MeasurementModel, MeasurementProcessor
from dhflocalization.rawdata import RawDataLoader, ConfigExporter
from dhflocalization.customtypes import StateHypothesis, Track
from dhflocalization.rawdata import ResultExporter
from dhflocalization.filters import EKF
from dhflocalization.gridmap import GridMap
from dhflocalization import perform_evaluation as evaluate


def main():
    print("Starting state estimation...")

    # Exports every variable starting with cfg_ to a config YAML file.
    config_exporter = ConfigExporter()

    cfg_random_seed = 4302948723190478  # 2021
    rng = np.random.default_rng(cfg_random_seed)

    cfg_map_config_filename = "gt_map_05"

    cfg_simu_data_filename = "house_true_cut"

    do_plotting = True

    simulation_data = RawDataLoader.load_from_json(cfg_simu_data_filename)

    ogm = GridMap(cfg_map_config_filename)

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

    cfg_measurement_range_noise_std = 0.1
    robot_sensor_dx = -0.032
    measurement_model = MeasurementModel(
        ogm, cfg_measurement_range_noise_std, robot_sensor_dx
    )

    cfg_medh_particle_number = 1
    cfg_aedh_particle_number = 1
    cfg_ledh_particle_number = 50
    cfg_cledh_particle_number = 100

    cfg_medh_lambda_number = 10
    cfg_naedh_step_number = 10
    cfg_cledh_cluster_number = 5

    cfg_init_gaussian_mean = np.array([-3.0, 1.0, 0.0003])
    cfg_init_gaussian_covar = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.025]])

    particle_init_variables = [cfg_init_gaussian_mean, cfg_init_gaussian_covar, rng]

    measurement_processer = MeasurementProcessor(max_ray_number=cfg_max_ray_number)

    # init ekf
    ekf = EKF(measurement_model)
    ekf_prior = StateHypothesis(
        state_vector=cfg_init_gaussian_mean, covar=cfg_init_gaussian_covar
    )
    ekf_track = Track(ekf_prior)
    ekf_comptimes = []

    # init edh variants
    ledh_updater = LEDHUpdater(
        measurement_model, cfg_medh_lambda_number, cfg_ledh_particle_number
    )
    ledh = EDH(ledh_updater, *particle_init_variables)
    medh_updater = MEDHUpdater(
        measurement_model, cfg_medh_lambda_number, cfg_medh_particle_number
    )
    medh = EDH(medh_updater, *particle_init_variables)
    cledh_updater = CLEDHUpdater(
        measurement_model,
        cfg_medh_lambda_number,
        cfg_cledh_particle_number,
        cfg_cledh_cluster_number,
    )
    cledh = EDH(cledh_updater, *particle_init_variables)
    aedh_updater = AEDHUpdater(measurement_model, cfg_aedh_particle_number)
    aedh = EDH(aedh_updater, *particle_init_variables)
    naedh_updater = NAEDHUpdater(
        measurement_model, cfg_naedh_step_number, cfg_aedh_particle_number
    )
    naedh = EDH(naedh_updater, *particle_init_variables)

    edh_variants = []

    for i in range(1, simulation_data.simulation_steps, 1):
        # print("{}/{}".format(i, simulation_data.simulation_steps))
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

    filtered_states = {
        "ekf": {
            "track": ekf_track,
            "comptime": np.array(ekf_comptimes).mean(),
        },
    }

    for filter in edh_variants:
        filtered_states.update(filter.get_results())

    # TODO move to results
    cfg_avg_ray_number = measurement_processer.get_avg_ray_number()

    print("Calculations completed")

    cfg_result_filename = ResultExporter().save(filtered_states)
    config_exporter.export(locals(), cfg_result_filename)

    if do_plotting:
        evaluate.main(cfg_result_filename)


if __name__ == "__main__":
    main()