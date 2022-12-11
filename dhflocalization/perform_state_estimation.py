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

    cfg_map_filename = "sztaki_gmapping"
    cfg_map_resolution = 0.05  # m/cell

    cfg_simu_data_filename = "take01"

    do_plotting = True

    simulation_data = RawDataLoader.load_from_json(cfg_simu_data_filename)
    ogm = GridMap.load_grid_map_from_array(
        PgmProcesser.read_pgm(cfg_map_filename),
        resolution=cfg_map_resolution,
        center_x=10,
        center_y=10.05,
    )

    cfg_max_ray_number = 180
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

    cfg_measurement_range_noise_std = 0.1
    measurement_model = MeasurementModel(ogm, cfg_measurement_range_noise_std)

    cfg_edh_particle_number = 100
    cfg_ledh_particle_number = 1
    cfg_cledh_particle_number = 10

    cfg_edh_lambda_number = 10
    cfg_naedh_step_number = 5
    cfg_cledh_cluster_number = 1

    cfg_init_gaussian_mean = np.array([0.05, 0.075, 0])
    cfg_init_gaussian_covar = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.0025]])

    measurement_processer = MeasurementProcessor(max_ray_number=cfg_max_ray_number)
    ekf = EKF(measurement_model)
    edh = EDH(
        measurement_model=measurement_model,
        particle_num=cfg_edh_particle_number,
        lambda_num=cfg_edh_lambda_number,
    )
    aedh = EDH(
        measurement_model=measurement_model,
        particle_num=cfg_edh_particle_number,
        lambda_num=cfg_edh_lambda_number,
    )
    naedh = EDH(
        measurement_model=measurement_model,
        particle_num=cfg_edh_particle_number,
        step_num=cfg_naedh_step_number,
    )
    ledh = EDH(
        measurement_model=measurement_model,
        particle_num=cfg_ledh_particle_number,
        lambda_num=cfg_edh_lambda_number,
    )
    cledh = EDH(
        measurement_model=measurement_model,
        particle_num=cfg_cledh_particle_number,
        lambda_num=cfg_edh_lambda_number,
        cluster_num=cfg_cledh_cluster_number,
    )

    prior = ParticleState.init_from_gaussian(
        cfg_init_gaussian_mean,
        cfg_init_gaussian_covar,
        cfg_edh_particle_number,
        rng=rng,
    )
    ledh_prior = ParticleState.init_from_gaussian(
        cfg_init_gaussian_mean,
        cfg_init_gaussian_covar,
        cfg_ledh_particle_number,
        rng=rng,
    )
    cledh_prior = ParticleState.init_from_gaussian(
        cfg_init_gaussian_mean,
        cfg_init_gaussian_covar,
        cfg_cledh_particle_number,
        rng=rng,
    )

    edh_prior = aedh_prior = naedh_prior = prior

    ekf_prior = StateHypothesis(
        state_vector=cfg_init_gaussian_mean, covar=cfg_init_gaussian_covar
    )

    ekf_track = []
    edh_track = []
    aedh_track = []
    naedh_track = []
    ledh_track = []
    cledh_track = []
    ekf_track.append(ekf_prior.state_vector)
    edh_track.append(edh_prior.mean())
    aedh_track.append(aedh_prior.mean())
    naedh_track.append(naedh_prior.mean())
    ledh_track.append(ledh_prior.mean())
    cledh_track.append(cledh_prior.mean())

    ekf_comptimes = []
    edh_comptimes = []
    aedh_comptimes = []
    naedh_comptimes = []
    ledh_comptimes = []
    cledh_comptimes = []

    for i in range(1, simulation_data.simulation_steps, 1):
        print("{}/{}".format(i, simulation_data.simulation_steps))
        control_input = [simulation_data.x_odom[i - 1], simulation_data.x_odom[i]]
        measurement = measurement_processer.filter_measurements(
            simulation_data.measurement[i]
        )

        # propagate particles and perform ekf prediction
        ekf_prediction, ekf_prop_comptime = motion_model.propagate(
            ekf_prior, control_input
        )
        edh_prediction, edh_prop_comptime = motion_model.propagate_particles(
            edh_prior, control_input
        )
        aedh_prediction, aedh_prop_comptime = motion_model.propagate_particles(
            aedh_prior, control_input
        )
        naedh_prediction, naedh_prop_comptime = motion_model.propagate_particles(
            naedh_prior, control_input
        )
        ledh_prediction, ledh_prop_comptime = motion_model.propagate_particles(
            ledh_prior, control_input
        )
        cledh_prediction, cledh_prop_comptime = motion_model.propagate_particles(
            cledh_prior, control_input
        )

        ekf_posterior, ekf_update_comptime = ekf.update(ekf_prediction, measurement)
        edh_posterior, edh_update_comptime = edh.update_mean_flow(
            edh_prediction, ekf_prediction, measurement
        )
        aedh_posterior, aedh_update_comptime = aedh.update_analytic(
            aedh_prediction, ekf_prediction, measurement
        )
        naedh_posterior, naedh_update_comptime = naedh.update_analytic_multistep(
            naedh_prediction, ekf_prediction, measurement
        )
        ledh_posterior, ledh_update_comptime = ledh.update_local_flow(
            ledh_prediction, ekf_prediction, measurement
        )
        cledh_posterior, cledh_update_comptime = cledh.update_clustered_flow(
            cledh_prediction, ekf_prediction, measurement
        )

        ekf_track.append(ekf_posterior.state_vector)
        edh_track.append(edh_posterior.mean())
        aedh_track.append(aedh_posterior.mean())
        naedh_track.append(naedh_posterior.mean())
        ledh_track.append(ledh_posterior.mean())
        cledh_track.append(cledh_posterior.mean())

        ekf_comptimes.append(ekf_prop_comptime + ekf_update_comptime)
        edh_comptimes.append(edh_prop_comptime + edh_update_comptime)
        aedh_comptimes.append(aedh_prop_comptime + aedh_update_comptime)
        naedh_comptimes.append(naedh_prop_comptime + naedh_update_comptime)
        ledh_comptimes.append(ledh_prop_comptime + ledh_update_comptime)
        cledh_comptimes.append(cledh_prop_comptime + cledh_update_comptime)

        ekf_prior = ekf_posterior
        edh_prior = edh_posterior
        aedh_prior = aedh_posterior
        naedh_prior = naedh_posterior
        ledh_prior = ledh_posterior
        cledh_prior = cledh_posterior

    filtered_states = {
        "ekf": {
            "state": np.asarray(ekf_track),
            "comptime": np.array(ekf_comptimes).mean(),
        },
        "edh": {
            "state": np.asarray(edh_track),
            "comptime": np.array(edh_comptimes).mean(),
        },
        "aedh": {
            "state": np.asarray(aedh_track),
            "comptime": np.array(aedh_comptimes).mean(),
        },
        "naedh": {
            "state": np.asarray(naedh_track),
            "comptime": np.array(naedh_comptimes).mean(),
        },
        "ledh": {
            "state": np.asarray(ledh_track),
            "comptime": np.array(ledh_comptimes).mean(),
        },
        "cledh": {
            "state": np.asarray(cledh_track),
            "comptime": np.array(cledh_comptimes).mean(),
        },
    }
    print(filtered_states)

    # reference_states = {
    #     "odom": simulation_data.x_odom,
    #     # "odom": simulation_data.x_odom + np.array([-3, 1, 0]),
    #     # "true": x_true,
    # }

    # TODO move to results
    # cfg_avg_ray_number = measurement_processer.get_avg_ray_number()
    # print("Calcuations completed, saving results...")
    # cfg_result_filename = resultExporter().save(filtered_states, reference_states)
    # config_exporter.export(locals(), cfg_result_filename)

    # if do_plotting:
    #     evaluate.main(cfg_result_filename)


if __name__ == "__main__":
    main()
