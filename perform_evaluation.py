from dhflocalization.rawdata import ResultLoader, ConfigImporter, RawDataLoader
from dhflocalization.gridmap import GridMap
from dhflocalization.visualization import TrackPlotter
from dhflocalization.evaluator import metrics
from dhflocalization.rawdata import YamlWriter


def from_data(true_states, filtered_results, do_plot=False, map_filename=None):
    (err_mean_sqare, err_mean_abs, std) = metrics.eval(true_states, filtered_results)

    print(err_mean_sqare)
    print("---")
    print(err_mean_abs)
    print("---")
    print(std)

    if not do_plot or not map_filename:
        return

    ogm = GridMap.load_map_from_config(map_filename)
    track_plotter = TrackPlotter(background_map=ogm)
    track_plotter.plot_results(true_states, filtered_results)


def from_file(results_filename, do_plot=False):
    filtered_results = ResultLoader.load(results_filename)
    meta_data = ConfigImporter.read(results_filename)
    simulation_data = RawDataLoader.load_from_json(meta_data["cfg_simu_data_filename"])

    true_states = simulation_data.x_true
    (err_mean_squares, err_mean_abss, err_stds) = metrics.eval(
        true_states, filtered_results
    )

    # update the original config file with the results
    metrics_dict = {
        "RMSE": err_mean_squares,
        "MAE": err_mean_abss,
        "STD": err_stds,
    }
    YamlWriter().updateFile(
        payload=metrics_dict,
        filename=results_filename,
    )

    print(err_mean_squares)
    print("---")
    print(err_mean_abss)
    print("---")
    print(err_stds)

    if not do_plot:
        return

    ogm = GridMap.load_map_from_config(meta_data["cfg_map_config_filename"])
    track_plotter = TrackPlotter(background_map=ogm)
    track_plotter.plot_results(simulation_data.x_true, filtered_results)


if __name__ == "__main__":
    # from ./resources/results/
    results_filename = "23-05-30T113500"
    from_file(results_filename, do_plot=True)
