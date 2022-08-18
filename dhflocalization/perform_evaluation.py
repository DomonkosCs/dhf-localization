from .rawdata import resultLoader, ConfigImporter
from .gridmap import GridMap
from .gridmap import PgmProcesser
from .visualization import TrackPlotter
from .evaluator import metrics as metrics

def main(results_filename):
    # load results from the pickle file
    # TODO fix name
    results = resultLoader.load(results_filename)
    config = ConfigImporter.importData(results_filename)
    (err_mean_sqare, err_mean_abs, std) = metrics.eval(
        filtered_states=results[0],
        reference_states=results[1],
        export_filename=results_filename,
        return_results=True,
    )

    print(err_mean_sqare)
    print(err_mean_abs)
    print(std)

    map_fn = config["cfg_map_filename"]

    ogm = GridMap.load_grid_map_from_array(
        PgmProcesser.read_pgm(map_fn), 0.05, 10, 10.05
    )
    track_plotter = TrackPlotter()
    track_plotter.background_map = ogm
    track_plotter.plot_tracks(results[0], results[1])


if __name__ == "__main__":
    results_filename = "22-07-06T142456"
    main(results_filename)
