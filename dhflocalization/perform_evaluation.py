from rawdata import resultLoader
from gridmap import GridMap
from gridmap import PgmProcesser
from visualization import TrackPlotter
import evaluator.metrics as metrics


def main(results_filename):
    results = resultLoader.load(results_filename)
    (err_mean_sqare, err_mean_abs, std) = metrics.eval(
        filtered_states=results[0],
        reference_states=results[1],
        export_filename=results_filename,
        return_results=True,
    )

    print(err_mean_sqare)
    print(err_mean_abs)
    print(std)
    # TODO load from yaml, or find another solution
    map_fn = "tb3_house_lessnoisy"

    ogm = GridMap.load_grid_map_from_array(
        PgmProcesser.read_pgm(map_fn), 0.05, 10, 10.05
    )
    track_plotter = TrackPlotter()
    track_plotter.background_map = ogm
    track_plotter.plot_tracks(results[0], results[1])


if __name__ == "__main__":
    results_filename = "22-02-21T120226"
    main(results_filename)
