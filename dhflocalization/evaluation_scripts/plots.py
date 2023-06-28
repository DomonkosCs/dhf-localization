import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from dhflocalization.gridmap import GridMap
from dhflocalization.rawdata import RawDataLoader


# distance transform plot
def dt_plot():
    map_filename = "gt_map_01_table"
    ogm = GridMap.load_map_from_config(map_filename)

    fig, ax = plt.subplots()
    fig.set_figwidth(7)
    fig.set_figheight(5)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])

    data = np.sqrt(np.flipud(ogm.distance_transform) / 12)
    # roi = [590:1350, 350:2100]

    plt.imshow(data, cmap="gray", vmin=1, vmax=6)
    plt.savefig("dt_map_whole.png")
    plt.show()


def gt_plot():
    plt.rcParams["text.usetex"] = True
    plt.rcParams.update({"font.size": 15, "font.family": "serif"})
    map_filename = "gt_map_05_table"
    simu_data_filename = "5hz_o1e-4_l1e-2_filtered"
    ogm = GridMap.load_map_from_config(map_filename)
    simulation_data = RawDataLoader.load_from_json(simu_data_filename)

    fig, ax = plt.subplots()
    fig.set_figwidth(6)
    # fig.set_figheight(5)
    ax.set_xlabel(r"$x\,(\mathrm{m})$")
    ax.set_ylabel(r"$y\,(\mathrm{m})$")
    ax.grid(which="major", linestyle="--", alpha=0.5)
    ax.grid(which="minor", linestyle=":", alpha=0.2)
    ## plot gridmap
    grid_data = np.reshape(np.array(ogm.data), (ogm.height, ogm.width))

    # plot tick labels in meters, so that (0,0) is at the origin of the map
    extent = [
        ogm.left_lower_x,
        ogm.left_lower_x + ogm.width * ogm.resolution,
        ogm.left_lower_y,
        ogm.left_lower_y + ogm.height * ogm.resolution,
    ]
    ax.imshow(
        grid_data,
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        extent=extent,
    )

    ## plot starting position
    ax.scatter(-3, 1, marker="*", color="tab:purple", s=80, zorder=2)

    ## plot ground truth track
    ax.plot(
        simulation_data.x_true[:, 0],
        simulation_data.x_true[:, 1],
        zorder=1,
        linewidth="2",
    )

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(20))
    ax.yaxis.set_minor_locator(AutoMinorLocator(20))
    # hide ticks
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlim((-8, 8))
    ax.set_ylim((-6, 6))

    plt.savefig("gt_trajectory.eps")
    plt.show()
    # track_plotter = TrackPlotter(background_map=ogm)
    # track_plotter.plot_results(true_states, [])


gt_plot()
