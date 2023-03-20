from ..visualization import Plotter
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerPatch
import matplotlib.pyplot as plt

# import numpy as np


class TrackPlotter(Plotter):  # TODO remove inheritance
    def __init__(self, background_map=None) -> None:
        Plotter.__init__(self)
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel(r"$x\,(\mathrm{m})$")
        self.ax.set_ylabel(r"$y\,(\mathrm{m})$")
        self.ax.grid(which="major", linestyle="--", alpha=0.5)
        self.ax.grid(which="minor", linestyle=":", alpha=0.2)

        # Create empty lists for legend handles and labels
        self.handles_list = []
        self.labels_list = []
        self.background_map = background_map

        if self.background_map is not None:
            self.background_map.plot_grid_map(ax=self.ax)

    # Based on Stone Soup: https://github.com/dstl/Stone-Soup
    def _plot_track(
        self,
        track,
        mapping,
        uncertainty=False,
        particle=False,
        track_label="Track",
        **kwargs
    ):

        track_kwargs = dict(linestyle="-", marker=".", color=None)
        track_kwargs.update(kwargs)

        line = self.ax.plot(track[:, mapping[0]], track[:, mapping[1]], **track_kwargs)

        track_kwargs["color"] = plt.getp(line[0], "color")

        # Generate legend items for track
        track_handle = Line2D(
            [],
            [],
            linestyle=track_kwargs["linestyle"],
            marker=track_kwargs["marker"],
            color=track_kwargs["color"],
        )
        self.handles_list.append(track_handle)
        self.labels_list.append(track_label)

        if particle:
            # Plot particles
            for state in track:
                data = state.particles[:, mapping[:2]]
                self.ax.plot(
                    data[:, 0],
                    data[:, 1],
                    linestyle="",
                    marker=".",
                    markersize=1,
                    alpha=0.5,
                )

            # Generate legend items for particles
            particle_handle = Line2D(
                [], [], linestyle="", color="black", marker=".", markersize=1
            )
            particle_label = "Particles"
            self.handles_list.append(particle_handle)
            self.labels_list.append(particle_label)

            # Generate legend
            self.ax.legend(handles=self.handles_list, labels=self.labels_list)

        else:
            self.ax.legend(handles=self.handles_list, labels=self.labels_list)

    def plot_tracks(self, filtered_states, reference_states=None):
        for key in filtered_states:
            self._plot_track(
                filtered_states[key]["state"],
                [0, 1],
                marker=None,
                linestyle="--",
                track_label=key,
            )
        if reference_states is not None:
            for key in reference_states:
                self._plot_track(
                    reference_states[key],
                    [0, 1],
                    marker=None,
                    linestyle=":" if key == "odom" else "-",
                    track_label=key,
                )
        plt.show()


class _HandlerEllipse(HandlerPatch):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = Ellipse(xy=center, width=width + xdescent, height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]
