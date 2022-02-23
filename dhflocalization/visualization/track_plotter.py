from gridmap import GridMap
from visualization import Plotter
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerPatch
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np


class TrackPlotter(Plotter):  # TODO remove inheritance
    def __init__(self) -> None:
        Plotter.__init__(self)
        self.fig = plt.figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel(r"$x\,(\mathrm{cell})$")
        self.ax.set_ylabel(r"$y\,(\mathrm{cell})$")
        self.ax.axis("equal")

        # Create empty lists for legend handles and labels
        self.handles_list = []
        self.labels_list = []
        self.background_map: Optional[GridMap] = None

    # Based on Stone Soup: https://github.com/dstl/Stone-Soup
    def _plot_ground_truths(
        self, truths, mapping, truths_label="Ground Truth", **kwargs
    ):
        """Plots ground truth(s)

        Plots each ground truth path passed in to :attr:`truths` and generates a legend
        automatically. Ground truths are plotted as dashed lines with default colors.

        Users can change linestyle, color and marker using keyword arguments. Any changes
        will apply to all ground truths.

        Parameters
        ----------
        truths : set of :class:`~.GroundTruthPath`
            Set of  ground truths which will be plotted. If not a set, and instead a single
            :class:`~.GroundTruthPath` type, the argument is modified to be a set to allow for
            iteration.
        mapping: list
            List of 2 items specifying the mapping of the x and y components of the state space.
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function. Default is ``linestyle="--"``.
        """

        truths_kwargs = dict(linestyle="--")
        truths_kwargs.update(kwargs)
        if not isinstance(truths, set):
            truths = {tuple(truths)}  # Make a set of length 1

        if self.background_map is not None:
            self.background_map.plot_grid_map(ax=self.ax)
            for truth in truths:
                x_coords = [
                    (state.pose[mapping[0], 0] - self.background_map.left_lower_x)
                    / self.background_map.resolution
                    for state in truth
                ]
                y_coords = [
                    (state.pose[mapping[1], 0] - self.background_map.left_lower_y)
                    / self.background_map.resolution
                    for state in truth
                ]
                self.ax.plot(x_coords, y_coords, **truths_kwargs, zorder=2)
        else:
            for truth in truths:
                x_coords = [state.pose[mapping[0], 0] for state in truth]
                y_coords = [state.pose[mapping[1], 0] for state in truth]
                self.ax.plot(x_coords, y_coords, **truths_kwargs, zorder=2)

        # Generate legend items
        truths_handle = Line2D(
            [], [], linestyle=truths_kwargs["linestyle"], color="black"
        )
        self.handles_list.append(truths_handle)
        self.labels_list.append(truths_label)

        # Generate legend
        self.ax.legend(handles=self.handles_list, labels=self.labels_list)

    # Based on Stone Soup: https://github.com/dstl/Stone-Soup
    def _plot_filtered_tracks(
        self,
        tracks,
        mapping,
        uncertainty=False,
        particle=False,
        track_label="Track",
        **kwargs
    ):
        """Plots track(s)

        Plots each track generated, generating a legend automatically. If ``uncertainty=True``,
        uncertainty ellipses are plotted. If ``particle=True``, particles are plotted.
        Tracks are plotted as solid lines with point markers and default colors.
        Uncertainty ellipses are plotted with a default color which is the same for all tracks.

        Users can change linestyle, color and marker using keyword arguments. Uncertainty ellipses
        will also be plotted with the user defined colour and any changes will apply to all tracks.

        Parameters
        ----------
        tracks : set of :class:`~.Track`
            Set of tracks which will be plotted. If not a set, and instead a single
            :class:`~.Track` type, the argument is modified to be a set to allow for iteration.
        mapping: list
            List of 2 items specifying the mapping of the x and y components of the state space.
        uncertainty : bool
            If True, function plots uncertainty ellipses.
        particle : bool
            If True, function plots particles.
        track_label: str
            Label to apply to all tracks for legend.
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function. Defaults are ``linestyle="-"``,
            ``marker='.'`` and ``color=None``.
        """

        tracks_kwargs = dict(linestyle="-", marker=".", color=None)
        tracks_kwargs.update(kwargs)

        if not isinstance(tracks, set):
            tracks = {tuple(tracks)}  # Make a set of length 1

        # Plot tracks
        track_colors = {}
        for track in tracks:

            # TODO refactor transformation to a function
            line = self.ax.plot(
                [
                    (state.pose[mapping[0], 0] - self.background_map.left_lower_x)
                    / self.background_map.resolution
                    for state in track
                ],
                [
                    (state.pose[mapping[1], 0] - self.background_map.left_lower_y)
                    / self.background_map.resolution
                    for state in track
                ],
                **tracks_kwargs
            )
            track_colors[track] = plt.getp(line[0], "color")

        # Assuming a single track or all plotted as the same colour then the following will work.
        # Otherwise will just render the final track colour.
        tracks_kwargs["color"] = plt.getp(line[0], "color")

        # Generate legend items for track
        track_handle = Line2D(
            [],
            [],
            linestyle=tracks_kwargs["linestyle"],
            marker=tracks_kwargs["marker"],
            color=tracks_kwargs["color"],
        )
        self.handles_list.append(track_handle)
        self.labels_list.append(track_label)

        if uncertainty:
            # Plot uncertainty ellipses
            for track in tracks:
                # Get position mapping matrix
                HH = np.eye(len(track[0].pose))[mapping, :]
                for state in track:
                    w, v = np.linalg.eig(HH @ state.covar @ HH.T)
                    max_ind = np.argmax(w)
                    min_ind = np.argmin(w)
                    orient = np.arctan2(v[1, max_ind], v[0, max_ind])
                    coords = (
                        (state.pose[mapping[0], 0] - self.background_map.left_lower_x)
                        / self.background_map.resolution,
                        (state.pose[mapping[1], 0] - self.background_map.left_lower_y)
                        / self.background_map.resolution,
                    )
                    ellipse = Ellipse(
                        xy=coords,
                        width=2 * np.sqrt(w[max_ind]) / self.background_map.resolution,
                        height=2 * np.sqrt(w[min_ind]) / self.background_map.resolution,
                        angle=np.rad2deg(orient),
                        alpha=0.2,
                        color=track_colors[track],
                    )
                    self.ax.add_artist(ellipse)

            # Generate legend items for uncertainty ellipses
            ellipse_handle = Ellipse(
                (0.5, 0.5), 0.5, 0.5, alpha=0.2, color=tracks_kwargs["color"]
            )
            ellipse_label = "Uncertainty"

            self.handles_list.append(ellipse_handle)
            self.labels_list.append(ellipse_label)

            # Generate legend
            self.ax.legend(
                handles=self.handles_list,
                labels=self.labels_list,
                handler_map={Ellipse: _HandlerEllipse()},
            )

        elif particle:
            # Plot particles
            for track in tracks:
                for state in track:
                    data = state.particles[:, mapping[:2]]
                    self.ax.plot(
                        (data[:, 0] - self.background_map.left_lower_x)
                        / self.background_map.resolution,
                        (data[:, 1] - self.background_map.left_lower_y)
                        / self.background_map.resolution,
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

    def plot_tracks(self, filtered_states, reference_states):
        for key in filtered_states:
            self._plot_filtered_tracks(
                filtered_states[key],
                [0, 1],
                marker=None,
                linestyle="--",
                track_label=key,
            )
        for key in reference_states:
            self._plot_ground_truths(
                reference_states[key],
                [0, 1],
                truths_label=key,
                linestyle=":" if key == "odom" else "-",
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
