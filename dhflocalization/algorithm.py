#!/usr/bin/env python

# %%
from gridmap.grid_map import GridMap
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
%matplotlib widget
# %%


def do_measurement():
    def produce_ranges(inf_prob):
        ranges = 100*np.random.rand(samples)-50
        for idx, range in enumerate(ranges):
            ranges[idx] = np.inf if np.random.rand() < inf_prob else range
        return ranges

    samples = 360
    thetas = np.linspace(0, 360, samples, endpoint=False)
    measurements = np.zeros([samples, 2])
    measurements[:, 0] = thetas
    measurements[:, 1] = produce_ranges(0.1)

    return measurements


def transform_to_global(measurements, robot_state):
    return np.array([robot_state[0] + measurements[:, 1]*np.cos(measurements[:, 0]+robot_state[2]),
                     robot_state[1] + measurements[:, 1] *
                     np.sin(measurements[:, 0]+robot_state[2])
                     ]).T
# %%


test = GridMap.load_grid_map_from_csv('map_test.csv', 1, 0, 0)
test.plot_grid_map()
fig = plt.figure()
test.plot_distance_transform(fig)
test.plot_distance_transform_interp(fig)
grid_data = np.reshape(np.array(test.data), (test.height, test.width))
edt = ndimage.distance_transform_edt(1 - grid_data)
edt_interp = RectBivariateSpline(
    np.arange(test.width), np.arange(test.height), edt)

ocg.
# %%
