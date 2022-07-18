"""
Grid map library in python
author: Atsushi Sakai
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.interpolate.fitpack2 import RectBivariateSpline
from matplotlib import cm


class GridMap:
    """
    GridMap class
    """

    def __init__(self, width, height, resolution, center_x, center_y, init_val=0.0):
        """__init__
        :param width: number of grid for width
        :param height: number of grid for heigt
        :param resolution: grid resolution [m]
        :param center_x: center x position  [m]
        :param center_y: center y position [m]
        :param init_val: initial value for all grid
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.center_x = center_x
        self.center_y = center_y

        # self.left_lower_x = self.center_x - self.width / 2.0 * self.resolution
        # self.left_lower_y = self.center_y - self.height / 2.0 * self.resolution
        self.left_lower_x = -self.center_x
        self.left_lower_y = -self.center_y

        self.ndata = self.width * self.height
        self.data = [init_val] * self.ndata

        self.distance_transform = None
        self.distance_transform_interp = None

    @classmethod
    def load_grid_map_from_array(cls, map, resolution, center_x, center_y):
        height = map.shape[0]
        width = map.shape[1]

        gm = cls(
            height=height,
            width=width,
            resolution=resolution,
            center_x=center_x,
            center_y=center_y,
        )

        for idx, val in np.ndenumerate(map):
            cls.set_value_from_xy_index(
                self=gm, x_ind=idx[1], y_ind=height - idx[0], val=val
            )

        gm.distance_transform = cls.calc_distance_transform(gm)
        gm.distance_transform_interp = RectBivariateSpline(
            np.arange(width) * resolution,
            np.arange(height) * resolution,
            gm.distance_transform * resolution,
        )

        cls.discretize_dt_derivative(gm)

        return gm

    def discretize_dt_derivative(self):
        n = 10  # how many times finer grid than default

        stepnum = self.width * n
        gridx, stepsize = np.linspace(
            0, self.width * self.resolution, stepnum, retstep=True
        )
        gridy = np.linspace(0, self.height * self.resolution, stepnum)

        self.dt_derivative_stepsize = stepsize
        self.distance_transform_dx = self.distance_transform_interp(gridy, gridx, 0, 1)
        self.distance_transform_dy = self.distance_transform_interp(gridy, gridx, 1, 0)

    def get_value_from_xy_index(self, x_ind, y_ind):
        """get_value_from_xy_index
        when the index is out of grid map area, return None
        :param x_ind: x index
        :param y_ind: y index
        """

        grid_ind = self.calc_grid_index_from_xy_index(x_ind, y_ind)

        if 0 <= grid_ind < self.ndata:
            return self.data[grid_ind]
        else:
            return None

    def get_xy_index_from_xy_pos(self, x_pos, y_pos):
        """get_xy_index_from_xy_pos
        returns the closest cell, if position is out of boundary
        :param x_pos: x position [m]
        :param y_pos: y position [m]
        """
        x_ind = self.calc_xy_index_from_position(x_pos, self.left_lower_x, self.width)
        y_ind = self.calc_xy_index_from_position(y_pos, self.left_lower_y, self.height)

        return x_ind, y_ind

    def set_value_from_xy_pos(self, x_pos, y_pos, val):
        """set_value_from_xy_pos
        return bool flag, which means setting value is succeeded or not
        :param x_pos: x position [m]
        :param y_pos: y position [m]
        :param val: grid value
        """

        x_ind, y_ind = self.get_xy_index_from_xy_pos(x_pos, y_pos)

        if (not x_ind) or (not y_ind):
            return False  # NG

        flag = self.set_value_from_xy_index(x_ind, y_ind, val)

        return flag

    def set_value_from_xy_index(self, x_ind, y_ind, val):
        """set_value_from_xy_index
        return bool flag, which means setting value is succeeded or not
        :param x_ind: x index
        :param y_ind: y index
        :param val: grid value
        """

        if (x_ind is None) or (y_ind is None):
            return False, False

        grid_ind = int(y_ind * self.width + x_ind)

        if 0 <= grid_ind < self.ndata:
            self.data[grid_ind] = val
            return True  # OK
        else:
            return False  # NG

    def calc_grid_index_from_xy_index(self, x_ind, y_ind):
        grid_ind = int(y_ind * self.width + x_ind)
        return grid_ind

    def calc_grid_central_xy_position_from_xy_index(self, x_ind, y_ind):
        x_pos = self.calc_grid_central_xy_position_from_index(x_ind, self.left_lower_x)
        y_pos = self.calc_grid_central_xy_position_from_index(y_ind, self.left_lower_y)

        return x_pos, y_pos

    def calc_grid_central_xy_position_from_index(self, index, lower_pos):
        return lower_pos + index * self.resolution + self.resolution / 2.0

    def calc_xy_index_from_position(self, pos, lower_pos, array_size):
        # returns the closest cell, if pos is out of boundary

        index = np.floor((pos - lower_pos) / self.resolution)
        if type(pos) is np.ndarray:
            # position is less than lower_pos
            index[index < 0] = 0
            index[index >= array_size] = array_size - 1
            return index.astype(int)
        else:
            index = 0 if index < 0 else index
            index = array_size - 1 if index >= array_size else index
            return int(index)

    def calc_distance_transform(self):
        grid_data = np.reshape(np.array(self.data), (self.height, self.width))
        edt = ndimage.distance_transform_edt(1 - grid_data)
        return edt

    def calc_distance_transform_xy_index(self, x_ind, y_ind):
        edt = self.distance_transform
        return edt[y_ind, x_ind]

    def calc_distance_transform_xy_pos(self, xy):
        edt_interp = self.distance_transform_interp
        x = xy[:, 0]
        y = xy[:, 1]
        # zero at the middle of the cell
        return edt_interp.ev(
            y - self.left_lower_y - self.resolution / 2.0,
            x - self.left_lower_x - self.resolution / 2.0,
        )

    def calc_distance_function_derivate_interp(self, xy_pos):

        """xy is in the coord system of the map
        (shifted, so the origin is around the middle of the map).
        However, the DT of the map has the origin in the bottom left corner.
        So the transformation of xy is needed:
        x: [-10,9.2) -> [0,19.2)
        y: [-10.05,9.15) -> [0,19.2).
        Also, the middle of the cell is considered, instead of the bottom left corner."""

        x_transf = xy_pos[:, 0] - self.left_lower_x - self.resolution / 2.0
        y_transf = xy_pos[:, 1] - self.left_lower_y - self.resolution / 2.0

        rounded = np.round(
            (np.array([x_transf, y_transf]).T / self.dt_derivative_stepsize)
        )
        rounded_int = rounded.astype(int)

        df_d_x = self.distance_transform_dx[rounded_int[:, 1], rounded_int[:, 0]]
        df_d_y = self.distance_transform_dy[rounded_int[:, 1], rounded_int[:, 0]]
        return df_d_x, df_d_y

    def calc_distance_function_derivate_interp_old(self, xy_pos):
        edt_interp = self.distance_transform_interp

        """xy is in the coord system of the map
        (shifted, so the origin is around the middle of the map).
        However, the DT of the map has the origin in the bottom left corner.
        So the transformation of xy is needed:
        x: [-10,9.2) -> [0,19.2)
        y: [-10.05,9.15) -> [0,19.2).
        Also, the middle of the cell is considered, instead of the bottom left corner."""

        x_transf = xy_pos[:, 0] - self.left_lower_x - self.resolution / 2.0
        y_transf = xy_pos[:, 1] - self.left_lower_y - self.resolution / 2.0

        df_d_x = edt_interp.ev(
            y_transf,
            x_transf,
            0,
            1,
        )
        df_d_y = edt_interp.ev(
            y_transf,
            x_transf,
            1,
            0,
        )
        return df_d_x, df_d_y

    def check_occupied_from_xy_index(self, xind, yind, occupied_val=1.0):

        val = self.get_value_from_xy_index(xind, yind)

        if val is None or val >= occupied_val:
            return True
        else:
            return False

    def expand_grid(self):
        xinds, yinds = [], []

        for ix in range(self.width):
            for iy in range(self.height):
                if self.check_occupied_from_xy_index(ix, iy):
                    xinds.append(ix)
                    yinds.append(iy)

        for (ix, iy) in zip(xinds, yinds):
            self.set_value_from_xy_index(ix + 1, iy, val=1.0)
            self.set_value_from_xy_index(ix, iy + 1, val=1.0)
            self.set_value_from_xy_index(ix + 1, iy + 1, val=1.0)
            self.set_value_from_xy_index(ix - 1, iy, val=1.0)
            self.set_value_from_xy_index(ix, iy - 1, val=1.0)
            self.set_value_from_xy_index(ix - 1, iy - 1, val=1.0)

    def print_grid_map_info(self):
        print("width:", self.width)
        print("height:", self.height)
        print("resolution:", self.resolution)
        print("center_x:", self.center_x)
        print("center_y:", self.center_y)
        print("left_lower_x:", self.left_lower_x)
        print("left_lower_y:", self.left_lower_y)
        print("ndata:", self.ndata)

    def plot_grid_map(self, ax=None, zorder=1):

        grid_data = np.reshape(np.array(self.data), (self.height, self.width))

        # plot tick labels in meters, so that (0,0) is at the origin of the map
        extent = [
            -self.center_x,
            -self.center_x + self.width * self.resolution,
            -self.center_y,
            -self.center_y + self.height * self.resolution,
        ]
        if not ax:
            fig, ax = plt.subplots()
        ax.imshow(
            grid_data,
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
            zorder=zorder,
            origin="lower",
            extent=extent,
        )
        plt.axis("equal")
        plt.grid()

    def plot_distance_transform(self, fig):

        edt = self.distance_transform.ravel()
        _x = np.arange(self.width)
        _y = np.arange(self.height)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()
        bottom = np.zeros_like(edt)
        width = depth = 1

        ax = fig.add_subplot(121, projection="3d")

        cmap = cm.get_cmap("jet")
        print(edt.min(), edt.max())
        rgba = [cmap((bar - edt.min()) / edt.max()) for bar in edt.ravel()]

        ax.bar3d(x, y, bottom, width, depth, edt.ravel(), color=rgba, shade=True)

        return ax

    def plot_distance_transform_interp(self, fig):

        _x = np.linspace(0, self.width, 101)
        _y = np.linspace(0, self.height, 101)
        _xx, _yy = np.meshgrid(_x, _y)

        z = [
            self.calc_distance_transform_xy_pos(x, y)[0][0]
            for x, y in zip(_xx.ravel(), _yy.ravel())
        ]
        ax = fig.add_subplot(122, projection="3d")
        ax.plot_surface(
            _xx, _yy, np.array(z).reshape(_xx.shape), cmap="jet", edgecolor="none"
        )
