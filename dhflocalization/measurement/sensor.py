

import numpy as np
from gridmap.grid_map import GridMap


class Sensor():
    def __init__(self, range_min, range_max, sample_num, inf_prob):
        self.range_min = range_min
        self.range_max = range_max
        self.sample_num = sample_num
        self.inf_prob = inf_prob


class Detection():
    def __init__(self, state, ogm: GridMap, sensor: Sensor, timestamp):
        self.timestamp = timestamp
        self.state = state
        self.ogm = ogm
        self.sensor = sensor
        self.z = np.empty((sensor.sample_num, 2))
        self.z[:, 0] = np.linspace(
            0, 2*np.pi, sensor.sample_num, endpoint=False)
        # don't divide by 0 if ray is vertical/horizontal
        self.z[:, 0] += np.finfo(float).eps
        self.z[:, 1] = [self.raycast(self.z[i, 0])
                        for i in range(sensor.sample_num)]

    def initRaycast(self, angle):
        ogm = self.ogm
        state = self.state
        # get current tile
        tileX, tileY = ogm.get_xy_index_from_xy_pos(
            state.state[0], state.state[1])
        # ray direction vector
        dirX = np.cos(angle)
        dirY = np.sin(angle)
        # based on direction, calculate where the next tile is
        dTileX = 1 if (dirX > 0) else -1
        # calculate the distance from the next tile in x in the units of dirX
        dX = ((tileX + (0.5*dTileX + 0.5) -
               ogm.get_xy_index_from_xy_pos(0, 0)[0]) * ogm.resolution - state.state[0]) / dirX

        dTileY = 1 if (dirY > 0) else -1
        dY = ((tileY + (0.5*dTileY + 0.5) -
               ogm.get_xy_index_from_xy_pos(0, 0)[1]) * ogm.resolution - state.state[1]) / dirY

        return tileX, tileY, dTileX, dTileY, dX, dY,  dTileX * ogm.resolution / dirX, dTileY * ogm.resolution / dirY

    def raycast(self, angle):
        state = self.state
        ogm = self.ogm
        sensor = self.sensor

        tileX, tileY, dTileX, dTileY, dX, dY, ddX, ddY = self.initRaycast(
            angle)

        t = 0
        if (ogm.get_value_from_xy_index(tileX, tileY) != 1):
            while ((tileX > 0) & (tileX <= ogm.width) & (tileY > 0) & (tileY <= ogm.height)):
                if (dX < dY):
                    tileX = tileX + dTileX
                    dt = dX
                    t = t + dt
                    dX = dX + ddX - dt
                    dY = dY - dt
                else:
                    tileY = tileY + dTileY
                    dt = dY
                    t = t + dt
                    dX = dX - dt
                    dY = dY + ddY - dt

                if (t > sensor.range_max):
                    t = np.inf
                    break

                if (ogm.get_value_from_xy_index(tileX, tileY)):
                    break
        else:
            print(
                'Robot cannot be located on an occupied cell, returning 0 as distance...')
        return t
