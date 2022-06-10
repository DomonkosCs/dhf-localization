import numpy as np
import copy


class MeasurementProcessor:
    def __init__(self, max_ray_number: int) -> None:
        self.max_ray_number = max_ray_number
        self.number_of_rays = []

    def filter_measurements(self, raw_measurement):
        return self.delete_rays_evenly(raw_measurement)

    def delete_rays_evenly(self, raw_measurement):
        raw_ray_number = len(raw_measurement)
        number_to_remove = raw_ray_number - self.max_ray_number

        if number_to_remove <= 0:
            self.number_of_rays.append(raw_ray_number)
            return raw_measurement

        idxs_to_remove = np.round(np.linspace(0, raw_ray_number - 1, number_to_remove))
        idxs_to_remove = idxs_to_remove.astype(int)

        measurement_copy = copy.copy(raw_measurement)
        for idx in idxs_to_remove[::-1]:
            measurement_copy.pop(idx)

        self.number_of_rays.append(len(measurement_copy))
        return measurement_copy

    def get_avg_ray_number(self):
        return np.mean(self.number_of_rays)
