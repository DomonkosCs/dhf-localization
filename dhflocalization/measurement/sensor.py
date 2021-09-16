

import numpy as np


class Sensor():
    def __init__(self, range_min, range_max, sample_num, inf_prob):
        self.range_min = range_min
        self.range_max = range_max
        self.sample_num = sample_num
        self.inf_prob = inf_prob


class Detection():
    def __init__(self, sensor, timestamp):
        self.sensor: Sensor = sensor
        self.timestamp: int = timestamp
        self.random_detection()

    def random_detection(self):

        r = (self.sensor.range_max-self.sensor.range_min) * \
            np.random.rand(self.sensor.sample_num)+self.sensor.range_min
        for idx, range in enumerate(r):
            r[idx] = np.inf if np.random.rand() < self.sensor.inf_prob else range

        theta = np.linspace(0, 2*np.pi, self.sensor.sample_num, endpoint=False)
        z = np.zeros([self.sensor.sample_num, 2])
        z[:, 0] = theta
        z[:, 1] = r

        self.z = z
