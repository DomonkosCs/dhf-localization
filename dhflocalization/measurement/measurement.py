

import numpy as np


class Measurement():
    def __init__(self):
        pass

    def do_measurement(self, range_min, range_max, sample_num, inf_prob):

        r = (range_max-range_min)*np.random.rand(sample_num)+range_min
        for idx, range in enumerate(r):
            r[idx] = np.inf if np.random.rand() < inf_prob else range

        theta = np.linspace(0, 360, sample_num, endpoint=False)
        z = np.zeros([sample_num, 2])
        z[:, 0] = theta
        z[:, 1] = r

        return z
