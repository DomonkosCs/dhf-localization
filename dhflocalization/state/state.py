

import numpy as np


class StateHypothesis:
    def __init__(self, pose: np.matrix = [], covar: np.matrix = [], timestamp=0):
        self.pose = pose
        self.covar = covar
        self.timestamp = timestamp
