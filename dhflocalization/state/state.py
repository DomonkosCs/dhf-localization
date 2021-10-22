

import numpy as np


class StateHypothesis:
    def __init__(self, pose: np.array = [], covar: np.array = [], timestamp=0):
        self.pose = pose
        self.covar = covar
        self.timestamp = timestamp
