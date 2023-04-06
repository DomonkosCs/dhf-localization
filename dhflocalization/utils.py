import numpy as np


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def calc_angle_diff(a, b):
    a = normalize_angle(a)
    b = normalize_angle(b)
    d1 = a - b
    d2 = 2 * np.pi - abs(d1)

    if isinstance(d1, np.ndarray):
        return_arr = d2
        d2[d1 > 0] *= -1.0
        return_arr[abs(d1) < abs(d2)] = d1[abs(d1) < abs(d2)]
        return return_arr

    if d1 > 0:
        d2 *= -1.0
    if abs(d1) < abs(d2):
        return d1
    else:
        return d2


def angle_set_diff(set_1, set_2):
    return [calc_angle_diff(a, b) for a, b in zip(set_1, set_2)]
