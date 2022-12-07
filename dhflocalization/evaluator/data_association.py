from bisect import bisect_left


def optitrack_with_filter(optitrack_stamps, filter_stamps):
    closest_indices = list(
        map(lambda i: bisect_left(optitrack_stamps, i), filter_stamps)
    )

    return closest_indices
