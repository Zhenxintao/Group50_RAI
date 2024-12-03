from geometry_msgs.msg import Quaternion

import numpy as np
from numpy.typing import NDArray

def rotmat2q(T: NDArray) -> Quaternion:
    # Function that transforms a 3x3 rotation matrix to a ros quaternion representation
    
    if T.shape != (3, 3):
        raise ValueError

    # # TODO: implement this
    # raise NotImplementedError

    # return q
        # Calculate the trace of the matrix
    trace = np.trace(T)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # S = 4w
        w = 0.25 * s
        x = (T[2, 1] - T[1, 2]) / s
        y = (T[0, 2] - T[2, 0]) / s
        z = (T[1, 0] - T[0, 1]) / s
    else:
        # We are in the case where the trace is negative
        if (T[0, 0] > T[1, 1]) and (T[0, 0] > T[2, 2]):
            s = np.sqrt(1.0 + T[0, 0] - T[1, 1] - T[2, 2]) * 2  # S = 4x
            w = (T[2, 1] - T[1, 2]) / s
            x = 0.25 * s
            y = (T[0, 1] + T[1, 0]) / s
            z = (T[0, 2] + T[2, 0]) / s
        elif T[1, 1] > T[2, 2]:
            s = np.sqrt(1.0 + T[1, 1] - T[0, 0] - T[2, 2]) * 2  # S = 4y
            w = (T[0, 2] - T[2, 0]) / s
            x = (T[0, 1] + T[1, 0]) / s
            y = 0.25 * s
            z = (T[1, 2] + T[2, 1]) / s
        else:
            s = np.sqrt(1.0 + T[2, 2] - T[0, 0] - T[1, 1]) * 2  # S = 4z
            w = (T[1, 0] - T[0, 1]) / s
            x = (T[0, 2] + T[2, 0]) / s
            y = (T[1, 2] + T[2, 1]) / s
            z = 0.25 * s

    # Construct the quaternion
    q = Quaternion(x=x, y=y, z=z, w=w)
    return q