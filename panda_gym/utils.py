from typing import Union

import numpy as np


def distance(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
    """Compute the distance between two array. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        Union[float, np.ndarray]: The distance between the arrays.
    """
    assert a.shape == b.shape
    return np.linalg.norm(a - b, axis=-1)


def angle_distance(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
    """Compute the geodesic distance between two array of angles. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        Union[float, np.ndarray]: The geodesic distance between the angles.
    """
    assert a.shape == b.shape
    dist = 1 - np.inner(a, b) ** 2
    return dist

def ikfast(quat, pos):
    rotm = quat2rotm(quat)
    T = np.eye(4)
    T[:3, :3] = rotm
    T[:3, 3] = pos
    manip = robot.SetActiveManipulator(manip_name)
    ik_solutions = self.manip.FindIKSolutions(T, IK_CHECK_COLLISION)
    ik_sols = self.get_ik_solutions(quat, pos)
    ik_rank, ik_scores = self.rank_ik_sols(ik_sols)
    ik_sols = ik_sols.flatten()
    return [ik_sols, ik_rank]

def quat2rotm(quat):
    """
    Quaternion to rotation matrix.
    
    @type  quat: numpy array
    @param quat: quaternion (w, x, y, z)
    """
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    s = w * w + x * x + y * y + z * z
    return np.array([[1 - 2 * (y * y + z * z) / s, 2 * (x * y - z * w) / s, 2 * (x * z + y * w) / s], [2 * (x * y + z * w) / s, 1 - 2 * (x * x + z * z) / s, 2 * (y * z - x * w) / s], [2 * (x * z - y * w) / s, 2 * (y * z + x * w) / s, 1 - 2 * (x * x + y * y) / s]])