import numpy as np
import random


def get_circle(p1, p2, p3):
    r12 = p1[0] ** 2 + p1[1] ** 2
    r22 = p2[0] ** 2 + p2[1] ** 2
    r32 = p3[0] ** 2 + p3[1] ** 2
    A = p1[0] * (p2[1] - p3[1]) - p1[1] * (p2[0] - p3[0]) + p2[0] * p3[1] - p3[0] * p2[1]
    B = r12 * (p2[1] - p3[1]) + r22 * (p3[1] - p1[1]) + r32 * (p1[1] - p2[1])
    C = r12 * (p2[0] - p3[0]) + r22 * (p3[0] - p1[0]) + r32 * (p1[0] - p2[0])

    center_x = B / (2 * A)
    center_y = C / (2 * A)
    radius2 = (center_x - p1[0]) ** 2 + (center_y - p1[1]) ** 2

    return np.array([center_x, center_y]), radius2


def get_random_points(max_val, n):
    return random.sample(range(max_val), n)


def fit_circle_ransac(points, iter_num, ratio):
    n = len(points)
    target_n = int(n * ratio)
    best_inliers = []

    for _ in range(iter_num):
        # Get 3 random points
        sample_points = points[np.random.choice(points.shape[0], 3, replace=False)]
        x1, y1 = sample_points[0]
        x2, y2 = sample_points[1]
        x3, y3 = sample_points[2]

        # Fit a circle to these 3 points
        area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if area < 1e-6:
            continue

        A = np.array([[x1, y1, 1],
                      [x2, y2, 1],
                      [x3, y3, 1]])
        B = np.array([x1 ** 2 + y1 ** 2, x2 ** 2 + y2 ** 2, x3 ** 2 + y3 ** 2])

        try:
            params = np.linalg.inv(A).dot(B)
        except np.linalg.LinAlgError:
            continue

        cx = params[0] / 2
        cy = params[1] / 2
        r_squared = (params[2] + cx ** 2 + cy ** 2)

        seed_center = [cx, cy]
        seed_radius2 = r_squared

        maybe_inliers = []
        for pt in points:
            # Check if point is an inlier
            distance_squared = (pt[0] - seed_center[0]) ** 2 + (pt[1] - seed_center[1]) ** 2
            if distance_squared < seed_radius2:
                maybe_inliers.append(pt)

        if len(maybe_inliers) > target_n:
            # If enough inliers, fit the circle by least squares
            center, radius = fit_circle_by_least_squares(maybe_inliers)
            return center.astype(int), radius.astype(int)

        else:
            if len(maybe_inliers) > len(best_inliers):
                best_inliers = maybe_inliers
                best_inliers.extend(sample_points)

    center, radius = fit_circle_by_least_squares(best_inliers)
    return center.astype(int), radius.astype(int)


def fit_circle_by_least_squares(points):
    # Fit a circle using the least squares method
    points = np.array(points)
    A = np.vstack([2 * points[:, 0], 2 * points[:, 1], np.ones(len(points))]).T
    b = points[:, 0] ** 2 + points[:, 1] ** 2

    # Solve the linear system to find the circle parameters
    params = np.linalg.lstsq(A, b, rcond=None)[0]
    center = np.array([params[0], params[1]])
    radius = np.sqrt(center[0] ** 2 + center[1] ** 2 + params[2])

    return center.astype(int), radius.astype(int)
