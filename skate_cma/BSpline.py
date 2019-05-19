import numpy as np


class BSpline:
    def __init__(self):
        self.ctrlPoint = dict()

    def generate_control_point(self, points, points_offset=None):
        assert(len(points) >= 2)
        if points_offset is not None:
            assert(len(points) == len(points_offset))
        num_points = len(points)
        cubic_spline_to_B_spline = 4.*np.eye(num_points+2) + np.eye(num_points+2, k=1) + np.eye(num_points+2, k=-1)
        cubic_spline_to_B_spline[0, :3] = np.array([1., -2., 1.])
        cubic_spline_to_B_spline[-1, -3:] = np.array([1., -2., 1.])
        cubic_spline_to_B_spline *= 1./6.

        points_matrix = np.zeros((num_points+2, len(points[0])))
        for point_idx in range(num_points):
            if points_offset is not None:
                points_matrix[point_idx+1, :] = np.asarray(points[point_idx]) + np.asarray(points_offset[point_idx])
            else:
                points_matrix[point_idx+1, :] = np.asarray(points[point_idx])

        control_points_matrix = np.linalg.inv(cubic_spline_to_B_spline).dot(points_matrix)
        for point_idx in range(num_points+2):
            self.ctrlPoint[point_idx] = control_points_matrix[point_idx, :]

    def add_control_point(self, controlPoint):
        self.ctrlPoint[len(self.ctrlPoint.keys())] = controlPoint

    def set_control_point(self, idx, controlPoint):
        self.ctrlPoint[idx] = controlPoint

    def get_control_points(self):
        return self.ctrlPoint

    def get_control_point(self, idx):
        return self.ctrlPoint[idx]

    def get_value(self, tt):
        assert(len(self.ctrlPoint.keys()) >= 4)
        idx = int(tt)
        t = tt - idx

        if idx >= len(self.ctrlPoint.keys()) - 3:
            idx = len(self.ctrlPoint.keys()) - 4
            t = 1.

        sq_t = t * t
        cub_t = sq_t * t
        inv_t = 1. - t
        sq_inv_t = inv_t * inv_t
        cub_inv_t = sq_inv_t * inv_t

        return (
                cub_inv_t * self.ctrlPoint[idx]
                + (3.*cub_t - 6.*sq_t + 4.)*self.ctrlPoint[idx+1]
                + (-3.*cub_t + 3.*sq_t + 3.*t + 1.)*self.ctrlPoint[idx+2]
                + cub_t * self.ctrlPoint[idx+3]
               )/6.

    def clear(self):
        self.ctrlPoint.clear()
