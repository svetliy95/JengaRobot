import numpy as np


def point_projection_on_line(line_point1, line_point2, point):
    ap = point - line_point1
    ab = line_point2 - line_point1
    result = line_point1 + np.dot(ap, ab) / np.dot(ab, ab) * ab
    return result
