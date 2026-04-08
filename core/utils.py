import numpy as np
import cv2


def get_lane_polygon(frame):
    height, width, _ = frame.shape

    lane_polygon = np.array([
        [int(width * 0.25), height],
        [int(width * 0.45), height],
        [int(width * 0.5), int(height * 0.55)]
    ])

    return lane_polygon


def is_inside_lane(frame, box):
    lane_polygon = get_lane_polygon(frame)

    x1, y1, x2, y2 = box

    # Use bottom-center for realism
    center_x = int((x1 + x2) / 2)
    center_y = int(y2)

    result = cv2.pointPolygonTest(lane_polygon, (center_x, center_y), False)

    return result >= 0