from core.utils import is_inside_lane


def evaluate_risk(frame, box, label):

    height, width, _ = frame.shape
    x1, y1, x2, y2 = box

    box_height = y2 - y1
    center_x = (x1 + x2) / 2

    # -----------------------------
    # Lane Check
    # -----------------------------
    left_bound = width * 0.33
    right_bound = width * 0.66
    inside_lane = left_bound < center_x < right_bound

    # -----------------------------
    # Distance Approximation
    # -----------------------------
    size_ratio = box_height / height

    small_threshold = 0.15
    medium_threshold = 0.30

    # -----------------------------
    # Class Categorization
    # -----------------------------
    vehicle_classes = ["car", "truck", "bus"]
    vulnerable_classes = ["person", "bicycle", "motorcycle"]
    signal_classes = ["traffic light", "stop sign"]

    # -----------------------------
    # Risk Logic
    # -----------------------------

    # Traffic signals are informational
    if label in signal_classes:
        return "SAFE"

    # Vulnerable road users (pedestrians etc.)
    if label in vulnerable_classes:
        if inside_lane and size_ratio > small_threshold:
            return "DANGER"
        elif size_ratio > small_threshold:
            return "CAUTION"
        else:
            return "SAFE"

    # Vehicles
    if label in vehicle_classes:

        # OUTSIDE LANE
        if not inside_lane:
            if size_ratio > medium_threshold:
                return "CAUTION"
            else:
                return "SAFE"

        # INSIDE LANE
        else:
            if size_ratio > medium_threshold:
                return "DANGER"
            elif size_ratio > small_threshold:
                return "CAUTION"
            else:
                return "SAFE"

    # Default for other objects
    return "SAFE"