from tabnanny import verbose

import cv2
from ultralytics import YOLO

# ---------------------------------
# Load Model
# ---------------------------------
# def load_model(model_path="models/yolov8n.pt"):
#     return YOLO(model_path)


# ---------------------------------
# Process Frame
# ---------------------------------
def process_frame(model, frame, evaluate_risk, conf_threshold, selected_classes):

    # CONF_THRESHOLD = 0.5
    print("Current threshold:", conf_threshold)
    results = model(frame, verbose=False)[0]
    danger_count = 0

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = results.names[cls]

        if label not in selected_classes:
            continue

        if conf < conf_threshold:
            continue

        status = evaluate_risk(frame, (x1, y1, x2, y2), label)

        if status == "DANGER":
            color = (0, 0, 255)
            danger_count += 1
        elif status == "CAUTION":
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame,
                    f"{label} {conf:.2f} - {status}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

    if danger_count > 0:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
        cv2.putText(frame,
                    f"⚠ HIGH RISK: {danger_count} OBJECT(S)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    3)

    # 👇 SHOW THRESHOLD SETTING
    cv2.putText(frame,
                f"Conf Thresh: {conf_threshold:.2f}",
                (frame.shape[1] - 260, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2)

    return frame