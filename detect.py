from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

def detect_objects(image_path):
    results = model(image_path)[0]
    annotated_frame = results.plot()

    # Get classes and bounding boxes
    classes = results.names
    boxes = results.boxes
    class_ids = [int(cls) for cls in boxes.cls]
    labels = [classes[cls] for cls in class_ids]

    # Object counts
    obj_count = {}
    for label in labels:
        obj_count[label] = obj_count.get(label, 0) + 1

    # Crop objects
    crops = []
    for i, box in enumerate(boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        crop = annotated_frame[y1:y2, x1:x2]
        label = classes[class_ids[i]]
        crops.append((label, crop))

    return annotated_frame, obj_count, crops


def detect_video(input_path, output_path="output.mp4"):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        annotated = results.plot()
        out.write(annotated)

    cap.release()
    out.release()
    return output_path

