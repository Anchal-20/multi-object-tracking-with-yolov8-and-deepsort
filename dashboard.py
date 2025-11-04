# dashboard.py
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

st.set_page_config(page_title="Object Tracking Dashboard", layout="wide")

# Sidebar controls
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox("Choose YOLO model", ["yolov8n.pt", "yolov8s.pt"], key="model_choice")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, key="conf_threshold")
show_classes = st.sidebar.multiselect("Classes to show (leave blank = all)", list(range(0,80)), key="show_classes")

# Load model and tracker
model = YOLO(model_choice)
tracker = DeepSort(max_age=30)

# Placeholders
video_placeholder = st.empty()
count_placeholder = st.empty()

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

st.write("### Press 'Stop' in sidebar to end")

# Add Stop button with unique key
stop_button = st.sidebar.button("Stop", key="stop_button")

while True:
    if stop_button:
        break

    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame from webcam.")
        break

    start_time = time.time()
    results = model(frame)[0]

    detections = []
    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        for i in range(len(boxes)):
            if confs[i] < conf_threshold:
                continue
            cls_id = int(classes[i])
            if show_classes and (cls_id not in show_classes):
                continue
            x1, y1, x2, y2 = boxes[i].tolist()
            left, top = float(x1), float(y1)
            width, height = float(x2 - x1), float(y2 - y1)
            conf = float(confs[i])
            detections.append([[left, top, width, height], conf, cls_id])

    tracks = tracker.update_tracks(detections, frame=frame)

    object_counts = {}
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        cls_id = track.det_class
        class_name = model.model.names[cls_id]
        object_counts[class_name] = object_counts.get(class_name, 0) + 1

        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name}-{track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    fps = 1.0 / (time.time() - start_time)

    # Display results
    video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    count_text = "\n".join([f"{cls}: {count}" for cls, count in object_counts.items()])
    count_placeholder.text(f"FPS: {fps:.2f}\n{count_text}")

cap.release()
cv2.destroyAllWindows()
