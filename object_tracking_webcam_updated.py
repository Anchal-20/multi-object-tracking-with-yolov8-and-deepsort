import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def main():
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")  # you can change to yolov8s.pt for more accuracy

    # Initialize DeepSORT tracker
    tracker = DeepSort(max_age=30)

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 detection
        results = model(frame)[0]

        # Prepare list of detections: [ [left, top, width, height], confidence, class_id ]
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()   # x1, y1, x2, y2
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                left = float(x1)
                top = float(y1)
                width = float(x2 - x1)
                height = float(y2 - y1)
                conf = float(confs[i])
                cls_id = int(classes[i])

                # Append detection entry
                detections.append([[left, top, width, height], conf, cls_id])

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        # Build object counts per class
        object_counts = {}

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # left, top, right, bottom
            class_id = track.det_class
            class_name = model.model.names[class_id]

            # Count this track for the class
            object_counts[class_name] = object_counts.get(class_name, 0) + 1

            # Draw bounding box + ID
            x1, y1, x2, y2 = map(int, ltrb)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}-{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display object counts
        y_offset = 20
        for cls, count in object_counts.items():
            cv2.putText(frame, f"{cls}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

        # Show frame
        cv2.imshow("YOLOv8 + DeepSORT Tracking (All Classes)", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
