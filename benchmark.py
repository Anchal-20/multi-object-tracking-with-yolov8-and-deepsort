# benchmark.py
import cv2
import time
import csv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def run_benchmark(model_path, resolution=(640, 480), num_frames=200):
    """Runs detection + tracking for a given configuration and measures performance."""
    # Load detection model
    model = YOLO(model_path)
    # Initialize tracker
    tracker = DeepSort(max_age=30)
    # Video capture (webcam)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    frame_count = 0
    start_time = time.time()

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Detection
        results = model(frame)[0]
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                width = float(x2 - x1)
                height = float(y2 - y1)
                left = float(x1)
                top  = float(y1)
                conf = float(confs[i])
                cls_id = int(classes[i])
                detections.append([[left, top, width, height], conf, cls_id])

        # Tracking
        tracks = tracker.update_tracks(detections, frame=frame)

        frame_count += 1

    total_time = time.time() - start_time
    cap.release()

    fps = frame_count / total_time if total_time > 0 else 0
    avg_latency = total_time / frame_count if frame_count > 0 else 0

    return {
        "model": model_path,
        "resolution": f"{resolution[0]}x{resolution[1]}",
        "num_frames": frame_count,
        "total_time_sec": total_time,
        "fps": fps,
        "avg_latency_sec": avg_latency
    }

def main():
    configs = [
        {"model": "yolov8n.pt", "resolution": (640, 480)},
        {"model": "yolov8s.pt", "resolution": (640, 480)},
        {"model": "yolov8n.pt", "resolution": (1280, 720)}
    ]

    results = []
    for cfg in configs:
        print(f"Running config: {cfg}")
        res = run_benchmark(cfg["model"], cfg["resolution"], num_frames=200)
        print(f"Result: {res}")
        results.append(res)

    # Save results to CSV
    csv_file = "benchmark_results.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Saved results to {csv_file}")

if __name__ == "__main__":
    main()
