# 🧠 Real-Time Object Analytics System using YOLOv8 and DeepSORT

🚀 AI-powered real-time object detection, tracking, counting, and analytics platform built using **YOLOv8**, **DeepSORT**, **OpenCV**, **PyTorch**, and **Streamlit**.

The system performs real-time multi-object detection and tracking on webcam streams while maintaining persistent object identities across frames using DeepSORT's tracking-by-detection framework.

Beyond object tracking, the project includes an interactive analytics dashboard, performance benchmarking module, object counting, configurable inference controls, and real-time monitoring capabilities.

---

# 🎯 Project Overview

Traditional object detection systems identify objects independently in each frame but fail to maintain object identities over time.

This project addresses that challenge by combining:

- YOLOv8 for object detection
- DeepSORT for object tracking
- OpenCV for video processing
- Streamlit for analytics visualization

The system assigns unique IDs to detected objects and tracks them across video frames while providing real-time analytics and performance monitoring.

---

# 🎥 Demo Videos

## 🖌️ Brush Tracking

https://github.com/user-attachments/assets/835198d2-4bcd-4263-b9ca-f7e4a1a16dc8

## 🧸 Teddy Bear Tracking

https://github.com/user-attachments/assets/3fc0925a-6ecc-4afc-a227-9b382393bd40

## 📱 Cell Phone Tracking

https://github.com/user-attachments/assets/56ccfdff-8677-41cf-9529-c6aa0c04c52c

---

# 🚀 Key Features

✅ Real-Time Object Detection

✅ Multi-Object Tracking

✅ Persistent Object Identity Assignment

✅ YOLOv8 Integration

✅ DeepSORT Integration

✅ Object Counting

✅ Real-Time Webcam Processing

✅ Interactive Streamlit Dashboard

✅ Dynamic Confidence Threshold Adjustment

✅ Multi-Class Filtering

✅ FPS Monitoring

✅ Latency Benchmarking

✅ Performance Evaluation

---

# 🏗️ System Architecture

```text
Video/Webcam Input
        │
        ▼
YOLOv8 Object Detection
        │
        ▼
Bounding Box Generation
        │
        ▼
DeepSORT Tracker
(Kalman Filter + Hungarian Matching)
        │
        ▼
Object Association
        │
        ▼
Persistent Object IDs
        │
        ▼
Object Counting
        │
        ▼
Analytics Dashboard
        │
        ▼
Visualization & Monitoring
```

---

# ⚙️ How It Works

## Step 1 — Video Acquisition

Frames are captured from a live webcam stream using OpenCV.

---

## Step 2 — Object Detection

YOLOv8 detects objects in each frame and returns:

- Bounding Boxes
- Confidence Scores
- Class Labels

Supported object classes include:

- Person
- Car
- Bus
- Truck
- Bicycle
- Motorcycle
- Cell Phone
- Teddy Bear
- And other COCO dataset classes

---

## Step 3 — Feature Extraction

DeepSORT extracts appearance features from each detected object.

These features help distinguish similar-looking objects across frames.

---

## Step 4 — Object Association

DeepSORT combines:

- Motion Prediction (Kalman Filter)
- Appearance Embeddings
- Hungarian Matching Algorithm

to associate current detections with existing tracks.

---

## Step 5 — Identity Assignment

Each object receives a persistent tracking ID.

Example:

```text
Person-3
Cell Phone-7
Car-12
```

The ID remains consistent while the object stays visible.

---

## Step 6 — Analytics Generation

The system computes:

- Object Counts
- FPS
- Detection Statistics
- Tracking Information

in real time.

---

## Step 7 — Visualization

The final frame displays:

- Bounding Boxes
- Class Labels
- Tracking IDs
- Object Counts

for easy monitoring and analysis.

---

# 🧠 Why DeepSORT?

Object detection alone cannot track objects over time.

DeepSORT enables:

- Persistent object identities
- Robust tracking during temporary occlusions
- Motion prediction using Kalman Filters
- Efficient object association using Hungarian Matching
- Multi-object identity management

This allows the system to maintain object identities throughout a video sequence.

---

# 📊 Analytics Dashboard

The project includes a Streamlit-based dashboard for real-time monitoring.

### Dashboard Features

- Live webcam feed
- Real-time object counts
- FPS monitoring
- Dynamic confidence threshold control
- Object class filtering
- Interactive analytics interface

---

# 📈 Benchmarking Module

The benchmarking framework evaluates:

- Frames Per Second (FPS)
- Average Inference Latency
- Model Performance
- Resolution Scalability

Supported configurations:

- YOLOv8n @ 640×480
- YOLOv8s @ 640×480
- YOLOv8n @ 1280×720

Benchmark results are automatically exported to:

```text
benchmark_results.csv
```

---

# 📊 Example Benchmark Metrics

The benchmarking framework records:

| Metric | Description |
|----------|-------------|
| FPS | Frames processed per second |
| Latency | Average processing time per frame |
| Resolution | Input frame resolution |
| Model | YOLO model variant |
| Total Frames | Number of benchmarked frames |

---

# 🧩 Tech Stack

## Programming Language

- Python

## Computer Vision

- OpenCV

## Deep Learning

- PyTorch
- YOLOv8

## Object Tracking

- DeepSORT
- Kalman Filter
- Hungarian Matching

## Analytics & Visualization

- Streamlit
- Pandas
- NumPy
- Matplotlib

---

# 💡 Skills Demonstrated

- Computer Vision
- Object Detection
- Multi-Object Tracking
- Deep Learning
- Real-Time Video Analytics
- Tracking-by-Detection Systems
- Performance Benchmarking
- Dashboard Development
- Data Visualization
- AI System Deployment

---

# 📂 Project Structure

```text
multi-object-tracking-with-yolov8-and-deepsort/
│
├── README.md
├── benchmark.py
├── dashboard.py
├── object_tracking_webcam_updated.py
├── benchmark_results.csv
├── yolov8n.pt
├── yolov8s.pt
│
└── demo/
    ├── brush_tracking.mp4
    ├── teddy_bear_tracking.mp4
    └── cell_phone_tracking.mp4
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Anchal-20/multi-object-tracking-with-yolov8-and-deepsort.git

cd multi-object-tracking-with-yolov8-and-deepsort
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Real-Time Tracking

```bash
python object_tracking_webcam_updated.py
```

---

# 📊 Launch Dashboard

```bash
streamlit run dashboard.py
```

---

# 📈 Run Benchmarking

```bash
python benchmark.py
```

---

# 📌 Applications

🚗 Traffic Monitoring

🛡️ Smart Surveillance Systems

🛒 Retail Analytics

🚁 Drone-Based Tracking

🤖 Autonomous Robotics

🏭 Industrial Monitoring

🏙️ Smart City Infrastructure

👥 Crowd Monitoring

---

# 🔮 Future Improvements

- Multi-Camera Tracking
- Trajectory Visualization
- Line-Crossing Analytics
- Entry/Exit Counting
- Speed Estimation
- Crowd Density Analytics
- Heatmap Generation
- ByteTrack Integration
- BoTSORT Integration
- TensorRT Optimization
- RTSP Stream Support
- Video Upload Support
- Custom-Trained YOLO Models

---

# 🗺️ Roadmap

### Phase 1 — Completed

- YOLOv8 Detection
- DeepSORT Tracking
- Object Counting
- Dashboard Development
- Performance Benchmarking

### Phase 2 — In Progress

- Trajectory Visualization
- Entry/Exit Analytics
- Improved Tracking Stability

### Phase 3 — Advanced Analytics

- Heatmap Generation
- Crowd Analytics
- Traffic Analytics
- Speed Estimation

### Phase 4 — Production Deployment

- Cloud Deployment
- API Development
- Multi-Camera Infrastructure

---

# 📈 Results

The system successfully performs real-time object detection and tracking while maintaining persistent object identities across frames.

By combining YOLOv8's detection capabilities with DeepSORT's robust tracking framework, the project achieves accurate object association, real-time monitoring, and interactive analytics suitable for modern computer vision applications.



# 🚧 Status

Actively Under Development

Upcoming updates will focus on transforming the system into a complete real-time object analytics platform with advanced tracking intelligence, visualization, and reporting capabilities.
## Anchal Gajbhiye
GitHub: https://github.com/Anchal-20
