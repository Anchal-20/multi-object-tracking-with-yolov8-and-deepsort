# 🧠 Multi-Object Tracking with YOLOv8 and DeepSORT

Real-time AI-powered multi-object detection, tracking, and counting system built using **YOLOv8**, **DeepSORT**, **OpenCV**, and **Streamlit**.

This project performs real-time object tracking on webcam/video streams while maintaining persistent object identities across frames using DeepSORT data association and Kalman filtering.

The system supports:
- Real-time webcam inference
- Persistent object ID tracking
- Interactive dashboard analytics
- FPS & latency benchmarking
- Configurable detection thresholds
- Multi-class object monitoring

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

- 🔍 Real-time object detection using YOLOv8
- 🧠 Multi-object tracking with DeepSORT
- 🆔 Persistent object identity assignment across frames
- 📦 Bounding box visualization with tracking IDs
- 📊 Interactive Streamlit dashboard
- ⚡ FPS and latency benchmarking
- 🎯 Real-time webcam stream processing
- 🎛️ Dynamic confidence threshold adjustment
- 🧩 Selective object-class filtering
- 🌗 Robust performance across varying lighting conditions

---

# 🏗️ System Architecture

```text
Video/Webcam Input
        │
        ▼
YOLOv8 Object Detection
        │
        ▼
DeepSORT Tracker
(Kalman Filter + Hungarian Matching)
        │
        ▼
Object ID Assignment
        │
        ▼
Visualization + Analytics Dashboard
```

---

# 🧩 Tech Stack

## Languages
- Python

## Deep Learning & Computer Vision
- YOLOv8
- DeepSORT
- OpenCV
- PyTorch

## Visualization & Analytics
- Streamlit
- Matplotlib
- Pandas
- NumPy

---

# 📂 Project Structure

```text
multi-object-tracking-with-yolov8-and-deepsort/
│
├── README.md
├── requirements.txt
├── benchmark.py
├── dashboard.py
├── object_tracking_webcam_updated.py
├── benchmark_results.csv
├── yolov8n.pt
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Anchal-20/multi-object-tracking-with-yolov8-and-deepsort.git

cd multi-object-tracking-with-yolov8-and-deepsort
```

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Real-Time Tracking

```bash
python object_tracking_webcam_updated.py
```

---

# 📊 Launch Streamlit Dashboard

```bash
streamlit run dashboard.py
```

---

# 📈 Run Performance Benchmarking

```bash
python benchmark.py
```

---

# 📊 Benchmarking Features

The benchmarking module evaluates:
- FPS (Frames Per Second)
- Average inference latency
- Model performance across resolutions
- YOLOv8n vs YOLOv8s tradeoffs

Benchmark results are automatically exported to:

```text
benchmark_results.csv
```

---

# 🧠 Technical Highlights

- YOLOv8 performs real-time object detection
- DeepSORT maintains persistent object identities across frames
- Kalman filtering improves tracking stability during motion and partial occlusion
- Hungarian matching enables efficient object association
- Streamlit dashboard provides live analytics and monitoring controls
- Confidence thresholds and class filtering allow customizable inference

---

# 📌 Applications

- 🚗 Traffic monitoring
- 🛡️ Smart surveillance systems
- 🛒 Retail analytics
- 🚁 Drone/object tracking
- 🤖 Autonomous robotics
- 🏭 Industrial monitoring

---

# 🔮 Future Improvements

- Multi-camera tracking support
- Trajectory/path visualization
- Line-crossing analytics
- GPU acceleration support
- TensorRT optimization
- ByteTrack / BoTSORT integration
- Custom-trained YOLO models
- Video upload & RTSP stream support

---

# 👨‍💻 Author

## Anchal Gajbhiye
GitHub: https://github.com/Anchal-20
