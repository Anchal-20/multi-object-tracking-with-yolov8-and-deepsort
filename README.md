# 🧠 Multi-Object Tracking with YOLOv8 and DeepSORT

Real-time AI-powered multi-object detection, tracking, and counting system built using **YOLOv8**, **DeepSORT**, **OpenCV**, and **Streamlit**.

This project performs real-time object tracking on webcam/video streams while maintaining persistent object identities across frames using DeepSORT association metrics and Kalman filtering.

Designed for applications such as:
- Smart surveillance
- Traffic analytics
- Retail monitoring
- Autonomous systems
- Drone/object tracking

---

# 🎥 Demo Videos

## 🖌️Brush Tracking
https://github.com/user-attachments/assets/835198d2-4bcd-4263-b9ca-f7e4a1a16dc8

## 🧸 Teddy Bear Tracking
https://github.com/user-attachments/assets/3fc0925a-6ecc-4afc-a227-9b382393bd40

##  📱 Cell Phone Tracking
https://github.com/user-attachments/assets/56ccfdff-8677-41cf-9529-c6aa0c04c52c

---

# 🚀 Key Features

- 🔍 Real-time object detection using YOLOv8
- 🧠 Multi-object tracking with DeepSORT
- 🆔 Persistent object ID assignment across frames
- 📦 Bounding box visualization with tracking IDs
- 📊 Live Streamlit dashboard for analytics and monitoring
- ⚡ FPS and latency benchmarking
- 🎯 Real-time webcam/video stream support
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
Analytics Dashboard + Visualization
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
├── benchmark.py
├── dashboard.py
├── object_tracking_webcam_updated.py
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

# 🧠 Technical Highlights

- Uses YOLOv8 for high-speed real-time object detection
- DeepSORT performs identity association using:
  - Kalman Filtering
  - Hungarian Algorithm
  - Motion + appearance embeddings
- Supports continuous tracking even during partial occlusion
- Provides persistent tracking IDs across frames

# 📌 Applications

- 🚗 Traffic monitoring
- 🛡️ Smart surveillance systems
- 🛒 Retail analytics
- 🚁 Drone tracking
- 🤖 Autonomous robotics
- 🏭 Industrial monitoring

# 🔮 Future Improvements

- Multi-camera tracking support
- Person re-identification (ReID)
- TensorRT optimization for edge deployment
- ByteTrack / BoTSORT integration
- Custom-trained YOLO models

# 👨‍💻 Author
## Anchal Gajbhiye
GitHub: https://github.com/Anchal-20
