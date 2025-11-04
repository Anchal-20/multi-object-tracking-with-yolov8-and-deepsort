# 🧠 Multi-Object Tracking with YOLOv8 and DeepSORT

Real-time AI system that detects, tracks, and counts multiple objects (vehicles, people, drones, etc.) in live video streams.  
Built using **YOLOv8** for object detection and **DeepSORT** for tracking, with an interactive **Streamlit dashboard** for analytics and performance benchmarking.

> The system performs real-time detection and tracking on live webcam/video feeds.  
> Each object is assigned a unique ID and color-coded bounding box, with live count and performance metrics displayed in the dashboard.

---

## ⚙️ Features
- 🔍 **Real-time detection** using YOLOv8  
- 🧠 **Multi-object tracking** with DeepSORT  
- 🧾 **Object counting** and ID persistence across frames  
- 📊 **Interactive Streamlit dashboard** for insights (heatmaps, track paths, counts)  
- ⚡ **Benchmarking tool** to compare FPS, latency, and accuracy  
- 🌗 Works seamlessly in **day/night** or **indoor/outdoor** conditions  

---

## 🧩 Tech Stack
- **Languages:** Python  
- **Frameworks:** PyTorch, OpenCV, Streamlit  
- **Models:** YOLOv8 / Detectron2, DeepSORT / ByteTrack  
- **Libraries:** NumPy, Pandas, Matplotlib  

---

## 💻 Installation

Clone the repository:
```bash
git clone https://github.com/<your-username>/multi-object-tracking-with-yolov8-and-deepsort.git
cd multi-object-tracking-with-yolov8-and-deepsort
