<div align="center">
  <h1>🛡️ NEXUS AI Security System</h1>
  <p><b>Next-Generation Smart Surveillance & Threat Detection Engine</b></p>

  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
  [![YOLOv8](https://img.shields.io/badge/YOLOv8-Computer_Vision-yellow.svg)](https://github.com/ultralytics/ultralytics)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-YAMNet-orange.svg)](https://www.tensorflow.org/)
  [![Gemini](https://img.shields.io/badge/Google_Gemini-GenAI-purple.svg)](https://deepmind.google/technologies/gemini/)
</div>

<br/>

Welcome to **NEXUS AI Security**, a production-grade Python surveillance system built to demonstrate state-of-the-art Computer Vision, Audio Analysis, and Generative AI integrations. Designed for real-world scenarios, it processes multi-camera feeds in real-time to detect threats ranging from unattended bags to violent behavior.

This project is perfectly suited to demonstrate end-to-end system design, multi-threading, machine learning integration, and modern web application development.

---

## ✨ Core Features

* 👁️ **Suspicious Bag Detection**: Uses YOLOv8 & DeepSORT tracking to monitor bags. If a bag is separated from its owner for too long, a threat alert is generated.
* ⚔️ **Weapon Detection**: Identifies firearms and knives in real-time using highly-optimized weights, triggering instant high-priority alerts.
* 🏃‍♂️ **Behavior & Posture Analysis**: Utilizes YOLOv8-Pose to map human skeletons. Detects falling, violent physical altercations (fights), panic/running, and loitering.
* 🎙️ **Audio Threat Recognition**: Integrates Google’s YAMNet model to classify environmental audio (screams, gunshots, breaking glass, explosions).
* 🤖 **AI Incident Reports**: Integrates with Google Gemini Vision AI to automatically draft detailed, police-style incident reports with attached photographic evidence.
* 📱 **Mobile Push Notifications**: Real-time Telegram Bot integration sends instant alerts with photos straight to your phone.
* 🔥 **Smart Motion Heatmaps**: Generates real-time heatmaps overlaying human traffic patterns.
* 🌐 **Modern Glassmorphism Web Dashboard**: A stunning, responsive FastAPI + Vanilla JS web interface with live MJPEG streaming, system health metrics, and dynamic settings.

---

## 🏗️ System Architecture

NEXUS is built with a highly modular, decoupled architecture:
1. **Camera Workers**: Independent threads for every camera feed (Webcam, RTSP, Video File) utilizing OpenCV.
2. **Detector Pipeline**: Each frame passes through a dynamic, configurable chain of AI models (Bag, Weapon, Behavior).
3. **Threat Engine**: A centralized manager that scores incidents. Threats escalate dynamically from `LOW` to `CRITICAL`.
4. **Broadcast & Logging**: Threat events are logged via JSONL, snapshots are saved, and WebSockets/Web endpoints are updated in real-time.

---

## 🚀 Quick Start Guide

### 1. Environment Setup
You must run this on **Python 3.10+**. We strongly recommend using a virtual environment.

```powershell
# Enter the repository directory
cd security_ai_system

# Create and activate a virtual environment
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Launching the System
You can start the system using the stunning Web Dashboard (Recommended) or the local Desktop PyQt5 Dashboard.

**Start Web Dashboard (Recommended):**
```powershell
python main.py --web --source 0
```
*Navigate to `http://localhost:8000` in your browser. Default login: `admin` / `securepassword` (changeable in `.env`).*

**Start Desktop Dashboard:**
```powershell
python main.py --dashboard --source 0
```
*(Note: You can pass multiple `--source` arguments or video file paths to simulate a multi-camera CCTV control room!)*

---

## ⚙️ Integrations & Setup

* **Google Gemini AI**: Get a free API key from [Google AI Studio](https://aistudio.google.com/). Paste it into the Dashboard Settings to unlock AI Incident Reports.
* **Telegram Alerts**: Talk to `@BotFather` on Telegram to create a bot, and `@userinfobot` to get your Chat ID. Enter these in Settings to enable mobile push notifications.
* **Custom YOLO Weights**: The system ships with standard COCO weights. For true weapon detection, place custom `.pt` weights in the `/models` directory.

---

## 🧪 Testing

The system is fully unit-tested. The test suite avoids heavy ML initializations by using lightweight mock objects, meaning it runs instantly.

```powershell
# Run the complete test suite
python -m unittest discover -s tests
```

---

<div align="center">
  <i>Engineered for the future of automated security.</i>
</div>
