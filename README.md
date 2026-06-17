# Vision to Decision: Object Detection System for Self-Driving Cars 

An AI-powered object detection system that simulates the perception module of a self-driving vehicle. This project uses **YOLOv8** for real-time object detection and provides driving decisions based on detected road objects such as vehicles, pedestrians, traffic signals, and stop signs.

## 📌 Project Overview

Self-driving cars rely heavily on computer vision to understand their surroundings and make safe driving decisions. This project demonstrates a simplified version of that process by:

- Detecting objects in videos using YOLOv8.
- Identifying important road entities such as cars, trucks, buses, pedestrians, bicycles, traffic lights, and stop signs.
- Generating driving recommendations based on detected objects.
- Providing a foundation for autonomous vehicle perception systems.

## 🎯 Features

- Real-time object detection using YOLOv8
- Detection of multiple road objects
- Decision-making logic based on detected objects
- Video processing support
- Annotated output with bounding boxes and labels
- Easy-to-use and customizable

## 🛠️ Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy


## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Mamta06102004/Object-Detection-System-for-Self-Driving.git
cd Object-Detection-System-for-Self-Driving
```

### 2. Install Dependencies

```bash
pip install ultralytics opencv-python numpy
```

### 3. Run the Project

Open the notebook:

```bash
jupyter notebook Vision_to_Decision.ipynb
```

Or run it directly in Google Colab.

## 🚘 How It Works

### Step 1: Input Video

The system takes a video frame as input.

### Step 2: Object Detection

For each detected object, the model provides:

- Bounding Box Coordinates
- Class Label
- Confidence Score

### Step 3: Lane Analysis

The image is divided into lane regions to determine whether an object is:

- In the vehicle's driving lane
- In a neighboring lane
- Outside the driving path

Objects present in the driving lane are considered more critical than those outside it.

### Step 4: Distance Estimation

The system estimates the relative distance of each detected object using the size and position of its bounding box.

Objects are categorized as:

- Near
- Medium Distance
- Far

### Step 5: Risk Assessment

The system combines:

- Object Type
- Lane Position
- Estimated Distance

to evaluate the level of driving risk.

### Step 6: Decision Generation

Based on the risk assessment, the system classifies the situation into one of three categories:

| Status | Description |
|----------|-------------|
| 🟢 Safe | No immediate obstacle in the driving lane or all detected objects are at a safe distance. |
| 🟡 Caution | An object is present nearby or may require driver attention. |
| 🔴 Danger | A close object is detected directly in the driving lane, indicating a potential collision risk. |

### Step 7: Output Visualization

Detected objects are displayed with labeled bounding boxes and confidence scores.

## 📸 Sample Output
<img width="1103" height="408" alt="Screenshot 2026-04-09 220853" src="https://github.com/user-attachments/assets/7cb76778-7a93-40a0-ab34-ec0a26a60efb" />

## 💡 Applications

- Autonomous Vehicle Research
- Intelligent Transportation Systems
- Driver Assistance Systems (ADAS)
- Computer Vision Learning Projects
- AI-based Traffic Monitoring

## 🔮 Future Improvements

- Vehicle tracking across frames
- Real-time webcam support
- Advanced decision-making algorithms
  

### Project Goal

The goal of this project is to demonstrate how computer vision can bridge the gap between **seeing the environment (Vision)** and **taking intelligent actions (Decision)**, which is one of the fundamental concepts behind autonomous driving systems.
