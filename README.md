# 👤 Face Recognition System (Python)

A real-time **Face Recognition system** built with Python that can detect faces from a webcam, recognize known people, and label unknown faces.  
This project uses **InsightFace (ArcFace)** for high accuracy and supports both **CPU and GPU** execution.

---

## 🚀 Features

- ✅ Real-time face detection using webcam
- ✅ Accurate face recognition using ArcFace embeddings
- ✅ Save new faces with **popup name input**
- ✅ Recognize multiple faces simultaneously
- ✅ Label faces as **Known / Unknown**
- ✅ Easy to extend and customize
- ✅ Works on Windows, Linux, and macOS

---

## 🧠 Concepts Used

This project follows the **standard face recognition pipeline**:

1. **Face Detection** – Locate faces in the frame  
2. **Face Alignment** – Normalize face orientation  
3. **Face Embedding** – Convert face into numeric vector  
4. **Face Matching** – Compare embeddings using cosine similarity  

> 🔑 Face recognition is **not possible without face detection**.

---

## 📂 Project Structure

Face_recognition/

── main.py # Main application
── faces/ # Stored face embeddings (.npy)
── requirements.txt # Python dependencies
── README.md # Project documentation
── .gitignore
---

## ⚙️ Requirements

- Python **3.7+**
- Webcam
- Supported OS: Windows / Linux / macOS

### 📦 Python Libraries Used

- `opencv-python`
- `insightface`
- `onnxruntime` / `onnxruntime-gpu`
- `numpy`
- `tkinter` (built-in)

---

## 🔧 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Shaikhhamza29/Face_recognition.git
cd Face_recognition



