import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import tkinter as tk
from tkinter import simpledialog

# ---------------- CONFIG ----------------
CAMERA_INDEX = 1
SIM_THRESHOLD = 0.55
FACE_DIR = "faces"
os.makedirs(FACE_DIR, exist_ok=True)

# ---------------- MODEL ----------------
app = FaceAnalysis(
    name="buffalo_l",       # VERY accurate
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

# ---------------- UI ----------------
def popup_input_name():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    name = simpledialog.askstring("Save Face", "Enter person name:")
    root.destroy()
    return name

# ---------------- HELPERS ----------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def is_front_face(face, max_ratio=0.35):
    """
    Rejects side faces using eye-nose geometry
    """
    left_eye, right_eye, nose = face.kps[0], face.kps[1], face.kps[2]

    eye_dist = abs(right_eye[0] - left_eye[0])
    nose_offset = abs(nose[0] - (left_eye[0] + right_eye[0]) / 2)

    ratio = nose_offset / eye_dist
    return ratio < max_ratio

# ---------------- LOAD KNOWN FACES ----------------
known_faces = []

def load_known_faces():
    for file in os.listdir(FACE_DIR):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            emb = np.load(os.path.join(FACE_DIR, file))
            known_faces.append({"name": name, "emb": emb})

load_known_faces()

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(CAMERA_INDEX)

print("Press 's' to save new face | 'q' to quit")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:

        x1, y1, x2, y2 = map(int, face.bbox)

        # ❌ SIDE FACE → UNKNOWN
        if not is_front_face(face):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                "Unknown",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            continue

        # ✔ FRONT FACE → RECOGNITION
        emb = face.embedding
        name = "Unknown"
        best_sim = 0

        for person in known_faces:
            sim = cosine_similarity(emb, person["emb"])
            if sim > best_sim and sim > SIM_THRESHOLD:
                best_sim = sim
                name = person["name"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            name,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    # ---------------- SAVE FACE ----------------
    if key == ord('s') and faces:
        face = faces[0]

        if not is_front_face(face):
            print("❌ Please face the camera directly")
            continue

        name = popup_input_name()
        if not name:
            print("❌ Name cancelled")
            continue

        np.save(os.path.join(FACE_DIR, f"{name}.npy"), face.embedding)
        known_faces.append({"name": name, "emb": face.embedding})

        print(f"✔ Face saved: {name}")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
