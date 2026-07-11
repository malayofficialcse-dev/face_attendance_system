"""
=========================================================
AI Face Attendance System
Part 1
Author : Malay Maity
=========================================================
"""

import cv2
import numpy as np
import pandas as pd
import pickle
import os
import time
from datetime import datetime
from openpyxl import Workbook, load_workbook

import config
from email_report import send_attendance_report

# =========================================================
# CONFIGURATION
# =========================================================

MODEL_FILE = config.MODEL_FILE
LABELS_FILE = config.LABELS_FILE
CSV_FILE = config.CSV_FILE
EXCEL_FILE = config.EXCEL_FILE
FACE_DETECTION_MODEL = config.FACE_DETECTION_MODEL
CONFIDENCE_THRESHOLD = config.LBPH_CONFIDENCE_THRESHOLD
PROCESS_SCALE = config.PROCESS_SCALE
FONT = cv2.FONT_HERSHEY_SIMPLEX
UNKNOWN_FOLDER = config.UNKNOWN_FOLDER
SAVE_UNKNOWN_FACE = config.SAVE_UNKNOWN_FACE
UNKNOWN_IMAGE_FORMAT = config.UNKNOWN_IMAGE_FORMAT
UNKNOWN_PREFIX = config.UNKNOWN_PREFIX
SHOW_CONFIDENCE = config.SHOW_CONFIDENCE
WINDOW_NAME = config.WINDOW_NAME
CAMERA_INDEX = config.CAMERA_INDEX
FRAME_WIDTH = config.FRAME_WIDTH
FRAME_HEIGHT = config.FRAME_HEIGHT
SKIP_FRAMES = config.SKIP_FRAMES
SHOW_FPS = config.SHOW_FPS
HAAR_CASCADE_PATH = config.HAAR_CASCADE_PATH

# =========================================================
# CREATE ATTENDANCE DIRECTORY
# =========================================================

if not os.path.exists(config.ATTENDANCE_FOLDER):
    os.makedirs(config.ATTENDANCE_FOLDER)

# =========================================================
# LOAD LBPH MODEL
# =========================================================

print("=" * 60)
print("Loading Face Recognition Model...")
print("=" * 60)

if not os.path.exists(MODEL_FILE) or not os.path.exists(LABELS_FILE):
    print("Error: LBPH model or labels not found.")
    print("Run encode_faces.py first.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_FILE)

with open(LABELS_FILE, "rb") as f:
    label_map = pickle.load(f)

reverse_labels = {value: key for key, value in label_map.items()}

print(f"Loaded {len(reverse_labels)} faces")

if not os.path.exists(CSV_FILE):

    df = pd.DataFrame(
        columns=[
            "Name",
            "Date",
            "Time"
        ]
    )

    df.to_csv(
        CSV_FILE,
        index=False
    )


if not os.path.exists(EXCEL_FILE):

    workbook = Workbook()

    sheet = workbook.active

    sheet.title = "Attendance"

    sheet.append([
        "Name",
        "Date",
        "Time"
    ])

    workbook.save(EXCEL_FILE)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def current_date():

    return datetime.now().strftime(config.DATE_FORMAT)


def current_time():

    return datetime.now().strftime(config.TIME_FORMAT)


# =========================================================
# CONFIDENCE CALCULATION
# =========================================================

def face_confidence(confidence):
    return round(confidence, 2)


# =========================================================
# CHECK IF ALREADY MARKED TODAY
# =========================================================

def already_marked(name):

    if not os.path.exists(CSV_FILE):
        return False

    df = pd.read_csv(CSV_FILE)

    today = current_date()

    records = df[
        (df["Name"] == name)
        &
        (df["Date"] == today)
    ]

    return len(records) > 0


# =========================================================
# MARK ATTENDANCE
# =========================================================

def mark_attendance(name):

    if already_marked(name) and not config.ALLOW_DUPLICATE_ATTENDANCE:

        print(f"{name} already marked today.")

        return

    date = current_date()
    time = current_time()

    # ---------------- CSV ----------------

    df = pd.read_csv(CSV_FILE)

    new_row = pd.DataFrame(
        [[name, date, time]],
        columns=[
            "Name",
            "Date",
            "Time"
        ]
    )

    df = pd.concat(
        [df, new_row],
        ignore_index=True
    )

    df.to_csv(
        CSV_FILE,
        index=False
    )

    # ---------------- Excel ----------------

    workbook = load_workbook(EXCEL_FILE)

    sheet = workbook.active

    sheet.append([
        name,
        date,
        time
    ])

    workbook.save(EXCEL_FILE)

    print(f"Attendance Marked -> {name}")


# =========================================================
# DRAW FACE BOX
# =========================================================

def draw_face_box(
    frame,
    top,
    right,
    bottom,
    left,
    name,
    confidence,
    color
):

    cv2.rectangle(
        frame,
        (left, top),
        (right, bottom),
        color,
        2
    )

    label = name

    if SHOW_CONFIDENCE:
        label = f"{name} ({confidence:.1f}%)"

    cv2.rectangle(
        frame,
        (left, bottom - 30),
        (right, bottom),
        color,
        cv2.FILLED
    )

    cv2.putText(
        frame,
        label,
        (left + 6, bottom - 8),
        FONT,
        0.55,
        (255, 255, 255),
        1
    )


def save_unknown_face(frame, top, right, bottom, left):

    if not SAVE_UNKNOWN_FACE:
        return

    if not os.path.exists(UNKNOWN_FOLDER):
        os.makedirs(UNKNOWN_FOLDER)

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    face_image = frame[top:bottom, left:right]

    filename = f"{UNKNOWN_PREFIX}{timestamp}{UNKNOWN_IMAGE_FORMAT}"
    filepath = os.path.join(UNKNOWN_FOLDER, filename)

    cv2.imwrite(filepath, face_image)


# =========================================================
# CAMERA INITIALIZATION
# =========================================================

print("=" * 60)
print("Starting Camera...")
print("=" * 60)

video_capture = cv2.VideoCapture(CAMERA_INDEX)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not video_capture.isOpened():

    print("Cannot open webcam.")
    exit()

print("Camera Started Successfully")
print("\nPress 'Q' to Quit\n")


# =========================================================
# START FACE RECOGNITION
# =========================================================

frame_counter = 0
last_time = time.time()

while True:

    # Read Frame
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Mirror Effect
    frame = cv2.flip(frame, 1)

    # Resize for faster processing
    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=PROCESS_SCALE,
        fy=PROCESS_SCALE
    )

    # Convert BGR -> RGB
    rgb_small_frame = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2RGB
    )

    should_process = SKIP_FRAMES <= 1 or frame_counter % SKIP_FRAMES == 0

    if should_process:

        gray_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        face_locations = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        face_names = []
        face_confidences = []

        for (x, y, w, h) in face_locations:
            face_image = gray_frame[y:y+h, x:x+w]
            face_image = cv2.resize(face_image, (200, 200))

            label, confidence = recognizer.predict(face_image)
            confidence = face_confidence(confidence)

            if confidence <= CONFIDENCE_THRESHOLD and label in reverse_labels:
                name = reverse_labels[label]
                mark_attendance(name)
                face_names.append(name)
                face_confidences.append(100 - confidence)
            else:
                face_names.append("Unknown")
                face_confidences.append(confidence)

    frame_counter += 1

    # =====================================================
    # DRAW RESULTS
    # =====================================================

    for (x, y, w, h), name, confidence in zip(
        face_locations,
        face_names,
        face_confidences
    ):

        # Scale back coordinates to original frame size
        x = int(x / PROCESS_SCALE)
        y = int(y / PROCESS_SCALE)
        w = int(w / PROCESS_SCALE)
        h = int(h / PROCESS_SCALE)

        top = y
        right = x + w
        bottom = y + h
        left = x

        if name == "Unknown":
            color = (0, 0, 255)
            save_unknown_face(frame, top, right, bottom, left)
        else:
            color = (0, 255, 0)

        draw_face_box(
            frame,
            top,
            right,
            bottom,
            left,
            name,
            confidence,
            color
        )

    # =====================================================
    # SHOW TOTAL DETECTED FACES
    # =====================================================

    cv2.putText(
        frame,
        f"Faces : {len(face_locations)}",
        (10, 30),
        FONT,
        0.8,
        (255, 255, 0),
        2
    )

    # =====================================================
    # SHOW ATTENDANCE COUNT
    # =====================================================

    try:

        attendance_df = pd.read_csv(CSV_FILE)

        today = current_date()

        attendance_count = len(
            attendance_df[
                attendance_df["Date"] == today
            ]
        )

        cv2.putText(
            frame,
            f"Today's Attendance : {attendance_count}",
            (10, 65),
            FONT,
            0.7,
            (0, 255, 255),
            2
        )

    except Exception:
        pass

    # =====================================================
    # DISPLAY DATE & TIME
    # =====================================================

    now = datetime.now()

    current_datetime = now.strftime(
        config.DATETIME_FORMAT
    )

    cv2.putText(
        frame,
        current_datetime,
        (10, frame.shape[0] - 15),
        FONT,
        0.6,
        (255, 255, 255),
        2
    )

    if SHOW_FPS:
        now_time = time.time()
        fps = 1.0 / max(now_time - last_time, 0.001)
        last_time = now_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (frame.shape[1] - 130, 30),
            FONT,
            0.6,
            (0, 255, 0),
            2
        )

    # =====================================================
    # WINDOW TITLE
    # =====================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )

    # =====================================================
    # EXIT
    # =====================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        print("\nClosing Camera...")

        break

# =========================================================
# CLEANUP
# =========================================================

video_capture.release()

cv2.destroyAllWindows()

print("=" * 60)
print("Attendance System Closed Successfully")
print("=" * 60)