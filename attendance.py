"""
=========================================================
AI Face Attendance System
Part 1
Author : Malay Maity
=========================================================
"""

import cv2
import face_recognition
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from openpyxl import Workbook, load_workbook

# =========================================================
# CONFIGURATION
# =========================================================

ENCODING_FILE = "encodings.pkl"
ATTENDANCE_FOLDER = "Attendance"

CSV_FILE = os.path.join(
    ATTENDANCE_FOLDER,
    "Attendance.csv"
)

EXCEL_FILE = os.path.join(
    ATTENDANCE_FOLDER,
    "Attendance.xlsx"
)

TOLERANCE = 0.50

FONT = cv2.FONT_HERSHEY_SIMPLEX

# =========================================================
# CREATE ATTENDANCE DIRECTORY
# =========================================================

if not os.path.exists(ATTENDANCE_FOLDER):
    os.makedirs(ATTENDANCE_FOLDER)

# =========================================================
# LOAD ENCODINGS
# =========================================================

print("=" * 60)
print("Loading Face Encodings...")
print("=" * 60)

if not os.path.exists(ENCODING_FILE):
    print("Error : encodings.pkl not found")
    print("Run encode_faces.py first.")
    exit()

with open(ENCODING_FILE, "rb") as f:
    data = pickle.load(f)

known_face_encodings = data["encodings"]
known_face_names = data["names"]

print(f"Loaded {len(known_face_names)} Faces")

# =========================================================
# CREATE CSV IF NOT EXISTS
# =========================================================

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

# =========================================================
# CREATE EXCEL FILE
# =========================================================

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

    return datetime.now().strftime("%d-%m-%Y")


def current_time():

    return datetime.now().strftime("%H:%M:%S")


# =========================================================
# CONFIDENCE CALCULATION
# =========================================================

def face_confidence(distance, threshold=0.6):
    """
    Convert face distance into confidence %
    """

    if distance > threshold:
        confidence = (
            (1.0 - distance)
            /
            (1.0 - threshold)
        )

    else:
        confidence = (
            1.0 -
            (distance / threshold)
        )

    confidence = max(
        0,
        min(confidence, 1)
    )

    return round(confidence * 100, 2)


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

    if already_marked(name):

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


# =========================================================
# CAMERA INITIALIZATION
# =========================================================

print("=" * 60)
print("Starting Camera...")
print("=" * 60)

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():

    print("Cannot open webcam.")
    exit()

print("Camera Started Successfully")

print("\nPress 'Q' to Quit\n")