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


# =========================================================
# START FACE RECOGNITION
# =========================================================

process_this_frame = True

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
        fx=0.25,
        fy=0.25
    )

    # Convert BGR -> RGB
    rgb_small_frame = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2RGB
    )

    if process_this_frame:

        # Detect Faces
        face_locations = face_recognition.face_locations(
            rgb_small_frame,
            model="hog"
        )

        # Generate Encodings
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame,
            face_locations
        )

        face_names = []
        face_confidences = []

        # =====================================================
        # LOOP THROUGH EACH DETECTED FACE
        # =====================================================

        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(
                known_face_encodings,
                face_encoding,
                tolerance=TOLERANCE
            )

            face_distances = face_recognition.face_distance(
                known_face_encodings,
                face_encoding
            )

            name = "Unknown"
            confidence = 0

            if len(face_distances) > 0:

                best_match_index = np.argmin(face_distances)

                confidence = face_confidence(
                    face_distances[best_match_index]
                )

                if matches[best_match_index]:

                    name = known_face_names[
                        best_match_index
                    ]

                    # Mark Attendance
                    mark_attendance(name)

            face_names.append(name)
            face_confidences.append(confidence)

    process_this_frame = not process_this_frame

    # =====================================================
    # DRAW RESULTS
    # =====================================================

    for (top,
         right,
         bottom,
         left), name, confidence in zip(
        face_locations,
        face_names,
        face_confidences
    ):

        # Scale Back Coordinates
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        if name == "Unknown":

            color = (0, 0, 255)

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

        cv2.putText(
            frame,
            f"Today's Attendance : {len(attendance_df)}",
            (10, 65),
            FONT,
            0.7,
            (0, 255, 255),
            2
        )

    except:
        pass

    # =====================================================
    # DISPLAY DATE & TIME
    # =====================================================

    now = datetime.now()

    current_datetime = now.strftime(
        "%d-%m-%Y %H:%M:%S"
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

    # =====================================================
    # WINDOW TITLE
    # =====================================================

    cv2.imshow(
        "AI Face Attendance System",
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