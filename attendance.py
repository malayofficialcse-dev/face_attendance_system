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
FONT_SCALE = config.FONT_SCALE
FONT_THICKNESS = config.FONT_THICKNESS
BOX_THICKNESS = config.BOX_THICKNESS
KNOWN_FACE_COLOR = config.KNOWN_FACE_COLOR
UNKNOWN_FACE_COLOR = config.UNKNOWN_FACE_COLOR
TEXT_COLOR = config.TEXT_COLOR
INFO_COLOR = config.INFO_COLOR

PANEL_BG_COLOR = (28, 28, 28)
PANEL_ACCENT_COLOR = (12, 115, 255)
STATUS_BG_COLOR = (10, 10, 10)

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


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def create_attendance_files():
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(CSV_FILE, index=False)
    else:
        try:
            df = pd.read_csv(CSV_FILE)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=["Name", "Date", "Time"])
            df.to_csv(CSV_FILE, index=False)
        else:
            if df.empty or not all(col in df.columns for col in ["Name", "Date", "Time"]):
                df = pd.DataFrame(columns=["Name", "Date", "Time"])
                df.to_csv(CSV_FILE, index=False)

    if not os.path.exists(EXCEL_FILE):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Attendance"
        sheet.append(["Name", "Date", "Time"])
        workbook.save(EXCEL_FILE)
    else:
        try:
            load_workbook(EXCEL_FILE)
        except Exception:
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Attendance"
            sheet.append(["Name", "Date", "Time"])
            workbook.save(EXCEL_FILE)


def load_workbook_safe(filename):
    try:
        return load_workbook(filename)
    except Exception:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Attendance"
        sheet.append(["Name", "Date", "Time"])
        workbook.save(filename)
        return workbook


def current_date():
    return datetime.now().strftime(config.DATE_FORMAT)


def current_time():
    return datetime.now().strftime(config.TIME_FORMAT)


create_attendance_files()


# =========================================================
# CONFIDENCE CALCULATION
# =========================================================

def face_confidence(raw_distance):
    percentage = max(0.0, min(100.0, 100.0 - raw_distance))
    return round(percentage, 2)


# =========================================================
# UI HELPERS
# =========================================================

def draw_transparent_panel(frame, x, y, w, h, color, alpha=0.75):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_side_panel(frame, status_message, attendance_count, face_names):
    panel_width = 360
    panel_margin = 12
    x = frame.shape[1] - panel_width - panel_margin
    y = panel_margin
    panel_height = frame.shape[0] - panel_margin * 2

    draw_transparent_panel(frame, x, y, panel_width, panel_height, PANEL_BG_COLOR, 0.88)
    cv2.rectangle(frame, (x, y), (x + panel_width, y + 46), PANEL_ACCENT_COLOR, cv2.FILLED)
    cv2.putText(
        frame,
        "ATTENDANCE PANEL",
        (x + 16, y + 33),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA
    )

    content_x = x + 18
    line_y = y + 74
    line_height = int(30 * FONT_SCALE) + 6

    cv2.putText(
        frame,
        f"Status: {status_message}",
        (content_x, line_y),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA
    )

    line_y += line_height
    cv2.putText(
        frame,
        f"Today: {attendance_count}",
        (content_x, line_y),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA
    )

    line_y += line_height + 4
    cv2.putText(
        frame,
        "Recent faces:",
        (content_x, line_y),
        FONT,
        FONT_SCALE,
        INFO_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA
    )

    recent = face_names[-4:] if face_names else ["None"]
    line_y += line_height
    for name in recent:
        cv2.putText(
            frame,
            f"- {name}",
            (content_x, line_y),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA
        )
        line_y += line_height

    footer_y = y + panel_height - 24
    cv2.putText(
        frame,
        "Press Q to quit",
        (content_x, footer_y),
        FONT,
        FONT_SCALE * 0.9,
        INFO_COLOR,
        1,
        cv2.LINE_AA
    )


def draw_status_bar(frame, status_message):
    y = frame.shape[0] - 52
    draw_transparent_panel(frame, 0, y, frame.shape[1], 52, STATUS_BG_COLOR, 0.7)
    cv2.putText(
        frame,
        status_message,
        (14, y + 34),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        "Center your face and avoid glare",
        (frame.shape[1] - 360, y + 34),
        FONT,
        FONT_SCALE * 0.85,
        INFO_COLOR,
        1,
        cv2.LINE_AA
    )


# =========================================================
# CHECK IF ALREADY MARKED TODAY
# =========================================================

def load_attendance_df():

    if not os.path.exists(CSV_FILE):
        return pd.DataFrame(columns=["Name", "Date", "Time"])

    try:
        df = pd.read_csv(CSV_FILE)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["Name", "Date", "Time"])

    if df.empty or not all(col in df.columns for col in ["Name", "Date", "Time"]):
        return pd.DataFrame(columns=["Name", "Date", "Time"])

    return df


def save_attendance_df(df):
    df.to_csv(CSV_FILE, index=False)


def already_marked(name):

    df = load_attendance_df()

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

    date = current_date()
    time = current_time()

    if already_marked(name) and not config.ALLOW_DUPLICATE_ATTENDANCE:
        message = f"{name} already marked today at {date} {time}"
        print(message)
        return message

    # ---------------- CSV ----------------

    df = load_attendance_df()

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

    save_attendance_df(df)

    # ---------------- Excel ----------------

    workbook = load_workbook_safe(EXCEL_FILE)

    sheet = workbook.active

    sheet.append([
        name,
        date,
        time
    ])

    workbook.save(EXCEL_FILE)

    message = f"Attendance marked: {name} at {date} {time}"
    print(message)
    return message


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

face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# =========================================================
# START FACE RECOGNITION
# =========================================================

frame_counter = 0
last_time = time.time()
prev_face_locations = []
prev_face_names = []
prev_face_confidences = []
prev_status_message = "No face detected"

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

    # Convert BGR -> grayscale
    gray_small_frame = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2GRAY
    )

    should_process = SKIP_FRAMES <= 1 or frame_counter % SKIP_FRAMES == 0

    face_locations = prev_face_locations
    face_names = prev_face_names
    face_confidences = prev_face_confidences
    status_message = prev_status_message

    if should_process:

        face_locations = face_cascade.detectMultiScale(
            gray_small_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        face_names = []
        face_confidences = []
        status_message = "No face detected"

        if len(face_locations) == 0:
            status_message = "No face detected"

        for (x, y, w, h) in face_locations:
            face_image = gray_small_frame[y:y+h, x:x+w]
            face_image = cv2.resize(face_image, (200, 200))
            face_image = cv2.equalizeHist(face_image)

            label, raw_confidence = recognizer.predict(face_image)
            confidence = face_confidence(raw_confidence)

            if raw_confidence <= CONFIDENCE_THRESHOLD and label in reverse_labels:
                name = reverse_labels[label]
                status_message = mark_attendance(name)
                face_names.append(name)
                face_confidences.append(confidence)
            else:
                status_message = "Unknown person detected"
                face_names.append("Unknown")
                face_confidences.append(0)

        prev_face_locations = face_locations
        prev_face_names = face_names
        prev_face_confidences = face_confidences
        prev_status_message = status_message

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

    attendance_df = load_attendance_df()

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

    draw_side_panel(frame, status_message, attendance_count, face_names)
    draw_status_bar(frame, status_message)

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