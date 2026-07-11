"""
=========================================================
AI Face Attendance System
Configuration File

Author : Malay Maity
=========================================================
"""

import os
import cv2

# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_FOLDER = os.path.join(BASE_DIR, "images")

ATTENDANCE_FOLDER = os.path.join(BASE_DIR, "Attendance")

ENCODING_FILE = os.path.join(BASE_DIR, "encodings.pkl")

MODEL_FILE = os.path.join(BASE_DIR, "lbph_model.yml")

LABELS_FILE = os.path.join(BASE_DIR, "labels.pkl")

CSV_FILE = os.path.join(
    ATTENDANCE_FOLDER,
    "Attendance.csv"
)

EXCEL_FILE = os.path.join(
    ATTENDANCE_FOLDER,
    "Attendance.xlsx"
)

UNKNOWN_FOLDER = os.path.join(
    BASE_DIR,
    "Unknown"
)

LOG_FOLDER = os.path.join(
    BASE_DIR,
    "logs"
)

HAAR_CASCADE_PATH = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)

# =========================================================
# CAMERA SETTINGS
# =========================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280

FRAME_HEIGHT = 720

PROCESS_SCALE = 0.25

# =========================================================
# FACE RECOGNITION SETTINGS
# =========================================================

FACE_DETECTION_MODEL = "haar"
# Options:
# "haar" -> OpenCV Haar Cascade
# "lbp" -> OpenCV LBP Cascade

LBPH_CONFIDENCE_THRESHOLD = 150

SHOW_CONFIDENCE = True

MARK_UNKNOWN_FACE = True

SAVE_UNKNOWN_FACE = False

# =========================================================
# ATTENDANCE SETTINGS
# =========================================================

ALLOW_DUPLICATE_ATTENDANCE = False

DATE_FORMAT = "%d-%m-%Y"

TIME_FORMAT = "%H:%M:%S"

DATETIME_FORMAT = "%d-%m-%Y %H:%M:%S"

# =========================================================
# DISPLAY SETTINGS
# =========================================================

WINDOW_NAME = "AI Face Attendance System"

FONT_SCALE = 0.6

FONT_THICKNESS = 2

BOX_THICKNESS = 2

KNOWN_FACE_COLOR = (0, 255, 0)

UNKNOWN_FACE_COLOR = (0, 0, 255)

TEXT_COLOR = (255, 255, 255)

INFO_COLOR = (255, 255, 0)

# =========================================================
# EMAIL SETTINGS
# =========================================================

ENABLE_EMAIL = False

SMTP_SERVER = "smtp.gmail.com"

SMTP_PORT = 587

EMAIL_SENDER = "your_email@gmail.com"

EMAIL_PASSWORD = "YOUR_APP_PASSWORD"

EMAIL_RECEIVER = "teacher@gmail.com"

EMAIL_SUBJECT = "Daily Attendance Report"

# =========================================================
# LOGGING
# =========================================================

ENABLE_LOGGING = True

LOG_FILE = os.path.join(
    LOG_FOLDER,
    "attendance.log"
)

# =========================================================
# UNKNOWN FACE SETTINGS
# =========================================================

UNKNOWN_IMAGE_FORMAT = ".jpg"

UNKNOWN_PREFIX = "unknown_"

# =========================================================
# EXCEL SETTINGS
# =========================================================

SHEET_NAME = "Attendance"

AUTO_ADJUST_COLUMNS = True

# =========================================================
# PERFORMANCE
# =========================================================

SKIP_FRAMES = 1

SHOW_FPS = True

# =========================================================
# SECURITY
# =========================================================

USE_ENCODING_CACHE = True

AUTO_CREATE_DIRECTORIES = True

# =========================================================
# CREATE REQUIRED DIRECTORIES
# =========================================================

if AUTO_CREATE_DIRECTORIES:

    directories = [
        IMAGE_FOLDER,
        ATTENDANCE_FOLDER,
        UNKNOWN_FOLDER,
        LOG_FOLDER
    ]

    for directory in directories:

        if not os.path.exists(directory):
            os.makedirs(directory)

print("=" * 60)
print("Configuration Loaded Successfully")
print("=" * 60)