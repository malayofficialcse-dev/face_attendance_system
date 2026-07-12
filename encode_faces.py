"""
=========================================================
AI Face Attendance System
Face Encoding Generation Script
Author : Malay Maity
=========================================================
"""

import os
import cv2
import numpy as np
import pickle
import config
import re


def create_encodings():
    if not os.path.exists(config.IMAGE_FOLDER):
        print(f"Image folder not found: {config.IMAGE_FOLDER}")
        print("Please add training images and run this script again.")
        return

    image_files = [
        f for f in os.listdir(config.IMAGE_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print("No images found in the images folder.")
        print("Add face pictures to the images directory and run this script again.")
        return

    face_cascade = cv2.CascadeClassifier(config.HAAR_CASCADE_PATH)

    training_faces = []
    training_labels = []
    label_map = {}
    next_label = 0

    for image_file in image_files:
        image_path = os.path.join(config.IMAGE_FOLDER, image_file)
        base_name = os.path.splitext(image_file)[0]
        # Remove trailing numbers like _1, -2, or space 3 to group multiple images for the same person
        name = re.sub(r'[\-_ ]\d+$', '', base_name)

        print(f"Processing: {image_file} -> {name}")

        image = cv2.imread(image_path)

        if image is None:
            print(f"  Warning: Could not read {image_file}. Skipping.")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            print(f"  Warning: No face found in {image_file}. Skipping.")
            continue

        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        face_image = gray[y:y+h, x:x+w]
        face_image = cv2.resize(face_image, (200, 200))
        face_image = cv2.equalizeHist(face_image)

        # Augmented face: horizontal flip to double the training data
        face_image_flipped = cv2.flip(face_image, 1)

        label = label_map.get(name)
        if label is None:
            label = next_label
            label_map[name] = label
            next_label += 1

        training_faces.append(face_image)
        training_labels.append(label)
        
        # Add the flipped version as well
        training_faces.append(face_image_flipped)
        training_labels.append(label)

    if not training_faces:
        print("No valid face images found. Check your images and try again.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(training_faces, np.array(training_labels))
    recognizer.write(config.MODEL_FILE)

    with open(config.LABELS_FILE, "wb") as f:
        pickle.dump(label_map, f)

    print(f"\nSaved {len(label_map)} face labels to {config.LABELS_FILE}")
    print(f"Saved LBPH model to {config.MODEL_FILE}")


if __name__ == '__main__':
    create_encodings()
