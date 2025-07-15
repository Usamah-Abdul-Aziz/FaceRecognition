import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import csv

# === Load known faces ===
known_face_encodings = []
known_face_names = []

face_dir = 'OpenCV/images'  # Directory containing known face images

for filename in os.listdir(face_dir):
    if filename.endswith(('.jpg', '.png')):
        img_path = os.path.join(face_dir, filename)
        img = face_recognition.load_image_file(img_path)

        if len(img.shape) == 2:  # If grayscale, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # If RGBA, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Check if the image is valid for face recognition
        if img.dtype != np.uint8:
            raise ValueError(f"Unsupported image type: {filename}")

        encodings = face_recognition.face_encodings(img)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

# === Initialize Attendance CSV ===
# Make sure file exists or create it if not
attendance_file = 'OpenCV/attendance.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Time\n')  # Header

def mark_attendance(name):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
    
    with open(attendance_file, 'r+') as f:
        existing_data = f.readlines()
        recorded_names = [line.split(',')[0] for line in existing_data]

        if name not in recorded_names:
            f.write(f'{name},{dt_string}\n')
            print(f'[LOG] Attendance marked for {name}')


# === Start Webcam ===
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, frame = cap.read()
    if not success:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            mark_attendance(name)

            top, right, bottom, left = [v * 4 for v in face_location]  # scale back up
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Attendance System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()