from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from uuid import uuid4

app = Flask(__name__)
CORS(app)

# === LOAD KNOWN FACES ===
known_face_encodings = []
known_face_names = []

face_dir = 'OpenCV/images'

for filename in os.listdir(face_dir):
  if filename.endswith(('.jpg', '.png')):
    path = os.path.join(face_dir, filename)
    image = face_recognition.load_image_file(path)

    if len(image.shape) == 2:
      image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
      image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    encodings = face_recognition.face_encodings(image)
    if encodings:
      known_face_encodings.append(encodings[0])
      known_face_names.append(os.path.splitext(filename)[0])

# === ATTENDANCE FILE ===
attendance_file = 'OpenCV/attendance.csv'
if not os.path.exists(attendance_file):
  with open(attendance_file, 'w') as f:
    f.write('Name,Time\n')

def mark_attendance(name):
  now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  with open(attendance_file, 'r+') as f:
    lines = f.readlines()
    names = [line.split(',')[0] for line in lines]
    if name not in names:
      f.write(f'{name},{now}\n')

def save_absen_image(temp_path, name):
  today = datetime.now().strftime('%Y-%m-%d')
  folder_path = os.path.join('OpenCV', 'absen', today)
  os.makedirs(folder_path, exist_ok=True)

  base_filename = secure_filename(name)
  final_path = os.path.join(folder_path, f"{base_filename}.jpg")
  count = 1

  while os.path.exists(final_path):
    final_path = os.path.join(folder_path, f"{base_filename}{count}.jpg")
    count += 1

  os.rename(temp_path, final_path)
  return final_path

@app.route('/recognize', methods=['POST'])
def recognize():
  if 'image' not in request.files:
    return jsonify({'status': 'error', 'message': 'No image uploaded'}), 400

  file = request.files['image']
  filename = f"{uuid4().hex}.jpg"
  temp_path = os.path.join('OpenCV', filename)
  file.save(temp_path)

  image = face_recognition.load_image_file(temp_path)
  rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  face_locations = face_recognition.face_locations(rgb_image)
  face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

  for face_encoding in face_encodings:
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    if len(face_distances) > 0:
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]
        mark_attendance(name)
        saved_path = save_absen_image(temp_path, name)
        return jsonify({'status': 'success', 'name': name, 'saved_to': saved_path})

  # Kalau wajah tidak dikenali
  saved_path = save_absen_image(temp_path, 'unknown')
  return jsonify({'status': 'unknown', 'message': 'Wajah tidak dikenali', 'saved_to': saved_path})

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)