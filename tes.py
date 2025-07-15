import face_recognition

# Load gambar menggunakan face_recognition langsung
image = face_recognition.load_image_file("OpenCV/images/fikri.png")

# Deteksi wajah
face_locations = face_recognition.face_locations(image)

print(f"Detected {len(face_locations)} face(s).")