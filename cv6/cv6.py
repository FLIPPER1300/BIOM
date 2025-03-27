import dlib
import cv2
import numpy as np

# Načítanie deep learningového modelu detekcie tvárí
detector = dlib.cnn_face_detection_model_v1(dlib.get_frontal_face_detector())

# Načítanie obrázka
image_path = "0fff1f631ee0d0c1.jpg"
image = cv2.imread(image_path)

# Konverzia na odtiene šedej
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detekcia tvárí
faces = detector(gray)

# Kreslenie obrysov detekovaných tvárí
for face in faces:
    x, y, w, h = (face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height())
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Zobrazenie výsledku
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
