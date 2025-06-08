import cv2 
import numpy as np

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Creating a function that detects a face based on input image. Using OpenCV´s Haar Cascade algorithm for detecting objects from an image
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(gray, 1.1, 5)
    
    # Checking if face is detected, and if so, create a rectangle around face when detecting
    if len(faces) > 0:
        for (x, y, w, h) in face:
          cv2.rectangle(image, (x,y), (x + w, y + h), (255, 255, 0), 2)  # Creating the rectangle based on measures and choosing color and line thickness
        return image
    else:
        return image
    
def capture():
    cap = cv2.VideoCapture(0)  # Start camera capture

    while True:
        ret, frame = cap.read()
        frame = detect_face(frame)

        cv2.imshow('Face Detection', frame)

        if cv2.waitkey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()