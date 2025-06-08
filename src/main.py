import cv2 
import numpy as np

# Creating a function that detects a face based on input image. Using OpenCV´s Haar Cascade algorithm for detecting objects from an image
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.