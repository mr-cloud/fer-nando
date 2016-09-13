"""
@AmineHorseman 
Sep 12th, 2016
"""
import tensorflow as tf
from tflearn import DNN
import imutils
import cv2
import time
import dlib

from parameters import NETWORK, DATASET
from model import build_model
from predict import load_model, predict

class EmotionRecognizer:

    CAMERA_SOURCE = 0
    FACE_DETECTION_CLASSIFIER = "lbpcascade_frontalface.xml"
    BOX_COLOR = (0, 255, 0)
    TEXT_COLOR = (0, 255, 0)
    TIME_TO_MEMORIZE_PERSON = 5
    SHOW_CONFIDENCE = False

    def __init__(self):
       
        # initializebevideo stream
        self.video_stream = cv2.VideoCapture(self.CAMERA_SOURCE)
  
        self.face_detector = cv2.CascadeClassifier(self.FACE_DETECTION_CLASSIFIER)

        self.shape_predictor = None
        if NETWORK.use_landmarks:
            self.shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)
        
        self.model = load_model()

    def predict_emotion(self, image):
        image.resize([NETWORK.input_size, NETWORK.input_size], refcheck=False)
        emotion, confidence = predict(image, self.model, self.shape_predictor)
        return emotion, confidence

    def recognize_emotions(self):
        failedFramesCount = 0
        detected_faces = []
        while True:
            grabbed, frame = self.video_stream.read()

            if grabbed:
                # detection phase
                frame = imutils.resize(frame, width=600)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # detect faces
                faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    if w < 30 and h<30: # skip the small faces (probably false detections)
                        continue

                    # bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), self.BOX_COLOR, 2)

                    # try to recognize emotion
                    face = gray[y:y+h, x:x+w].copy()
                    label, confidence = self.predict_emotion(face)
                    if self.SHOW_CONFIDENCE:
                        text = "{0} ({1:.1f}%)".format(label, confidence*100)
                    else:
                        text = label
                    if label is not None:
                        cv2.putText(frame, text, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.TEXT_COLOR, 2)

                # display images
                cv2.imshow("Facial Expression Recognition", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break            
            else:
                failedFramesCount += 1
                if failedFramesCount > 10:
                    print "can't grab frames"
                    break

        self.video_stream.release()
        cv2.destroyAllWindows()

r = EmotionRecognizer()
r.recognize_emotions()