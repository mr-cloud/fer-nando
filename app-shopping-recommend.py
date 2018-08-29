# -*- coding: utf-8 -*-
"""
@leo
2018/8/28
"""
import argparse
import operator
import os
import thread
import time

import cv2
import dlib
import imutils
import numpy as np

from parameters import NETWORK, DATASET, VIDEO_PREDICTOR
from predict import load_model, predict

import random


class EmotionRecognizer:
    
    BOX_COLOR = (0, 255, 0)
    TEXT_COLOR = (0, 255, 0)

    def clearCapture(self, capture):
        capture.release()
        cv2.destroyAllWindows()

    def countCameras(self):
        n = 0
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                ret, frame = cap.read()
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.clearCapture(cap)
                n += 1
            except:
                self.clearCapture(cap)
                break
        return n

    def __init__(self):

        n_cam = self.countCameras()
        print('#webcam={}'.format(n_cam))
        # initializebevideo stream
        self.video_stream = cv2.VideoCapture(n_cam - 1)
  
        self.face_detector = cv2.CascadeClassifier(VIDEO_PREDICTOR.face_detection_classifier)

        self.shape_predictor = None
        if NETWORK.use_landmarks:
            self.shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)
        
        self.model = load_model()

        self.house = None
        self.cur_pub_op = VIDEO_PREDICTOR.emotions[-1]
        # self.PUBLIC_OPINIONS = dict(zip(VIDEO_PREDICTOR.emotions, ['群情激愤', '其乐融融', '哀鸿遍野', '瞠目结舌', '索然无味']))
        self.PUBLIC_OPINIONS = dict(zip(VIDEO_PREDICTOR.emotions, VIDEO_PREDICTOR.emotions))
        self.OPINION_COLORS = dict(zip(VIDEO_PREDICTOR.emotions, [(0, 0, 255), (0, 173, 255), (255, 0, 77), (255, 0, 213), (0, 255, 26)]))
        self.pub_op = VIDEO_PREDICTOR.emotions[-1]
        self.is_camera_working = True
        self.is_exit = False
        self.opinions = dict(zip(VIDEO_PREDICTOR.emotions, np.zeros(5, dtype=np.int32)))

        self.items = dict()
        self.classified_items = dict(zip(VIDEO_PREDICTOR.emotions, np.ndarray(shape=(len(VIDEO_PREDICTOR.emotions), 0), dtype=np.int32).tolist()))
        self.sentiments = dict(zip(VIDEO_PREDICTOR.emotions, VIDEO_PREDICTOR.sentiment_scores))

    def predict_emotion(self, image):
        emotion, confidence = predict(image, self.model, self.shape_predictor)
        return emotion, confidence

    def recognize_emotions(self, thread_name, delay):
        failedFramesCount = 0
        while not self.is_exit:
            print('{}: {}'.format(thread_name, time.ctime(time.time())))

            grabbed, frame = self.video_stream.read()

            if grabbed:
                # detection phase
                # keep aspect ratio
                frame = imutils.resize(frame, width=600)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # detect faces
                faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                print('#faces detected: {}'.format(len(faces)))
                for (x,y,w,h) in faces:
                    if w < 30 and h<30: # skip the small faces (probably false detections)
                        continue

                    # bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), self.BOX_COLOR, 2)

                    # try to recognize emotion
                    face = gray[y:y+h, x:x+w].copy()
                    label, confidence = self.predict_emotion(face)
                    self.opinions[label] += 1
                    # display and send message by socket
                    if VIDEO_PREDICTOR.show_confidence:
                        text = "{0} ({1:.1f}%)".format(label, confidence*100)
                    else:
                        text = label
                    if label is not None:
                        cv2.putText(frame, text, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.TEXT_COLOR, 2)

                self.house = frame
                self.pub_op = max(self.opinions.iteritems(), key=operator.itemgetter(1))[0]
                if self.opinions[self.pub_op] == 0:
                    self.pub_op = VIDEO_PREDICTOR.emotions[-1]
                print('public opinion {} with count {}'.format(self.pub_op, self.opinions[self.pub_op]))

            else:
                failedFramesCount += 1
                if failedFramesCount > 10:
                    print "can't grab frames"
                    break
            time.sleep(delay)
        self.is_camera_working = False

    def display(self, thread_name, delay):
        start_time = time.time()
        while self.is_camera_working:
            print('{}: {}'.format(thread_name, time.ctime(time.time())))
            # display images
            if self.house is not None:
                height, width, channels = self.house.shape
                if time.time() - start_time > VIDEO_PREDICTOR.title_change_interval:
                    self.cur_pub_op = self.pub_op
                    self.opinions = dict(zip(VIDEO_PREDICTOR.emotions, np.zeros(5, dtype=np.int32)))
                    start_time = time.time()
                cv2.putText(self.house, 'Positive: ' + str(self.sentiments[self.cur_pub_op]), (width/2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.OPINION_COLORS[self.cur_pub_op], 2)
                cv2.imshow("Public Opinion Analysis", self.house)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            time.sleep(delay)
        self.is_exit = True
        self.video_stream.release()
        cv2.destroyAllWindows()

    def load_items(self, path):
        cnt = 0
        for file in os.listdir(path):
            filename = os.path.join(path, file)
            img = cv2.imread(filename)
            # show img and detect faces
            img = imutils.resize(img, width=300)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # detect faces
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
            print('#faces detected: {}'.format(len(faces)))
            opinions = dict(zip(VIDEO_PREDICTOR.emotions, np.zeros(5, dtype=np.int32)))
            for (x, y, w, h) in faces:
                if w < 30 and h < 30:  # skip the small faces (probably false detections)
                    continue

                # bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), self.BOX_COLOR, 2)

                # try to recognize emotion
                face = gray[y:y + h, x:x + w].copy()
                label, confidence = self.predict_emotion(face)
                opinions[label] += 1
                # display and send message by socket
                if VIDEO_PREDICTOR.show_confidence:
                    text = "{0} ({1:.1f}%)".format(label, confidence * 100)
                else:
                    text = label
                if label is not None:
                    cv2.putText(img, text, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.TEXT_COLOR, 2)

            emotion = max(opinions.iteritems(), key=operator.itemgetter(1))[0]
            if opinions[emotion] == 0:
                emotion = VIDEO_PREDICTOR.emotions[-1]
            cv2.imshow(filename + ':' + emotion, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # resize img

            self.items[cnt] = (emotion, img)
            self.classified_items[emotion].append(cnt)
            cnt += 1
        print('Loaded {} imgs.'.format(cnt))

    def recommend(self, thread_name, delay):
        while not self.is_exit:
            print('{}: {}'.format(thread_name, time.ctime(time.time())))
            # recommendation
            if self.sentiments[self.cur_pub_op] >= 0.5:
                item_index = random.choice(self.classified_items["happy"])
            else:
                item_index = random.choice(self.classified_items["neutral"])
            emotion = self.items[item_index][0]
            img = self.items[item_index][1]
            height, width, channels = self.house.shape
            cv2.putText(img, emotion, (width/2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.TEXT_COLOR, 2)
            cv2.imshow('Shopping Recommendation', img)
            time.sleep(delay)


# parse arg to see if we need to launch training now or not yet
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="Image files path to load")
args = parser.parse_args()

if args.path:
    if os.path.isdir(args.path):
        r = EmotionRecognizer()
        r.load_items(args.path)
        thread.start_new_thread(r.display, ('thread-display', VIDEO_PREDICTOR.time_to_wait_between_display))
        time.sleep(1)
        thread.start_new_thread(r.recognize_emotions, ('thread-recognize', VIDEO_PREDICTOR.frame_rate))
        time.sleep(1)
        thread.start_new_thread(r.recommend, ('thread-recommend', VIDEO_PREDICTOR.recommend_rate))
    else:
        print "Error: path '{}' not found".format(args.path)