# -*- coding: utf-8 -*-
"""
@leo
2018/8/25
"""
import operator
import thread
import time

import cv2
import dlib
import imutils
import numpy as np

from parameters import NETWORK, DATASET, VIDEO_PREDICTOR
from predict import load_model, predict


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

        # FIXME more accurate face detector
        self.face_detector = cv2.CascadeClassifier(VIDEO_PREDICTOR.face_detection_classifier)

        self.shape_predictor = None
        if NETWORK.use_landmarks:
            self.shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)
        
        self.model = load_model()

        self.house = None
        # self.PUBLIC_OPINIONS = dict(zip(VIDEO_PREDICTOR.emotions, ['群情激愤', '其乐融融', '哀鸿遍野', '瞠目结舌', '索然无味']))
        self.PUBLIC_OPINIONS = dict(zip(VIDEO_PREDICTOR.emotions, VIDEO_PREDICTOR.emotions))
        self.OPINION_COLORS = dict(zip(VIDEO_PREDICTOR.emotions, [(0, 0, 255), (0, 173, 255), (255, 0, 77), (255, 0, 213), (0, 255, 26)]))
        self.pub_op = VIDEO_PREDICTOR.emotions[-1]
        self.is_camera_working = True
        self.is_exit = False
        self.opinions = dict(zip(VIDEO_PREDICTOR.emotions, np.zeros(5, dtype=np.int32)))

    def predict_emotion(self, image):
        emotion, confidence = predict(image, self.model, self.shape_predictor)
        return emotion, confidence

    def recognize_emotions(self, thread_name, delay, time_to_wait_between_predictions):
        failedFramesCount = 0
        start_time = time.time()
        last_predicts = list()
        is_predict_enabled = True
        while not self.is_exit:
            # print('{}: {}'.format(thread_name, time.ctime(time.time())))
            grabbed, frame = self.video_stream.read()

            if grabbed:
                # detection phase
                # keep aspect ratio
                frame = imutils.resize(frame, width=600)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # detect faces
                # parameters: image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]
                # scaleFactor: the smaller, the more chance finding faces;
                # minNeighbors: the higher, the higher quality of face found
                faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=5, minSize=(30, 30))
                print('{}: #faces detected: {}'.format(thread_name, len(faces)))
                if time.time() - start_time > time_to_wait_between_predictions:
                    is_predict_enabled = True
                    last_predicts = list()
                idx = 0
                for (x, y, w, h) in faces:
                    if w < 30 and h < 30:  # skip the small faces (probably false detections)
                        print('{}: face is ignored cause too small: {}, {}', thread_name, w, h)
                        continue

                    # bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), self.BOX_COLOR, 2)

                    # try to recognize emotion
                    face = gray[y:y+h, x:x+w].copy()
                    if is_predict_enabled:
                        label, confidence = self.predict_emotion(face)
                        last_predicts.append((label, confidence))
                    else:
                        label, confidence = last_predicts[min(idx, len(last_predicts)-1)]
                    # display and send message by socket
                    if VIDEO_PREDICTOR.show_confidence:
                        text = "{0} ({1:.1f}%)".format(label, confidence*100)
                    else:
                        text = label
                    if label is not None:
                        self.opinions[label] += 1
                        cv2.putText(frame, text, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.TEXT_COLOR, 2)
                    idx += 1
                if is_predict_enabled and len(faces) > 0:
                    is_predict_enabled = False
                    start_time = time.time()
                self.house = frame
            else:
                failedFramesCount += 1
                if failedFramesCount > 10:
                    print "can't grab frames"
                    break
            time.sleep(delay)
        self.is_camera_working = False

    def display(self, thread_name, delay, title_change_interval):
        start_time = time.time()
        while self.is_camera_working:
            # print('{}: {}'.format(thread_name, time.ctime(time.time())))
            # display images
            if self.house is not None:
                height, width, channels = self.house.shape
                if time.time() - start_time > title_change_interval:
                    self.pub_op = max(self.opinions.iteritems(), key=operator.itemgetter(1))[0]
                    if self.opinions[self.pub_op] == 0:
                        self.pub_op = VIDEO_PREDICTOR.emotions[-1]
                    print('{}: public opinion {} with count {}'.format(thread_name, self.pub_op, self.opinions[self.pub_op]))
                    self.opinions = dict(zip(VIDEO_PREDICTOR.emotions, np.zeros(5, dtype=np.int32)))
                    start_time = time.time()
                # params: img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]
                cv2.putText(self.house, self.PUBLIC_OPINIONS[self.pub_op], (width/2, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.OPINION_COLORS[self.pub_op], 4)
                cv2.imshow("Public Opinion Analysis", self.house)
                # delay <= 1 ms, 0 means forever
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            time.sleep(delay)
        self.is_exit = True
        self.video_stream.release()
        cv2.destroyAllWindows()


r = EmotionRecognizer()
thread.start_new_thread(r.display, ('thread-display', VIDEO_PREDICTOR.time_to_wait_between_display_pub, VIDEO_PREDICTOR.title_change_interval_pub))
time.sleep(1)
thread.start_new_thread(r.recognize_emotions, ('thread-recognize', VIDEO_PREDICTOR.frame_rate_pub, VIDEO_PREDICTOR.time_to_wait_between_predictions_pub))
