# -*- coding: utf-8 -*-
"""
@leo
2018/8/25
"""
import operator
import threading

import time

import cv2
import dlib
import imutils
import numpy as np

from parameters import NETWORK, DATASET, VIDEO_PREDICTOR
from predict import load_model, predict

from PIL import ImageFont, ImageDraw, Image


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
        self.PUBLIC_OPINIONS = dict(zip(VIDEO_PREDICTOR.emotions, VIDEO_PREDICTOR.emotions))
        # color mode: BGR
        self.OPINION_COLORS = dict(zip(VIDEO_PREDICTOR.emotions, [(0, 0, 255), (0, 173, 255), (255, 0, 77), (255, 0, 213), (0, 255, 26)]))
        self.pub_op = VIDEO_PREDICTOR.emotions[-1]
        self.is_camera_working = True
        self.is_exit = False
        self.opinions = dict(zip(VIDEO_PREDICTOR.emotions, np.zeros(5, dtype=np.int32)))

        self.CHINESE_PUBLIC_OPINIONS = dict(zip(VIDEO_PREDICTOR.emotions, [u'群情激愤|>_<|', u'其乐融融 (^o^)', u'哀鸿遍野 /T_T\\', u'瞠目结舌)O.O(', u'索然无味...']))
        self.CHINESE_OPINION_COLORS = dict(zip(VIDEO_PREDICTOR.emotions, [(0, 0, 255, 0), (0, 173, 255, 0), (255, 0, 77, 0), (255, 0, 255, 0), (0, 255, 26, 0)]))
        self.frame = None
        # number of detected faces as the online users number
        self.numUser = 0
        self.numFace = 0

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
                frame = imutils.resize(frame, width=800)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # detect faces
                # parameters: image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]
                # scaleFactor: the smaller, the more chance finding faces;
                # minNeighbors: the higher, the higher quality of face found
                min_w = 100
                min_h = 100
                faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(min_w, min_h))
                self.numFace = len(faces)
                # print('{}: #faces detected: {}'.format(thread_name, len(faces)))
                if time.time() - start_time > time_to_wait_between_predictions:
                    is_predict_enabled = True
                    last_predicts = list()
                idx = 0
                for (x, y, w, h) in faces:
                    # print('{}: width={}, height={}'.format(thread_name, w, h))
                    if w < min_w and h < min_h:  # skip the small faces (probably false detections)
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
                self.frame = np.array(frame)
            else:
                failedFramesCount += 1
                if failedFramesCount > 10:
                    print("can't grab frames")
                    break
            time.sleep(delay)
        self.is_camera_working = False

    def display(self, thread_name, delay, title_change_interval):
        start_time = time.time()
        win_name = "Public Opinion Analysis"
        cv2.namedWindow(win_name)
        cv2.moveWindow(win_name, 400, 60)
        ## Use simsum.ttc to write Chinese.
        fontpath = "fonts/simsun.ttc"
        font = ImageFont.truetype(fontpath, 32)
        while self.is_camera_working:
            # print('{}: {}'.format(thread_name, time.ctime(time.time())))
            # display images
            if self.frame is not None:
                self.house = np.array(self.frame)
                height, width, channels = self.house.shape
                if time.time() - start_time > title_change_interval:
                    self.pub_op = max(self.opinions.items(), key=operator.itemgetter(1))[0]
                    self.numUser = self.numFace
                    if self.opinions[self.pub_op] == 0:
                        self.pub_op = VIDEO_PREDICTOR.emotions[-1]
                    print('{}: public opinion {} with count {}'.format(thread_name, self.pub_op, self.opinions[self.pub_op]))
                    self.opinions = dict(zip(VIDEO_PREDICTOR.emotions, np.zeros(5, dtype=np.int32)))
                    start_time = time.time()
                img_pil = Image.fromarray(self.house)
                draw = ImageDraw.Draw(img_pil)
                draw.text((int(width/2), 20), self.CHINESE_PUBLIC_OPINIONS[self.pub_op], font=font, fill=self.CHINESE_OPINION_COLORS[self.pub_op])
                draw.text((10, 20), u'直播间: ' + str(self.numUser) + u'人', font=font, fill=(255, 255, 255, 0))
                self.house = np.array(img_pil)
                cv2.imshow(win_name, self.house)
                # delay <= 1 ms, 0 means forever
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            time.sleep(delay)
        self.is_exit = True
        self.video_stream.release()
        cv2.destroyAllWindows()


r = EmotionRecognizer()
threading.Thread(target=r.display, args=('thread-display', VIDEO_PREDICTOR.time_to_wait_between_display_pub, VIDEO_PREDICTOR.title_change_interval_pub)).start()
time.sleep(1)
threading.Thread(target=r.recognize_emotions, args=('thread-recognize', VIDEO_PREDICTOR.frame_rate_pub, VIDEO_PREDICTOR.time_to_wait_between_predictions_pub)).start()
