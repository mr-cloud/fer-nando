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

        self.items = dict()
        self.recommend_pool = None
        self.classified_items = dict(zip(VIDEO_PREDICTOR.emotions, np.ndarray(shape=(len(VIDEO_PREDICTOR.emotions), 0), dtype=np.int32).tolist()))
        self.sentiments = dict(zip(VIDEO_PREDICTOR.emotions, VIDEO_PREDICTOR.sentiment_scores))

        self.frame = None
        self.interests = dict()

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
                min_w = 100
                min_h = 100
                faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(min_w, min_h))
                print('{}: #faces detected: {}'.format(thread_name, len(faces)))
                if time.time() - start_time > time_to_wait_between_predictions:
                    is_predict_enabled = True
                    last_predicts = list()
                idx = 0
                for (x, y, w, h) in faces:
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
                    print "can't grab frames"
                    break
            time.sleep(delay)
        self.is_camera_working = False

    # CV GUI should run in one thread for interaction with user.
    def display(self, thread_name, delay, title_change_interval, file_path):
        self.load_items(file_path)
        start_time = time.time()
        win_name_recommend = 'Recommendation Item'
        win_name_display = "Recommendation User Sentiment"
        cv2.namedWindow(win_name_display)
        cv2.moveWindow(win_name_display, 80, 60)
        cv2.namedWindow(win_name_recommend)
        cv2.moveWindow(win_name_recommend, 800, 60)
        time.sleep(1)
        thread.start_new_thread(r.recognize_emotions, ('thread-recognize', VIDEO_PREDICTOR.frame_rate_shop, VIDEO_PREDICTOR.time_to_wait_between_predictions_shop))
        title = 'likeness 0.5'
        fontpath = "fonts/simsun.ttc"
        font = ImageFont.truetype(fontpath, 24)
        item = None
        item_idx = -1
        pv_cnt = 0
        while self.is_camera_working:
            # print('{}: {}'.format(thread_name, time.ctime(time.time())))
            # display images
            if self.frame is not None:
                self.house = np.array(self.frame)
                height, width, channels = self.house.shape
                if time.time() - start_time > title_change_interval:
                    self.pub_op = max(self.opinions.iteritems(), key=operator.itemgetter(1))[0]
                    if self.opinions[self.pub_op] == 0:
                        self.pub_op = VIDEO_PREDICTOR.emotions[-1]
                    print('{}: public opinion {} with count {}'.format(thread_name, self.pub_op, self.opinions[self.pub_op]))
                    self.opinions = dict(zip(VIDEO_PREDICTOR.emotions, np.zeros(5, dtype=np.int32)))
                    title = 'likeness {}'.format(self.sentiments[self.pub_op])
                    if item is not None:
                        img = np.array(item)
                        width, height, channels = img.shape
                        cv2.putText(img, title, (80, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    self.OPINION_COLORS[self.pub_op], 4)
                        cv2.imshow(win_name_recommend, img)
                        cv2.waitKey(1)
                    start_time = time.time()
                # params: img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]
                cv2.putText(self.house, self.PUBLIC_OPINIONS[self.pub_op], (width/2, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.OPINION_COLORS[self.pub_op], 4)
                cv2.imshow(win_name_display, self.house)
                # delay <= 1 ms, 0 means forever
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord('n'):
                    # save state
                    if item_idx != -1:
                        self.interests[item_idx] = self.sentiments[self.pub_op]
                    item_idx, item = self.recommend()
                    if item_idx is None:
                        print('No more items to recommend! To be continued!')
                        break
                    cv2.imshow(win_name_recommend, item)
                    cv2.waitKey(1)
                    pv_cnt += 1
                    print('#recommendation={}'.format(pv_cnt))
            time.sleep(delay)
        cv2.destroyWindow(win_name_display)
        cv2.destroyWindow(win_name_recommend)
        self.is_exit = True
        self.video_stream.release()
        self.show_interests()

    def load_items(self, path):
        cnt = 0
        win_name = 'Predicting items while loading'
        cv2.namedWindow(win_name)
        cv2.moveWindow(win_name, 80, 60)
        for file in os.listdir(path):
            filename = os.path.join(path, file)
            img = cv2.imread(filename)
            # show img and detect faces
            img = imutils.resize(img, width=300)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # detect faces
            faces = self.face_detector.detectMultiScale(gray, 1.05, 3)
            print('#faces detected: {}'.format(len(faces)))
            opinions = dict(zip(VIDEO_PREDICTOR.emotions, np.zeros(5, dtype=np.int32)))
            for (x, y, w, h) in faces:
                print('face size: {}, {}', w, h)
                if (w < 30 and h < 30) or (w > 150 and h > 150):  # skip the small faces (probably false detections)
                    print('face is ignored cause too small or too large: {}, {}', w, h)
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
                    cv2.putText(img, text, (x + 20, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.TEXT_COLOR, 2)

            emotion = max(opinions.iteritems(), key=operator.itemgetter(1))[0]
            if opinions[emotion] == 0:
                emotion = VIDEO_PREDICTOR.emotions[-1]
            cv2.imshow(win_name, img)
            cv2.waitKey(0)
            self.items[cnt] = (emotion, img)
            self.classified_items[emotion].append(cnt)
            self.interests[cnt] = 0.5
            cnt += 1
        self.recommend_pool = self.items.keys()
        cv2.destroyWindow(win_name)
        print('Loaded {} imgs.'.format(cnt))

    def recommend(self):
        # uniformly randomly selection from the pool
        if len(self.recommend_pool) == 0:
            return None, None
        item_index = random.choice(self.recommend_pool)
        self.recommend_pool.remove(item_index)
        return item_index, np.array(self.items[item_index][1])

    def show_interests(self):
        ranks = sorted(self.interests, key=self.interests.get, reverse=True)
        print(ranks)
        print(self.interests)
        for rank, item_idx in enumerate(ranks[0:3]):
            win_name = '{}. likeness:{}'.format(rank + 1, self.interests[item_idx])
            cv2.namedWindow(win_name)
            cv2.moveWindow(win_name, 80 + rank * 100, 60 + rank * 100)
            cv2.imshow(win_name, self.items[item_idx][1])
            cv2.waitKey(0)
        cv2.destroyAllWindows()


# parse arg to see if we need to launch training now or not yet
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="Image files path to load")
args = parser.parse_args()

if args.path:
    if os.path.isdir(args.path):
        r = EmotionRecognizer()
        thread.start_new_thread(r.display, ('thread-display', VIDEO_PREDICTOR.time_to_wait_between_display_shop, VIDEO_PREDICTOR.title_change_interval_shop, args.path))
    else:
        print "Error: path '{}' not found".format(args.path)