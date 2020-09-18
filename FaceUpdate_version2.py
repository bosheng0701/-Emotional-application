from statistics import mode

import cv2
import tensorflow as tf
from keras.models import load_model
import numpy as np
import os

from model.decode_np import Decode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import threading
import time


#語音轉文字

from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave

from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import db_to_float



frame1 = 0
frame2 = 0
a=int(0)
b=int(0)

use_gpu = True

  # 显存分配。
if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

  # parameters for loading data and images
facesDetect = {'classes_path': 'data/voc_classes.txt', 'model_path': './weights/best_model.h5'}
emotion_model_path = 'model/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

  # hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

  # loading models
face_detector = Decode(facesDetect['classes_path'], facesDetect['model_path'])
emotion_classifier = load_model(emotion_model_path, compile=False)

  # getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

  # starting lists for calculating modes
emotion_window = []
def job1():
  global frame1
  global a
  capture = cv2.VideoCapture(0)
  if capture.isOpened():
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
      read_code, frame1 = capture.read()
      cv2.imshow("screen_title1", frame1)
      a=a+1
     # print("a",(frame1.shape))
      if cv2.waitKey(1) == ord('q'):
        capture.release()
        cv2.destroyWindow("screen_title1")
    
def job2():
  global frame2
  global b
  capture = cv2.VideoCapture(1)
  if capture.isOpened():
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
      read_code, frame2 = capture.read()
      cv2.imshow("screen_title2", frame2)
      b=b+1
     # print("a",(frame1.shape))
      if cv2.waitKey(1) == ord('q'):
        capture.release()
        cv2.destroyWindow("screen_title2")
    

t1 = threading.Thread(target = job1)
#t2 = threading.Thread(target = job2)
t1.start()
#t2.start()
while True:
    if a>0 :
        
        bgr_image = frame1
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = face_detector.detect_image(bgr_image)[1]
        
        if faces is None:
            faces = ()
        for face_coordinates in faces:
            x1, y1, x2, y2 = face_coordinates
            face_coordinates = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            for idx, probability in enumerate(emotion_prediction[0]):
                print(emotion_labels[idx], probability)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()
            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_mode,
                    color, 0, -45, 1, 1)
        
        bgr_image1 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame1', bgr_image1)


        bgr_image = frame1
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = face_detector.detect_image(bgr_image)[1]
        
        if faces is None:
            faces = ()
        for face_coordinates in faces:
            x1, y1, x2, y2 = face_coordinates
            face_coordinates = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            for idx, probability in enumerate(emotion_prediction[0]):
                print(emotion_labels[idx], probability)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()
            
            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_mode,
                    color, 0, -45, 1, 1)
        
        bgr_image2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame2', bgr_image2)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cv2.destroyAllWindows()
t1.join()
#t2.join()

