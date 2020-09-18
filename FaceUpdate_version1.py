from statistics import mode

import cv2
import tensorflow as tf
from keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt

from model.decode_np import Decode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input


import RecordAudio
import SpeechToWord

# 6G的卡，训练时如果要预测，则设置use_gpu = False，否则显存不足。

use_gpu = False
use_gpu = True

 #显存分配。
if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
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

record = {'angry':[0], 'disgust':[0], 'fear':[0], 'happy':[0], 'sad':[0], 'surprise':[0], 'neutral':[0]}
#record_diff = {'angry':[], 'disgust':[], 'fear':[], 'happy':[], 'sad':[], 'surprise':[], 'neutral':[]}'''
emo_record = []

#自創權重
def weights_change(record,Probability):
    weights={}
    for idx, emotion in enumerate(record):
        try:
            weights[emotion] = weights[emotion] +(idx+1)*0.02
        except:
            weights[emotion] = 1+(idx+1)*0.05
    for emo, weight in weights.items():
        Probability[emo] = Probability[emo]*weight
    #print(Probability)
    return Probability
            
# starting video streaming

rd1=[]
rd2=[]

cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
count=int(0)
s=count


while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_image(bgr_image)[1]
    if faces is None:
        faces = ()
    # print(faces)

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
        ###############移動平均公式################
        for idx, probability in enumerate(emotion_prediction[0]):
            alpha = 0.5
            record[emotion_labels[idx]].append(record[emotion_labels[idx]][-1] + alpha * (round(probability*100, 2)-record[emotion_labels[idx]][-1]))
            emotion_prediction[0][idx] = record[emotion_labels[idx]][-1]
            if len(record[emotion_labels[idx]])>10:
                record[emotion_labels[idx]].pop(0)
            print(record)
        print()
        #########################################
        """
        #################自創權重##############
        emotion_prediction[0] = weights_change(emo_record, emotion_prediction[0])
        data=[]
        for idx, probability in enumerate(emotion_prediction[0]):
            data.append((emotion_labels[idx],  probability))
        rd.append(data)
        count+=1
        if count!=0 and count%5==0:
            for i in range(s,count):
                for j in rd[i]:
                    print(j[0],j[1])
                print()
            s=count
        emo_record.append(np.argmax(emotion_prediction))
        if len(emo_record)>10:
            emo_record.pop(0)
        #######################################
            """
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

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
video_capture.release()
cv2.destroyAllWindows()
