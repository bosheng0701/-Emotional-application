from statistics import mode

import cv2
import tensorflow as tf
from keras.models import load_model
import numpy as np
import os
import threading
import time

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
import predict

# 6G的卡，训练时如果要预测，则设置use_gpu = False，否则显存不足。
use_gpu = True

# 显存分配。
if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# parameters for loading data and images


class face_detect():
    def __init__(self):
        # hyper-parameters for bounding boxes shape
        self.frame_window = 10
        self.emotion_offsets = (20, 40)

        # loading models
        self.face_detector = Decode('data/voc_classes.txt', './weights/best_model.h5')
        self.emotion_classifier = load_model('model/fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
        self.emotion_labels = get_labels('fer2013')

        # getting input model shapes for inference
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]

        # starting lists for calculating modes
        self.emotion_window = []

        self.probability = {'angry':0, 'disgust':0, 'fear':0, 'happy':0, 'sad':0, 'surprise':0, 'neutral':0}

    def faceDetect(self, bgr_image):
        faces = self.face_detector.detect_image(bgr_image)[1]
        probability = [0,0,0,0,0,0,0]
        if faces is not None and len(faces) == 1:
            bgr_image = self.emo_Detect(bgr_image, faces[0])
        return bgr_image,  self.probability

    def emo_Detect(self, bgr_image, face_coordinates):
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        x1, y1, x2, y2 = face_coordinates
        face_coordinates = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        x1, x2, y1, y2 = apply_offsets(face_coordinates, self.emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (self.emotion_target_size))
        except:
            return bgr_image
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = self.emotion_classifier.predict(gray_face)
        for idx, probability in enumerate(emotion_prediction[0]):
            self.probability[self.emotion_labels[idx]] = probability
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = self.emotion_labels[emotion_label_arg]
        self.emotion_window.append(emotion_text)

        if len(self.emotion_window) > self.frame_window:
            self.emotion_window.pop(0)
        # try:
        #     emotion_mode = mode(self.emotion_window)
        #     print("mode is "+emotion_mode)
        # except:
        #     pass

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((0, 255, 255))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((255, 255, 0))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, bgr_image, color)
        draw_text(face_coordinates, bgr_image, emotion_text,
                  color, 0, -45, 1, 1)

        return bgr_image


class videoDetect():
    def __init__(self):
        self.detection = face_detect()
        self.record1 = {'angry':[], 'disgust':[], 'fear':[], 'happy':[], 'sad':[], 'surprise':[], 'neutral':[]}
        self.record1_30s = []
        self.record2 = {'angry':[], 'disgust':[], 'fear':[], 'happy':[], 'sad':[], 'surprise':[], 'neutral':[]}
        self.record2_30s = []
        self.end=False
        self.frame1=None
        self.frame2=None
        self.showData=False
        self.a=int(0)
        self.b=int(0)
        
    def second(self):
        count=int(0)
        while True:
            if self.a==0:
                continue
            time.sleep(1)
            self.record1_30s.append(self.maxEmo(self.record1))
            self.record1 = {'angry':[], 'disgust':[], 'fear':[], 'happy':[], 'sad':[], 'surprise':[], 'neutral':[]}
            self.record2_30s.append(self.maxEmo(self.record2))
            self.record2 = {'angry':[], 'disgust':[], 'fear':[], 'happy':[], 'sad':[], 'surprise':[], 'neutral':[]}
            count=count+1
            if self.showData:
                f = open('AudioClassification.txt','r')
                print(f.read())
                f.close()
                print(self.record1_30s)
                print(self.record2_30s)
                self.record1_30s = []
                self.record2_30s = []
                self.showData=False
            if self.end:
                break
    def maxEmo(self, record):
        max = 0
        max_key = ""
        for key in record.keys():
            summary = sum(record[key])
            if summary > max:
                max = summary
                max_key = key
        return max_key
    def getFrame1(self):
        video_capture = cv2.VideoCapture(0)
        while(1):
            self.frame1 = video_capture.read()[1]
            self.a=self.a + 1
            if self.end:
                video_capture.release()
                break
    def getFrame2(self):
        video_capture = cv2.VideoCapture(1)
        while(1):
            self.frame2 = video_capture.read()[1]
            self.b=self.b + 1
            if self.end:    
                video_capture.release()
                break
    def predictword(self,fileName):
        os.system("py -3.8 AudioClassification.py "+fileName)
        #AudioClassification.classification(fileName)
        word=SpeechToWord.audiotoword(fileName)
        predict.predictEmotion(word)
        os.remove(fileName)
        self.showData=True
    def speech(self):
        Num=int(0)
        while True:
            fileName=str(Num)+".wav"
            print(fileName)
            print("please speak a word into the microphone")
            RecordAudio.record_to_file(fileName)
            print("done - result written to "+fileName)
            Job=threading.Thread(target=self.predictword,args=(fileName,))
            Job.setDaemon(True)
            Job.start()
            Num=Num+1
            if self.end:
                break
            
    def video_start(self):
        
        flag=True
        Job1 = threading.Thread(target=self.getFrame1)
        Job1.setDaemon(True)
        Job1.start()
        
        Job2 = threading.Thread(target=self.getFrame2)
        Job2.setDaemon(True)
        Job2.start()
        
        
        
        
        while True:
            if(self.a==0):
                continue
            if flag:
                Job4 = threading.Thread(target=self.second)
                Job4.setDaemon(True)
                Job4.start()
                Job3 = threading.Thread(target=self.speech)
                Job3.setDaemon(True)
                Job3.start()
                flag=False
            image = self.frame1
            bgr_image, probability = self.detection.faceDetect(image.copy())
            cv2.imshow('window_frame1', bgr_image)
            for key in self.record1.keys():
                self.record1[key].append(probability[key])
            
            image = self.frame2
            bgr_image, probability = self.detection.faceDetect(image.copy())
            cv2.imshow('window_frame2', bgr_image)
            for key in self.record2.keys():
                self.record2[key].append(probability[key])
            

            if cv2.waitKey(1) & 0xFF == 27:
                self.end=True
                break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    a = videoDetect()
    a.video_start()

    # detection = face_detect()
    # cv2.namedWindow('window_frame')
    # video_capture = cv2.VideoCapture(0)
    # while True:
    #     image = video_capture.read()[1]
    #     bgr_image, probability = detection.faceDetect(image)
    #     print(probability)
    #     print()
    #     cv2.imshow('window_frame', bgr_image)
    #     if cv2.waitKey(35) & 0xFF == 27:
    #         break
    # video_capture.release()
    # cv2.destroyAllWindows()

