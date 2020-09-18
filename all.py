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

import threading
import time

import RecordAudio
import SpeechToWord

# 6G的卡，训练时如果要预测，则设置use_gpu = False，否则显存不足。


# parameters for loading data and images


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
count1=int(0)
count2=int(0)
frame=None
a = int(0)



        
def job1():
    global count1
    global frame    
    
    use_gpu = True
        # 显存分配。
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))

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
    emotion_window1 = []
    record1 = {'angry':[0], 'disgust':[0], 'fear':[0], 'happy':[0], 'sad':[0], 'surprise':[0], 'neutral':[0]}
    #record_diff = {'angry':[], 'disgust':[], 'fear':[], 'happy':[], 'sad':[], 'surprise':[], 'neutral':[]}'''
    emo_record1 = []
    capture = cv2.VideoCapture(0)
    while True:
        
        if True:
            bgr_image = capture.read()[1]
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
                '''for idx, probability in enumerate(emotion_prediction[0]):
                    alpha = 0.5
                    record[emotion_labels[idx]].append(record[emotion_labels[idx]][-1] + alpha * (round(probability*100, 2)-record[emotion_labels[idx]][-1]))
                    emotion_prediction[0][idx] = record[emotion_labels[idx]][-1]
                    if len(record[emotion_labels[idx]])>10:
                        record[emotion_labels[idx]].pop(0)
                    #print(record)
                #print()'''
                #########################################

                
                #################自創權重##############
                emotion_prediction[0] = weights_change(emo_record1, emotion_prediction[0])
                data=[]
                for idx, probability in enumerate(emotion_prediction[0]):
                    data.append((emotion_labels[idx],  probability))
                rd1.append(data)
                count1+=1
                emo_record1.append(np.argmax(emotion_prediction))
                if len(emo_record1)>10:
                    emo_record1.pop(0)
                #######################################
                    
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label_arg]
                emotion_window1.append(emotion_text)
                
            
            
                
                if len(emotion_window1) > frame_window:
                    emotion_window1.pop(0)
                try:
                    emotion_mode1 = mode(emotion_window1)
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
                draw_text(face_coordinates, rgb_image, emotion_mode1,
                        color, 0, -45, 1, 1)

            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('window_frame1', bgr_image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cv2.destroyAllWindows()

def job2():
    global count2
    use_gpu = True
        # 显存分配。
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))

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
    emotion_window2 = []
    record2 = {'angry':[0], 'disgust':[0], 'fear':[0], 'happy':[0], 'sad':[0], 'surprise':[0], 'neutral':[0]}
    #record_diff = {'angry':[], 'disgust':[], 'fear':[], 'happy':[], 'sad':[], 'surprise':[], 'neutral':[]}'''
    emo_record2 = []
    capture = cv2.VideoCapture(1)
    while True:
        bgr_image = capture.read()[1]
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
            '''for idx, probability in enumerate(emotion_prediction[0]):
                alpha = 0.5
                record[emotion_labels[idx]].append(record[emotion_labels[idx]][-1] + alpha * (round(probability*100, 2)-record[emotion_labels[idx]][-1]))
                emotion_prediction[0][idx] = record[emotion_labels[idx]][-1]
                if len(record[emotion_labels[idx]])>10:
                    record[emotion_labels[idx]].pop(0)
                #print(record)
            #print()'''
            #########################################
            
            #################自創權重##############
            emotion_prediction[0] = weights_change(emo_record2, emotion_prediction[0])
            data=[]
            for idx, probability in enumerate(emotion_prediction[0]):
                data.append((emotion_labels[idx],  probability))
            rd2.append(data)
            count2+=1
            emo_record2.append(np.argmax(emotion_prediction))
            if len(emo_record2)>10:
                emo_record2.pop(0)
            #######################################
                
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window2.append(emotion_text)
            
           
          
            
            if len(emotion_window2) > frame_window:
                emotion_window2.pop(0)
            try:
                emotion_mode2 = mode(emotion_window2)
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
            draw_text(face_coordinates, rgb_image, emotion_mode2,
                      color, 0, -45, 1, 1)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame2', bgr_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
def job3():
    SpeechToWord.audiotoword('demo.wav')
s1=int(0)
s2=int(0)
#t4 = threading.Thread(target = job4)
#t4.start()

t1 = threading.Thread(target = job1)
t1.setDaemon(True) 
t1.start()
t2 = threading.Thread(target = job2)
t2.setDaemon(True) 
t2.start()

#flag=True
"""
while 1:
    if a>0 and flag:    
        t1 = threading.Thread(target = job1)
       # t2 = threading.Thread(target = job2)
      #  t2.daemon = True
        t1.start()
       # t2.start()
        flag=False
"""
"""
    if flag==False:            
        print("please speak a word into the microphone")
        RecordAudio.record_to_file('demo.wav')
        print("done - result written to demo.wav")
            
        t3 = threading.Thread(target = job3)
        t3.start()

    if count1 or count2:
        a=count1
        b=count2
        for i in range(s1,a):
            for j in rd1[i]:
                print(j[0],j[1])
            print()
        s1=a
        for i in range(s2,b):
            for j in rd2[i]:
                print(j[0],j[1])
            print()
        s2=b
    t3.join()
"""
t1.join()
#t2.join()
#t4.join()
