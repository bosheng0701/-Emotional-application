from pyAudioAnalysis import audioTrainTest as aT
import numpy as np
import os
'''
subdirectories = ['D:\Audio_Speech_Actors_01-24\Angry',
                  'D:\Audio_Speech_Actors_01-24\Calm',
                  'D:\Audio_Speech_Actors_01-24\Disgust',
                  'D:\Audio_Speech_Actors_01-24\Fearful',
                  'D:\Audio_Speech_Actors_01-24\Happy',
                  'D:\Audio_Speech_Actors_01-24\\Neutral',
                  'D:\Audio_Speech_Actors_01-24\Sad',
                  'D:\Audio_Speech_Actors_01-24\Surprised']
aT.extract_features_and_train(subdirectories,
                              1.0, 1.0,
                              aT.shortTermWindow,
                              aT.shortTermStep,
                              "svm", "svmModel", False)

aT.extract_features_and_train(subdirectories,
                              1.0, 1.0,
                              aT.shortTermWindow,
                              aT.shortTermStep,
                              "knn", "knnModel", False)
aT.extract_features_and_train(subdirectories,
                              1.0, 1.0,
                              aT.shortTermWindow,
                              aT.shortTermStep,
                              "randomforest", "randomforestModel", False)
'''                         

Result, P, classNames = aT.file_classification("D:\\Audio_Speech_Actors_01-24\\Fearful\\03-01-06-01-01-01-01.wav", "svmModel","svm")
maxx = np.argmax(P)
print(Result)
print(P)
print(classNames)
print(classNames[maxx], P[maxx])

'''
subdirectories = os.listdir('D:\MER_bimodal_dataset\Test')
subdirectories = ['D:\MER_bimodal_dataset\Test' + "\\" + subDirName for subDirName in subdirectories]
#print(subdirectories)

ans = ['SereneJoy', 'Happy', 'SereneJoy', 'Melancholy', 'Tense',
       'SereneJoy', 'Happy', 'Happy', 'Tense', 'Happy',
       'Melancholy', 'SereneJoys', 'Melancholy', 'Melancholy', 'Happy',
       'Happy', 'Tense', 'Happy', 'Tense', 'Happy',
       'Happy', 'Happy', 'Melancholy', 'Tense', 'Happy',
       'Tense', 'SereneJoy', 'Happy', 'Tense', 'SereneJoy', 'Happy', 'Tense']

svm_ac = knn_ac = rf_ac = 0
print('Answer SVM KNN RF')

for i in range(32):
    print(ans[i], end=' ')
    Result, P, classNames = aT.file_classification(subdirectories[i], "svmModel","svm")
    maxx = np.argmax(P)
    if classNames[maxx] == ans[i]: svm_ac+=1
    print(classNames[maxx], end=' ')

    Result, P, classNames = aT.file_classification(subdirectories[i], "knnModel","knn")
    maxx = np.argmax(P)
    if classNames[maxx] == ans[i]: knn_ac+=1
    print(classNames[maxx], end=' ')

    Result, P, classNames = aT.file_classification(subdirectories[i], "randomforestModel","randomforest")
    maxx = np.argmax(P)
    if classNames[maxx] == ans[i]: rf_ac+=1
    print(classNames[maxx])

print('svm:', svm_ac/32, 'knn:', knn_ac/32, 'rf:', rf_ac/32)
'''
