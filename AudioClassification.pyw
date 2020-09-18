from pyAudioAnalysis import audioTrainTest as aT
import numpy as np
import os
import sys
def classification(): #"knn", "knnModel" / "randomforest", "randomforestModel"
    # 丟入訓練模型得到結果
    path=sys.argv[1]
    print(path)
    Result, P, classNames = aT.file_classification(path, "svmModel","svm")

    # 獲取最大值的index
    maxx = np.argmax(P)
    f = open('AudioClassification.txt','w')
    # 印出結果
    f.write("%s\n"%(Result))
    f.write("%s\n"%(P))
    f.write("%s\n"%(classNames))
    f.write("%s %s\n"%(classNames[maxx], P[maxx]))
    f.close()
classification()