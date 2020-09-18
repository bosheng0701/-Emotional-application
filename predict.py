import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


# 匯入字典
with open('word_dict.pk', 'rb') as f:
    word_dictionary = pickle.load(f)
with open('label_dict.pk', 'rb') as f:
    output_dictionary = pickle.load(f)
def predictEmotion(text):    
    try:
        input_shape = 180
        sent = ''.join(text)
        x = [[word_dictionary[word] for word in sent]]
        x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)

        model_save_path = './sentiment_analysis.h5'
        lstm_model = load_model(model_save_path)

        # 模型預測
        y_predict = lstm_model.predict(x)
        label_dict = {v:k for k,v in output_dictionary.items()}
        print('輸入語句: %s' % sent)
        print('情感預測結果: %s' % label_dict[np.argmax(y_predict)])

    except KeyError as err:
        print('輸入語句: %s' % sent)
        print('情感預測結果: 中性')
