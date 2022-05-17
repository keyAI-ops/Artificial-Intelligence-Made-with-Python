from re import M
from joblib import PrintTime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.preprocessing import image
import os

train_folder = 'C://Users/ChangKI/Desktop/파이썬으로 만드는 인공지능/CUB200/train'
test_folder = 'C://Users/ChangKI/Desktop/파이썬으로 만드는 인공지능/CUB200/test'

class_reduce = 0.1   # 데이터 양 줄이기
no_class = int(len(os.listdir(train_folder))*class_reduce)  # 부류 개수

x_train, y_train = [],[]
for i, class_name in enumerate(os.listdir(train_folder)):
    if i < no_class:
        for fname in os.listdir(train_folder + '/' + class_name):
            img = image.load_img(train_folder + '/' + class_name + '/' + fname,target_size=(224,224))
            if len(img.getbands()) != 3:   # getbands는 데이터에 존재하는 모든 다른 채널을 포함하는 튜플 반환 / RGB 이미지일 경우 ('R','G','B')
                print("주의 : 유효하지 않는 영상 발생", class_name, fname)
                continue
            x = image.img_to_array(img)  # PIL 형식의 이미지를 배열로 변경
            x=preprocess_input(x)   # 이미지를 모델이 요구하는 형식으로 변경
            x_train.append(x)
            y_train.append(i)

x_test, y_test = [],[]
for i, class_name in enumerate(os.listdir(test_folder)):
    if i < no_class:
        for fname in os.listdir(test_folder + '/' + class_name):
            img = image.load_img(test_folder + '/' + class_name + '/' + fname,target_size=(224,224))
            if len(img.getbands()) != 3:   # getbands는 데이터에 존재하는 모든 다른 채널을 포함하는 튜플 반환 / RGB 이미지일 경우 ('R','G','B')
                print("주의 : 유효하지 않는 영상 발생", class_name, fname)
                continue
            x = image.img_to_array(img)  # PIL 형식의 이미지를 배열로 변경
            x=preprocess_input(x)   # 이미지를 모델이 요구하는 형식으로 변경
            x_test.append(x)
            y_test.append(i)
            
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
y_train = tf.keras.utils.to_categorical(y_train, no_class)
y_test = tf.keras.utils.to_categorical(y_test, no_class)

 # include_top : 가장 상단의 완전연결층(FC)를 포함시킬지 여부, 여기선 ResNet50 - FC1000과 softmax 층으 떼어내라는 의미
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) 
cnn = Sequential()
cnn.add(base_model)
cnn.add(Flatten())
cnn.add(Dense(1024,activation='relu'))
cnn.add(Dense(no_class, activation='softmax'))

cnn.compile(loss='categorical_crossentropy', optimizer=Adam(0.0002), metrics=['accuracy'])
hist = cnn.fit(x_train, y_train, batch_size=8, epochs=10, validation_data=(x_test, y_test), verbose=1)

res = cnn.evaluate(x_test, y_test, verbose=0)
print("정확률은",res[1]*100)



