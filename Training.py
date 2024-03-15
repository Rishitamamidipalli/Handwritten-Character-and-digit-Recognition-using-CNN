import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D,Activation,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization
import pickle

x_train=np.load("train_x.npy")
y_train=np.load("train_y.npy")
x_test=np.load("test_x.npy")
y_test=np.load("test_y.npy")
print(x_train.shape,y_train.shape)
model=Sequential()
model.add(Conv2D(32,(5,5),input_shape=(28,28,1),activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(32,(5,5),activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(5,5))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=36,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)
loss,accuracy=model.evaluate(x_test,y_test)

pickle.dump(model,open('CNN_digit_mode.pkl','wb'))