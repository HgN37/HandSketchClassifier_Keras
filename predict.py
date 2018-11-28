from keras.models import Sequential, Model
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D, Activation, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import Adam
import numpy as np
import os
import time

img_width, img_height = 128, 128
img_size = (img_width, img_height, 1)
batch_size = 50

print('Importing model...')

model = Sequential()

model.add( Conv2D(16, (3, 3), input_shape=img_size) )
model.add( Activation('relu') )
model.add( MaxPooling2D(pool_size=(2, 2)) )

model.add( Conv2D(32, (3, 3)) )
model.add( Activation('relu') )
model.add( MaxPooling2D(pool_size=(2, 2)) )

model.add( Conv2D(32, (3, 3)) )
model.add( Activation('relu') )
model.add( MaxPooling2D(pool_size=(2, 2)) )

model.add( Conv2D(64, (3, 3)) )
model.add( Activation('relu') )
model.add( MaxPooling2D(pool_size=(2, 2)) )

model.add( Dropout(0.2) )
model.add( Flatten() )
model.add( Dense(64) )
model.add( Activation('relu') )
model.add( Dropout(0.2) )
model.add( Dense(11) )
model.add( Activation('softmax') )

model.load_weights('./model/model_v10.h5')

print('Model imported')

print('Importing class list')

with open('./model/class_list_v10.txt') as f:
    class_list = f.readlines()
class_list = [x.strip() for x in class_list]

print('Class list imported')

predict_img = ImageDataGenerator();
while True:
    input('Press to test')
    os.system('clear')
    print(class_list)
    predict_gen = predict_img.flow_from_directory(
        './test/',
        target_size=(img_width,img_height),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='sparse'
    )
    output_raw = model.predict_generator(predict_gen, steps=1, verbose=0)
    onehot = output_raw.tolist()
    print()
    print(class_list[onehot[0].index(1)])
    print()
