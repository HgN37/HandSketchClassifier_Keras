from keras.models import Sequential, Model
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D, Activation, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import Adam
import numpy as np

img_width, img_height = 128, 128
img_size = (img_width, img_height, 1)
batch_size = 50

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

predict_img = ImageDataGenerator();
predict_gen = predict_img.flow_from_directory(
    './test/',
    target_size=(img_width,img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='sparse'
)
x = model.predict_generator(predict_gen, steps=1)
print(x)
