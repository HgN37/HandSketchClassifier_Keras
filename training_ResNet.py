from keras.models import Sequential, Model
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D, Activation, Reshape, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import Adam
import numpy as np

img_width, img_height = 128, 128
img_size = (img_width, img_height, 1)
batch_size = 50

train_data_dir = './data/train_mini'
val_data_dir = './data/validation'

imgGen_trainging = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
imgGen_validation = ImageDataGenerator(rescale=1. / 255)
dataTrain = imgGen_trainging.flow_from_directory(
    train_data_dir,
    target_size=(img_width,img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='sparse'
)

dataVal = imgGen_validation.flow_from_directory(
    val_data_dir,
    target_size=(img_width,img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='sparse'
)

model = Sequential()

model.add( Conv2D(16, (3, 3), input_shape=img_size) )
model.add( Activation('relu') )
model.add( MaxPooling2D(pool_size=(2, 2)) )

model.add( BatchNormalization() )
model.add( Conv2D(32, (3, 3)) )
model.add( Activation('relu') )
model.add( MaxPooling2D(pool_size=(2, 2)) )

model.add( BatchNormalization() )
model.add( Conv2D(32, (3, 3)) )
model.add( Activation('relu') )
model.add( MaxPooling2D(pool_size=(2, 2)) )

model.add( BatchNormalization() )
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

#im = np.random.rand(1, 128, 128, 1)
#x = model.predict(im)
#print(x)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit_generator(
    dataTrain,
    steps_per_epoch=150,
    epochs=15,
    validation_data=dataVal,
    validation_steps=100,
)

model.save('model_v10_res.h5')
