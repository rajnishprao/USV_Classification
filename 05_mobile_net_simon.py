#!/usr/bin/env python
# yayy this works for my data from folders, with DeepSqueak architechture


import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import regularizers
from keras.applications import MobileNet


# data generator for training set
train_datagen = ImageDataGenerator(rescale = 1./255)
    # shear_range = 0.2, # random application of shearing
    # zoom_range = 0.2,
    # horizontal_flip = True) # randomly flipping half of the images horizontally

# data generator for validation set
validation_datagen = ImageDataGenerator(rescale = 1./255)

# data generator for test set
#test_datagen = ImageDataGenerator(rescale = 1./255)

# generator for reading train data from folder
train_ds = train_datagen.flow_from_directory(
    './training_data',
    target_size = (334, 217),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode = 'categorical')

# generator for reading validation data from folder
validation_ds = validation_datagen.flow_from_directory(
    './validation_data',
    target_size = (334, 217),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode = 'categorical')

# # generator for reading test data from folder
# test_generator = test_datagen.flow_from_directory(
#     'data/test',
#     target_size = (256, 256),
#     color_mode = 'rgb',
#     batch_size = 1,
#     class_mode = 'binary',
#     shuffle = False)

K.clear_session()
callback = EarlyStopping(monitor='val_loss', mode ='min', patience=5)

model = Sequential()
model.add(MobileNet(include_top=False, pooling='max', input_shape=(334, 217, 3)))
model.add(Dense(3, activation='softmax'))
model.layers[0].trainable = False

model.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit_generator(train_ds, epochs=30, steps_per_epoch=32,
    validation_data=validation_ds, validation_steps=32, callbacks=[callback],
    class_weight={0:1.0, 1:1.4, 2:1.1})

# plot loss
loss_train = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(0,len(loss_train))
plt.plot(epochs, loss_train, 'g', label='Training Loss')
plt.plot(epochs, val_loss, 'k', label='Validation Loss')
plt.title('Loss: Training and Validation')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# # plot accuracy
acc_train = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(0,len(acc_train))
plt.plot(epochs, acc_train, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy: Training and Validation')
plt.xlabel('Epochs')
plt.ylim(0, 1)
plt.legend()
plt.show()
