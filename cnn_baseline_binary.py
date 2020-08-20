#!/usr/bin/env python
# yayy this works for my data from folders, with DeepSqueak architechture


import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix

# data generator for training set
train_datagen = ImageDataGenerator(rescale = 1./255)
#    width_shift_range=0.2,
#   height_shift_range=0.2)
#    horizontal_flip=True,
#    vertical_flip=True,
#    rotation_range=5)
# all these data augmentation methods decidedly make things worse! gotta rethink this issue.

# data generator for validation set
validation_datagen = ImageDataGenerator(rescale = 1./255)

# data generator for test set
test_datagen = ImageDataGenerator(rescale = 1./255)

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

# generator for reading test data from folder
test_ds = test_datagen.flow_from_directory(
    './test_data',
    target_size = (334, 217),
    color_mode = 'rgb',
    batch_size = 1,
    class_mode = 'categorical',
    shuffle = False)

K.clear_session()
callback = EarlyStopping(monitor='val_loss', mode ='min', patience=5)

model = Sequential([
    # First convolutional layer
    Conv2D(filters=8, kernel_size=(3,3), kernel_initializer='he_normal', padding='same',
          input_shape=(334, 217, 3),kernel_regularizer=regularizers.l2(0.01)),
    MaxPool2D(pool_size=(2, 2)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.2),

    # Second convolutional layer
    Conv2D(filters=16, kernel_size=(3,3), padding='same',kernel_regularizer=regularizers.l2(0.01)),
    MaxPool2D(pool_size=(2, 2)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.2),

    # Second convolutional layer
    Conv2D(filters=32, kernel_size=(3,3), padding='same',kernel_regularizer=regularizers.l2(0.01)),
    MaxPool2D(pool_size=(2, 2)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.2),


    Flatten(),
    Dense(64,kernel_regularizer=regularizers.l2(0.01)),
    Dense(2,kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    Activation('sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit_generator(train_ds, epochs=30, steps_per_epoch=32,
    validation_data=validation_ds, validation_steps=32, callbacks=[callback],
    class_weight={0:1.0, 1:1.3})

# plot loss
loss_train = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(0,len(loss_train))
plt.plot(epochs, loss_train, 'g', label='Training Loss')
plt.plot(epochs, val_loss, 'k', label='Validation Loss')
plt.title('Loss: Training and Validation')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('./figs_cnn_baseline/loss_two_classes.jpg', dpi=300)
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
plt.savefig('./figs_cnn_baseline/accuracy_two_classes.jpg', dpi=300)
plt.show()

# calculate test scores
test_loss, test_acc = model.evaluate_generator(test_ds, steps=50)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

#Confusion Matrix and Classification Report - with test data set
Y_pred = model.predict_generator(test_ds)#, num_of_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(test_ds.classes, y_pred)
print('Confusion Matrix - Test Data Set')
print(cm)
index = ['flat', 'trill']
columns = ['flat', 'trill']
cm_df = pd.DataFrame(cm, columns, index)
sns.heatmap(cm_df/np.sum(cm_df), annot=True, fmt='.2%', cmap='Blues', cbar=False) #fmt=d gives no. of calls as integers
plt.title('Confusion Matrix')
plt.xlabel('True Classes')
plt.ylabel('Predicted Classes')
plt.savefig('./figs_cnn_baseline/cnn_confusion_matrix.jpg', dpi=300)
plt.show()
print('Classification Report - Test Data Set')
print(classification_report(test_ds.classes, y_pred, target_names=columns))
