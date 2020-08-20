# trying out DeepSqueak architechture 
# seems to work ok

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.preprocessing import image_dataset_from_directory

train_ds = image_dataset_from_directory(
    directory='training_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(334, 217))
validation_ds = image_dataset_from_directory(
    directory='validation_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(334, 217))

model = Sequential([
    Conv2D(filters=8, kernel_size=(5,3),strides=(2,2),
        padding='same',input_shape=(334, 217, 3)),
    BatchNormalization(),
    Conv2D(filters=8, kernel_size=(5,5),strides=(1,1),
        padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(3,2), strides=(2,1)),
    Conv2D(filters=16, kernel_size=(5,5), strides=(2,2),
        padding='same'),
    BatchNormalization(),
    Conv2D(filters=20, kernel_size=(5,5),strides=(2,2),
        padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2,2), strides=(2,2)),
    Conv2D(filters=32, kernel_size=(5,5),strides=(2,2),
        padding='same'),
    BatchNormalization(),
    Conv2D(filters=40, kernel_size=(5,5), strides=(2,2),
        padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Flatten(),
    Dense(64),
    BatchNormalization(),
    Activation('relu'),
    Dense(3),
    Activation('softmax')
])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_ds, epochs=5, validation_data=validation_ds)

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
acc_train = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(0,len(acc_train))
plt.plot(epochs, acc_train, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy: Training and Validation')
plt.xlabel('Epochs')
plt.ylim(0, 1)
plt.legend()
plt.show()
