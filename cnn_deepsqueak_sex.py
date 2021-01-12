"""
Automated Classification of Rat Ultrasonic Vocalizations (USVs):
binary categorization of female and male calls
"""


import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPool2D)
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

# Activate PlaidML backend
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


# Data Generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_ds = train_datagen.flow_from_directory(
    "./training_data",
    target_size=(334, 217),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
)

validation_ds = validation_datagen.flow_from_directory(
    "./validation_data",
    target_size=(334, 217),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
)

test_ds = test_datagen.flow_from_directory(
    "./test_data",
    target_size=(334, 217),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
)

# Callbacks
K.clear_session()
callback = EarlyStopping(monitor="val_loss", mode="min", patience=10)

# Convolutional Neuronal Network
# based on DeepSqueak CNN (Coffey et al, 2019)
model = Sequential(
    [
        Conv2D(
            filters=8,
            kernel_size=(5, 5),
            strides=(2, 2),
            kernel_initializer="he_normal",
            padding="same",
            input_shape=(334, 217, 3),
            kernel_regularizer=regularizers.l2(0.01),
        ),
        BatchNormalization(),
        Conv2D(
            filters=8,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="same",
            kernel_regularizer=regularizers.l2(0.01),
        ),
        BatchNormalization(),
        Activation("relu"),
        MaxPool2D(pool_size=(3, 2), strides=(2, 1)),
        Dropout(0.20),
        Conv2D(
            filters=16,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            kernel_regularizer=regularizers.l2(0.01),
        ),
        BatchNormalization(),
        Dropout(0.20),
        Conv2D(
            filters=20,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            kernel_regularizer=regularizers.l2(0.01),
        ),
        BatchNormalization(),
        Activation("relu"),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.20),
        Conv2D(
            filters=32,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            kernel_regularizer=regularizers.l2(0.01),
        ),
        BatchNormalization(),
        Dropout(0.20),
        Conv2D(
            filters=40,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            kernel_regularizer=regularizers.l2(0.01),
        ),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.20),
        Flatten(),
        Dense(64, kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(),
        Activation("relu"),
        Dense(2, kernel_regularizer=regularizers.l2(0.01)),
        Activation("sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit_generator(
    train_ds,
    epochs=50,
    steps_per_epoch=32,
    validation_data=validation_ds,
    validation_steps=32,
    callbacks=[callback],
    class_weight={0: 1.0, 1: 1.09},
)

# Plot loss
loss_train = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(0, len(loss_train))
plt.plot(epochs, loss_train, "g", label="Training Loss")
plt.plot(epochs, val_loss, "k", label="Validation Loss")
plt.title("Loss: Training and Validation")
plt.xlabel("Epochs")
plt.legend()
plt.savefig("./cnn_deepsqueak_figs/loss.png", dpi=300)
plt.show()

# # Plot accuracy
acc_train = history.history["acc"]
val_acc = history.history["val_acc"]
epochs = range(0, len(acc_train))
plt.plot(epochs, acc_train, "b", label="Training Accuracy")
plt.plot(epochs, val_acc, "r", label="Validation Accuracy")
plt.title("Accuracy: Training and Validation")
plt.xlabel("Epochs")
plt.ylim(0, 1)
plt.legend()
plt.savefig("./cnn_deepsqueak_figs/accuracy.png", dpi=300)
plt.show()

# Calculate test scores
test_loss, test_acc = model.evaluate_generator(test_ds, steps=50)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# Confusion Matrix
Y_pred = model.predict_generator(test_ds)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(test_ds.classes, y_pred)
print("Confusion Matrix - Test Data Set")
print(cm)
index = ["female", "male"]
columns = ["female", "male"]
cm_df = pd.DataFrame(cm, columns, index)
sns.heatmap(cm_df / np.sum(cm_df), annot=True, fmt=".2%", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("True Classes")
plt.ylabel("Predicted Classes")
plt.savefig("./cnn_deepsqueak_figs/confusion_matrix.png", dpi=300)
plt.show()

# Classification Report
print("Classification Report - Test Data Set")
print(classification_report(test_ds.classes, y_pred, target_names=columns))
