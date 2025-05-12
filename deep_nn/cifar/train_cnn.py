
import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import os

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices:", tf.config.list_physical_devices())

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

NUM_CLASSES = 10

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = Sequential([
   Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3), strides=1),
   BatchNormalization(),
   LeakyReLU(alpha=0.1),
   Conv2D(32, (3,3), activation='relu', padding='same', strides=2),
   BatchNormalization(),
   LeakyReLU(alpha=0.1),
   Conv2D(64, (3,3), activation='relu', padding='same', strides=1),
   BatchNormalization(),
   LeakyReLU(alpha=0.1),
   Conv2D(64, (3,3), activation='relu', padding='same', strides=2),
   BatchNormalization(),
   LeakyReLU(alpha=0.1),
   Flatten(),
   Dense(128, activation='relu'),
   BatchNormalization(),
   LeakyReLU(alpha=0.1),
   Dropout(0.2),
   Dense(NUM_CLASSES, activation='softmax'),
])
model.summary()

optimizer = Adam(learning_rate=0.0005)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_cb = ModelCheckpoint(
    "checkpoints_cnn/cifar_model_best.h5",
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

os.makedirs("checkpoints_cnn", exist_ok=True)

history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    shuffle=True,
    callbacks=[checkpoint_cb]
)
model.evaluate(x_test, y_test)

model.save('cifar_model_cnn.h5')
