import os
import random
from pathlib import Path
import cv2
import pandas as pd
from keras.src.callbacks import ModelCheckpoint
from keras.src.layers import Conv2D, Activation, Dropout, Dense, MaxPooling2D, Flatten
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.models import Sequential
from matplotlib import pyplot as plt

ROOT_DIR = "../archive"
TRAIN_DIR = Path(f'{ROOT_DIR}/training/training/')
TEST_DIR = Path(f'{ROOT_DIR}/validation/validation/')
LABEL_FILE = f"{ROOT_DIR}/monkey_labels.txt"

IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 8
SEED = 100

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    featurewise_center=True,
    featurewise_std_normalization=True
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    class_mode='categorical'
)

train_generator = train_generator

test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    class_mode='categorical'
)

validation_generator = validation_generator

train_num = train_generator.samples
validation_num = validation_generator.samples

steps_per_epoch = None#int(train_num / train_generator.batch_size)
validation_steps = None#int(validation_num / validation_generator.batch_size)

def build_model(num_classes):
    model = Sequential([
        # Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        # Conv2D(64, (3, 3), strides=2),
        # Activation('relu'),
        # BatchNormalization(),
        # # Dropout(0.3),
        #
        # Conv2D(128, (3, 3), strides=2),
        # Activation('relu'),
        # BatchNormalization(),
        # # Dropout(0.3),
        #
        # Conv2D(256, (3, 3)),
        # Activation('relu'),
        # BatchNormalization(),
        # # Dropout(0.3),
        #
        # Conv2D(512, (3, 3), strides=2),
        # Activation('relu'),
        # BatchNormalization(),
        # # Dropout(0.3),
        #
        # Conv2D(512, (1, 1), strides=2),
        # Activation('relu'),
        #
        # Conv2D(num_classes, (1, 1)),
        # GlobalAveragePooling2D(),
        # Dense(128, activation='relu'),
        # Dropout(0.5),
        #
        # # Activation('softmax'),
        #
        # # Final output layer
        # Dense(num_classes, activation='softmax'),

        Conv2D(32, (3, 3), input_shape=(150, 150, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(32, (3, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), padding='same'),
        Activation('relu'),
        Conv2D(64, (3, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512),
        Activation('relu'),
        Dropout(0.5),
        Dense(num_classes),
        Activation('softmax')
    ])
    return model

num_classes = 10
model = build_model(num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

checkpoint = ModelCheckpoint("monkey.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

epochs = 100
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks_list,
    verbose=1
)

def visualize_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'r-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r-', label='Training Loss')
    plt.plot(epochs, val_loss, 'b-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

visualize_history(history)