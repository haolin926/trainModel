import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow import keras

train_dir = r"D:\FYP_DATA\food101\train"
test_dir = r"D:\FYP_DATA\food101\test"
META_PATH = "D:/FYP_DATA/food101/meta/meta/"
classes = pd.read_csv(META_PATH+'classes.txt', header=None)
labels = pd.read_csv(META_PATH+'labels.txt', header=None)
# Create ImageDataGenerator instances
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    shear_range=0.3,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.25,
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load images using flow_from_directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)


def train():
    # Load MobileNetV3 with ImageNet weights, without the top layer
    base_model = MobileNetV3Large(weights='imagenet', include_top=False,  input_shape=(224, 224, 3))

    # Freeze the base model layers
    base_model.trainable = False
    
    model = keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(101, activation='softmax'),
])

    # Compile the model
    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )

    # Define callbacks
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_accuracy',patience = 1,verbose = 1)
    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_accuracy',patience = 5,verbose = 1,restore_best_weights = True)
    model_checkpoint = keras.callbacks.ModelCheckpoint('mobilenet_v3_large_checkpoint.keras',monitor='val_accuracy',verbose=1,save_best_only=True)

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=20,
        callbacks=[early_stop, reduce_lr, model_checkpoint]
    )
    
    model.save('best_model.keras')
    
def load_trained_model():
    model = load_model('best_model.keras')
    
    # For example, to evaluate on the validation set:
    loss, accuracy = model.evaluate(validation_generator)
    print(f'Validation accuracy: {accuracy:.4f}')
    print(f'Validation loss: {loss:.4f}')

def showData():
    x, y = next(train_generator)
    fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(15,15))

    for i in range(5):
        for j in range(5):
            ax[i][j].imshow(x[i+j*5])
            ax[i][j].set_title(labels[0][y[i+j*5]])
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
    
    fig.show()
    wait = input("PRESS ENTER TO CONTINUE.")
train()
