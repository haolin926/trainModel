import os
import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow import device

IMG_FILE_PATH = "/home/hao/dataset/food101/images/"
META_PATH = "/home/hao/dataset/food101/meta/meta/"

# load data
# read in classes.txt
classes = pd.read_csv(META_PATH+'classes.txt', header=None)
# create a dictionary to map class names to numbers
class_to_norminal = dict(zip(classes[0].values, range(classes.shape[0])))

# read in train.txt into column called txt
train_df = pd.read_csv(META_PATH+'train.txt', names=['txt'], header=None)
# create a column called img that is the txt column with .jpg appended
# this is the file path to the images
train_df['img'] = train_df['txt'].apply(lambda x : x+'.jpg')
# create a column called label that is the txt column split by / and mapped to the class_to_norminal dictionary
# this gives the index of the coressponding class in the class_to_nominal dictionary
train_df['label'] = train_df['txt'].apply(lambda x: class_to_norminal[x.split('/')[0]])

# drop the txt column
train_df.drop(['txt'], axis=1, inplace=True)
# shuffle the data
train_df = train_df.sample(frac=1)

valid_df = pd.read_csv(META_PATH+'test.txt', names=['txt'], header=None)
valid_df['img'] = valid_df['txt'].apply(lambda x: x+'.jpg')
valid_df['label'] = valid_df['txt'].apply(lambda x: class_to_norminal[x.split('/')[0]])

valid_df.drop(['txt'], axis=1, inplace=True)
valid_df = valid_df.sample(frac=1)

# Create ImageDataGenerator instances to preprocess the images
train_dg = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        shear_range=0.3,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.25,
)

# validation set is scaled by dividing the pixel values by 255
valid_dg = ImageDataGenerator(
        rescale=1./255,
)

# define the batch size and image size
BATCH_SIZE = 64
IMAGE_SIZE = 224

# create data generators from the dataframes
train_data = train_dg.flow_from_dataframe(
    dataframe=train_df,
    directory=IMG_FILE_PATH,
    x_col="img",
    y_col="label",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=True,
)
valid_data = valid_dg.flow_from_dataframe(
    dataframe=valid_df,
    directory=IMG_FILE_PATH,
    x_col="img",
    y_col="label",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=True,
)

def train():
    # Load MobileNetV3 with ImageNet weights, without the top layer
    pre_trained = MobileNetV3Large(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),include_top=False,weights='imagenet')
    
    # Freeze the pre-trained model weights
    pre_trained.trainable = True

    # Define the model
    model = keras.Sequential([
    pre_trained,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(101, activation='softmax'),
    ])

    # Compile the model
    # uses the Adam optimizer for training
    # uses sparse_categorical_crossentropy as the loss function, wchich is suitable for multi-class classification with integer labels
    # track the accuracy as the performance metric during training and evaluation
    model.compile(
        optimizer='adam',
        loss = 'sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Define callbacks
    # reduce the learning rate by a factor of 0.1 if the validation accuracy does not improve for 1 epoch
    reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy',patience = 1,verbose = 1)
    # stop training if the validation accuracy does not improve for 5 epochs
    early_stop = EarlyStopping(monitor = 'val_accuracy',patience = 5,verbose = 1,restore_best_weights = True)
    # save the model with the best validation accuracy
    chkp = ModelCheckpoint('mobilenet_v3_large_checkpoint.keras',monitor='val_accuracy',verbose=1,save_best_only=True)

    # Train the model
    with device('/GPU:0'):
        history = model.fit(
        train_data,
        validation_data = valid_data,
        
        epochs = 20,
        callbacks=[early_stop, reduce_lr, chkp],
    )
    
    model.save('best_model.keras')
    with open('hisory.json', 'w') as f:
        json.dump(history.history, f)
    
train()
