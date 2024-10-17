import numpy as np
import time
import glob 
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.color import rgb2grey
import pandas as pd
import tensorflow as tf
from keras.models import model_from_json

#signs = ['Speed limit (20km/h)','Speed limit (30km/h)','Speed limit (50km/h)','Speed limit (60km/h)','Speed limit (70km/h)','Speed limit (80km/h)','End of speed limit (80km/h)','Speed limit (100km/h)','Speed limit (120km/h)','No passing','No passing for vechiles over 3.5 metric tons','Right-of-way at the next intersection','Priority road','Yield','Stop','No vechiles','Vechiles over 3.5 metric tons prohibited','No entry','General caution','Dangerous curve to the left','Dangerous curve to the right','Double curve','Bumpy road','Slippery road','Road narrows on the right','Road work','Traffic signals','Pedestrians','Children crossing','Bicycles crossing','Beware of ice/snow','Wild animals crossing','End of all speed and passing limits','Turn right ahead','Turn left ahead','Ahead only','Go straight or right','Go straight or left','Keep right','Keep left','Roundabout mandatory','End of no passing','End of no passing by vechiles over 3.5 metric tons']


images = []
image_labels  = []
roi = []

directory = 'train'
list_of_files = os.listdir(directory)
index = 0
for file in list_of_files:
    subfiles = os.listdir(directory+'/'+file)
    for sub in subfiles:
       extension = os.path.splitext(os.path.basename(sub))[1]
       if extension == '.csv':
            df = pd.read_csv(directory+'/'+file+'/'+sub, sep=';')
            roi.append(df)

#images = np.stack([img[:, :, np.newaxis] for img in images], axis=0).astype(np.float32)
#image_labels = np.matrix(image_labels).astype(np.float32)
image = np.load('images.txt.npy')
image_labels = np.load('labels.txt.npy')
images = np.asarray(image)
image_labels = np.asarray(image_labels)
print(images.shape)
print(image_labels.shape)
plt.imshow(images[45, :, :, :].reshape(32, 32), cmap='gray')
print(image_labels[45, :])
plt.show()
print(roi[0])

(train_X, test_X, train_y, test_y) = train_test_split(images, image_labels, test_size=0.2, random_state=42)
print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)

model = tf.keras.models.Sequential()
input_shape = (32, 32, 1) # grey-scale images of 32x32
 
model.add(tf.keras.layers.Conv2D(32, (5, 5), padding='same', 
            activation='relu', input_shape=input_shape))
model.add(tf.keras.layers.BatchNormalization(axis=-1))      
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.2))
        
model.add(tf.keras.layers.Conv2D(64, (5, 5), padding='same', 
                                 activation='relu'))
model.add(tf.keras.layers.BatchNormalization(axis=-1))
model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same', 
                                 activation='relu'))
model.add(tf.keras.layers.BatchNormalization(axis=-1))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.2))
 
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))
 
model.add(tf.keras.layers.Dense(43, activation='softmax'))
optimizer = tf.keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
              metrics=['accuracy'])
 
model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=10)
print(model.summary())
model.save_weights('model/model_weights.h5')
model_json = model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)

