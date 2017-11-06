import csv
import cv2
import numpy as np
import sklearn
import math
from random import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split

correction = [0.0, 0.2, -0.2]
lines = []
batch_size = 128

# Loads Data from file into global buffer to allow multiple files to be loaded
def getData(filename):
    with open(filename) as file:
        reader = csv.reader(file)
        for line in reader:
            lines.append(line)

# Generator function to batch data more efficiently due to memory constraints
def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_angle = float(batch_sample[3])
                # Get Left, Center, Right, and flipped L,C,R images and steering angles
                for i in range(3):
                    source_path = batch_sample[i]
                    image = cv2.imread(source_path)
                    images.append(image)
                    angles.append(center_angle + correction[i])
                    images.append(np.fliplr(image))
                    angles.append((center_angle + correction[i]) * -1)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

#Load Data
getData('./TrainingData/ORI/driving_log.csv')       # Basic Two Lap - Normal Speed Pass
getData('./TrainingData/driving_log.csv')           # Basic One Lap - Low Speed Pass
getData('./TrainingData_Track2/driving_log.csv')    # Advanced One Lap - Normal Speed Pass
#getData('./TrainingData_Default/driving_log.csv')  # Bundled Default Training Data

# Split training and validation samples from overall set
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320  # Trimmed image format

#Define Keras Model (Based on NVIDIA Network)
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(row, col, ch)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit_generator(X_train,y_train,validation_split=0.2, shuffle=True, nb_epoch=5, batch_size=512)
# Keras2 requires steps
steps_per_epoch = math.ceil(len(train_samples)/batch_size)
validation_steps = math.ceil(len(validation_samples)/batch_size)

# Train the model
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, validation_data=validation_generator, validation_steps=validation_steps, nb_epoch=3)

# Save model off for later use
model.save('model.h5')

exit()
