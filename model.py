import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

correction = [0.0,0.2,-0.2]
lines = []

def getLines(filename):
    with open(filename) as file:
        reader = csv.reader(file)
        for line in reader:
            lines.append(line)
    yield

def getData(filename):


    images = []
    measurements = []
    for line in lines:
        measurement = float(line[3])
        for i in range(3):
            source_path = line[i]
            image = cv2.imread(source_path)
            images.append(image)
            measurements.append(measurement + correction[i])
        for i in range(3):
            source_path = line[i]
            image = cv2.imread(source_path)
            images.append(np.fliplr(image))
            measurements.append((measurement + correction[i]) * -1)

    return images, measurements


images, measurements = getData('./TrainingData/driving_log.csv')
#images, measurements = getData('./TrainingData_Track2/driving_log.csv')
#images, measurements = getData('./TrainingData_Default/driving_log.csv')

X_train = np.array(images)
y_train = np.array(measurements)


model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2, shuffle=True, nb_epoch=3, batch_size=512)

model.save('model.h5')
exit()
