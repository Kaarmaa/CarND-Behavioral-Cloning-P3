import csv
import cv2
import numpy as np

correction = [0.0,0.2,-0.2]
lines = []

def getData(filename):
    with open(filename) as file:
        reader = csv.reader(file)
        for line in reader:
            lines.append(line)

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

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2, shuffle=True, nb_epoch=5, batch_size=128)

model.save('model.h5')
exit()
