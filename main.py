import cv2
import pandas
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout
import tensorflow as tf
from keras import backend
import sklearn
from sklearn.model_selection import train_test_split
import math
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

backend.set_session(sess)


# Base directory for the dataset
dataset_dir = '/home/fanta/datasets/udacity/data'

# Load telemetry from the dataset
csv_fname = dataset_dir + '/driving_log.csv'
telemetry = pandas.read_csv(csv_fname)
print('Read', len(telemetry.index), 'lines from input csv file', csv_fname)

# Load images from the dataset
images_dir = dataset_dir + '/IMG'
'''
images = []
steering = []
for _, line in telemetry.iterrows():
    image_fname = images_dir + '/' + line['center'].split('/')[-1]
    image = cv2.imread(image_fname)
    images.append(image)
    steering.append(line['steering'])
    images.append(np.fliplr(image))
    steering.append(line['steering'] * -1.0)
'''

# images=[cv2.imread(images_dir+'/'+f_name.split('/')[-1]) for f_name in telemetry['center']]
# print('Read', len(images), 'images from directory', images_dir)

# X_train = np.array(images)
# y_train = np.array(steering)


train_samples, validation_samples = train_test_split(telemetry.as_matrix(), test_size=0.2)
import matplotlib.pyplot as plt
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        np.random.shuffle(samples)
        flipped = False
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = images_dir+'/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2LAB)
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
batch_size = 32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 80, 320  # Trimmed image format

# Preprocess incoming data, centered around zero with small standard deviation
'''
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(ch, row, col),
        output_shape=(ch, row, col)))
model.add(... finish defining the rest of your model architecture here ...)
'''

model = Sequential()
# model.add(Lambda(lambda x: cv2.cvtColor(x.eval(session=sess), cv2.COLOR_RGB2LAB), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255 - .5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(384, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(192, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(96, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

start_time = time.time()
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)//batch_size),
                    validation_data=validation_generator, validation_steps=len(validation_samples), epochs=3, verbose=2,
                    workers=1)
end_time = time.time()

# model.fit(X_train, y_train, validation_split=.2, shuffle=True, epochs=4, verbose=2, batch_size=32)

print('Time elapsed during optimization in seconds:', end_time-start_time)
model_fname = 'mymodel.h5'
model.save(model_fname)
print('Model saved in file', model_fname)
