import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout
import tensorflow as tf
from keras import backend
import sklearn
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import math
import time
import csv

'''
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

backend.set_session(sess)
'''

# Base directory for the dataset
dataset_dir = '/home/fanta/datasets/udacity/data'

# Load telemetry from the dataset
csv_fname = dataset_dir + '/driving_log.csv'
telemetry = []
with open(csv_fname) as csv_file:
    reader = csv.reader(csv_file)
    header = True
    for line in reader:
        if header:
            header = False
            continue
        telemetry.append(line)

print('Read', len(telemetry), 'lines from input csv file', csv_fname)

# Load images from the dataset
images_dir = dataset_dir + '/IMG'

train_samples, validation_samples = train_test_split(telemetry, test_size=0.1)


# Translate the image by `shift` pixels (shift > 0 is to the right), fill in blank with black
def shift_image(image, shift):
    M = np.float32([[1, 0, shift], [0, 1, 0]])
    image = cv2.warpAffine(image, M, image.shape[1::-1])
    return image


def generator(samples, batch_size=48):
    assert batch_size % 8 == 0
    sub_batch_size = batch_size // 8
    # TODO make this more readable
    drift = (0, .5, -.5)  # (Center, Left, Right)
    add_drift = (0, .1, -.1)  # (Not used, Left, Right)
    shift = (0, 50, -50)  # (Not used, Left, Right)
    num_samples = len(samples)
    while True:
        np.random.shuffle(samples)
        for offset in range(0, num_samples, sub_batch_size):
            batch_samples = samples[offset:offset + sub_batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = images_dir + '/' + batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    if image is None:
                        print('File name:', name)
                        print('Source:', batch_sample[i])
                    assert image is not None
                    angle = float(batch_sample[3]) + drift[i]
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    images.append(image)
                    angles.append(angle)
                    images.append(np.fliplr(image))
                    angles.append(-angle)
                    # Add augmented data for the left and right camera images
                    if i == 1 or i == 2:
                        images.append(shift_image(image, shift[i]))
                        angles.append(angle + add_drift[i])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
batch_size = 48
# How many images (included augmented data) are processed for every image in the dataset
sub_batch_factor = 8
assert batch_size % sub_batch_factor == 0
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Preprocess incoming data, centered around zero with small standard deviation

model = Sequential()
# model.add(Lambda(lambda x: cv2.cvtColor(x.eval(session=sess), cv2.COLOR_RGB2LAB), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255 - .5) * 2, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (30, 30))))
model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
# model.add(Dropout(0.5))
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

start_time = time.time()

log = model.fit_generator(train_generator,
                          steps_per_epoch=math.ceil(sub_batch_factor * len(train_samples) / batch_size),
                          validation_data=validation_generator,
                          validation_steps=math.ceil(sub_batch_factor * len(validation_samples) / batch_size),
                          epochs=5,
                          verbose=1,
                          workers=1)

end_time = time.time()

print('Time elapsed during optimization in seconds:', end_time - start_time)
model_fname = 'mymodel.h5'
model.save(model_fname)
print('Model saved in file', model_fname)

plt.plot(log.history['loss'])
plt.plot(log.history['val_loss'])
plt.title('Mean squared error loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()
