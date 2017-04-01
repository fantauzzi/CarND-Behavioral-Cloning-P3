import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout
import tensorflow as tf
from keras import backend
import sklearn
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from random import randint, uniform
import math
import time
import csv

crop_top = 70
crop_bottom = 25
reduce_f = 6
# Translate the image by `shift` pixels (shift > 0 is to the right), fill in blank with black
def shift_image(image, amount):
    M = np.float32([[1, 0, 0], [0, 1, amount]])
    image = cv2.warpAffine(image, M, dsize=image.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)
    return image


def make_darker(image):
    Y_channel = image[:,:,0]
    Y_channel = (Y_channel/1.5 ).astype(int)
    image[:,:,0] = Y_channel
    return image


def pre_process(image):
    height = image.shape[0]
    image = image [crop_top:height-crop_bottom,:,:]
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # L_channel = image[:,:,0]
    # equalized = cv2.equalizeHist(L_channel)
    # image[:,:,0] = equalized
    # image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return image


def rotate_image(image, amount):
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows-1), amount, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    return rotated

def sign(n):
    if n>0:
        return 1
    elif n<0:
        return -1
    return 0

def generator(samples, batch_size, images_dir):
    assert batch_size % reduce_f == 0
    sub_batch_size = batch_size // reduce_f
    # TODO make this more readable
    drift = (0, .6, -.6)  # (Center, Left, Right)
    add_drift = (0, .1, -.1)  # (Not used, Left, Right)
    shift = (0, 50, -50)  # (Not used, Left, Right)
    num_samples = len(samples)
    while True:
        np.random.shuffle(samples)
        for offset in range(0, num_samples, sub_batch_size):
            batch_samples = samples[offset:offset + sub_batch_size]
            images, angles = [], []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = images_dir + '/' + batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    if image is None:
                        print('File not found:', name)
                        print('Source:', batch_sample[i])
                    assert image is not None
                    angle = float(batch_sample[3]) + drift[i]
                    '''
                    if i ==0 and abs(angle) >3:
                        rotation = -uniform(0, 20)*sign(angle)
                        rotated = rotate_image(image, rotation)
                        rotated = pre_process(rotated)
                        images.append(rotated)
                        angles.append(angle-(rotation/20.0)*0.5)
                        '''
                    if i == 99 and abs(angle) >10:
                        amount = randint(0,30)
                        shifted = shift_image(image, amount)
                        shifted = pre_process(shifted)
                        images.append(shifted)
                        shifted_angle = angle+sign(angle)*(amount/30)*2.
                        angles.append(shifted_angle)
                        flipped = np.fliplr(shifted)
                        images.append(flipped)
                        angles.append(-shifted_angle)
                    image = pre_process(image)
                    images.append(image)
                    angles.append(angle)
                    flipped = np.fliplr(image)
                    images.append(flipped)
                    angles.append(-angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def main():

    if True:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
    
    backend.set_session(sess)

    # Base directory for the dataset
    dataset_dir = '/home/fanta/datasets/udacity/data'

    # Load telemetry from the dataset
    # csv_fname = dataset_dir + '/driving_log.csv'
    telemetry = []
    input_fnames = os.listdir(dataset_dir)
    input_fnames = [fname for fname in input_fnames if fname.endswith('.csv')]

    for csv_fname in input_fnames:
        with open(dataset_dir+'/'+csv_fname) as csv_file:
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

    # compile and train the model using the generator function
    batch_size = 120

    # How many images (included augmented data) are processed for every image in the dataset
    sub_batch_factor = reduce_f

    assert batch_size % sub_batch_factor == 0
    train_generator = generator(train_samples, batch_size=batch_size, images_dir=images_dir)
    validation_generator = generator(validation_samples, batch_size=batch_size, images_dir=images_dir)

    # Preprocess incoming data, centered around zero with small standard deviation

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255 - .5) * 2, input_shape=(160-crop_top-crop_bottom, 320, 3)))
    # model.add(Lambda(lambda x: (x / 255 - .5) * 2, input_shape=(160, 320, 3)))
    # model.add(Cropping2D(cropping=((65, 25), (0, 0))))
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
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

if __name__ == '__main__':
    main()
