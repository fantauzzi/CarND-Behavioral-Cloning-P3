import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D
import tensorflow as tf
from keras import backend
import sklearn
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import math
import time
import csv
import argparse

crop_top = 70
crop_bottom = 25
reduce_f = 6


def sign(n):
    """
    Returns the sign of `n`, or 0 if `n` equals 0
    """
    if n > 0:
        return 1
    elif n < 0:
        return -1
    return 0


def pre_process(image):
    """
    Returns a copy of the given image after pre-processing. That is cropping and conversion fron BGR color space
    to YUV. It is meant to be used for network training and driving.
    """
    height = image.shape[0]
    image = image[crop_top:height - crop_bottom, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return image

def generator(samples, batch_size, images_dir):
    """
    Generates batches of samples to train the network. Images are pre-processed and then used to augment
    the dataset. For every pre-processed image, six images will go in the batch after augmentation. 
    :param samples: list-like with telemetry data, and their association with input images, from the CSV file. 
    :param batch_size: size of the batch to be returned, including augmented images; must be a multiple of 6.
    :param images_dir: path to the directory containing the input images referenced by `samples`.
    :return: a pair (X, y) where X is a numpy array with the batch samples, and y is a numpy array with the 
    corresponding correct values as provided by telemetry.
    """
    assert batch_size % reduce_f == 0

    ''' Correction to the steering angle for images coming from center, left and right camera respectively; used for
    data augmentation. '''
    drift = (0, .5, -.5)  # (Center, Left, Right)

    while True:
        np.random.shuffle(samples)
        # Go around the samples in batches
        for offset in range(0, len(samples), batch_size // reduce_f):
            batch_samples = samples[offset:offset + (batch_size // reduce_f)]
            images, angles = [], []
            # For every entry (line of the CSV file) in the telemetry data for this batch...
            for batch_sample in batch_samples:
                # For each of the three cameras (center, left and right respectively)...
                for i in range(3):
                    # Load the corresponding image
                    f_name = images_dir + '/' + batch_sample[i].split('/')[-1]
                    image = cv2.imread(f_name)
                    if image is None:
                        print('File not found:', f_name)
                        print('Source:', batch_sample[i])
                    assert image is not None
                    # Correct the steering angle based on which camera the image came from
                    angle = float(batch_sample[3]) + drift[i]
                    image = pre_process(image)
                    images.append(image)
                    angles.append(angle)
                    # Augment the dataset further, flipping the image around the vertical axis
                    flipped = np.fliplr(image)
                    images.append(flipped)
                    angles.append(-angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            # Yield a batch worth of training data
            yield sklearn.utils.shuffle(X_train, y_train)


def main():

    # Uncomment to prevent Tensorflow from pre-allocating all GPU memory
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    backend.set_session(sess)
    '''

    parser = argparse.ArgumentParser(description='Model Builder')
    parser.add_argument(
        'dataset_dir',
        type=str,
        nargs='?',
        default='',
        help='Path to dataset directory.'
    )
    args = parser.parse_args()

    # Base directory for the dataset
    dataset_dir = args.dataset_dir

    # Load telemetry from the dataset, reading all .csv files in the given directory
    telemetry = []
    input_fnames = os.listdir(dataset_dir)
    input_fnames = [fname for fname in input_fnames if fname.endswith('.csv')]

    for csv_fname in input_fnames:
        with open(dataset_dir + '/' + csv_fname) as csv_file:
            reader = csv.reader(csv_file)
            header = True
            for line in reader:
                if header:
                    header = False
                    continue
                telemetry.append(line)
        print('Read', len(telemetry), 'lines from input csv file', csv_fname)

    # Split dataset between training and validation
    train_samples, validation_samples = train_test_split(telemetry, test_size=0.1)

    '''Batch size includes augmented images; 5 augmented images are generated for every original image in the dataset,
    therefore the batch size should be a multiple of 6'''
    batch_size = 120

    assert batch_size % reduce_f == 0
    images_dir = dataset_dir + '/IMG'
    train_generator = generator(train_samples, batch_size=batch_size, images_dir=images_dir)
    validation_generator = generator(validation_samples, batch_size=batch_size, images_dir=images_dir)

    # Build the optimization model
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255 - .5) * 2, input_shape=(160 - crop_top - crop_bottom, 320, 3)))
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

    # Do your thing!
    log = model.fit_generator(train_generator,
                              steps_per_epoch=math.ceil(reduce_f * len(train_samples) / batch_size),
                              validation_data=validation_generator,
                              validation_steps=math.ceil(reduce_f * len(validation_samples) / batch_size),
                              epochs=10,
                              verbose=1,
                              workers=1)

    end_time = time.time()

    print('Time elapsed during optimization in seconds:', round(end_time - start_time))
    model_fname = 'model.h5'
    model.save(model_fname)
    print('Model saved in file', model_fname)

    # Chart validation and test loss per epoch
    plt.plot(log.history['loss'])
    plt.plot(log.history['val_loss'])
    plt.title('Mean squared error loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
