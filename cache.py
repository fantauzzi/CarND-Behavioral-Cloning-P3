# Adapted from source code provided by Udacity

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.layers import Input, AveragePooling2D
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.datasets import cifar10
from math import ceil
import pickle
import tensorflow as tf
import keras.backend as K
import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', '.', "Path to the dataset")
flags.DEFINE_string('network', 'resnet', "The model to bottleneck, one of 'vgg', 'inception', or 'resnet'")
flags.DEFINE_integer('batch_size', 48, 'The batch size for the generator')

batch_size = FLAGS.batch_size

# Set the input size expected by the model. TODO how to do to adjust to my input size?
h, w, ch = 224, 224, 3
if FLAGS.network == 'inception':
    h, w, ch = 299, 299, 3
    from keras.applications.inception_v3 import preprocess_input

# Used to resize the input images as necessary. I can use it, but is it OK to change aspect ratio?
img_placeholder = tf.placeholder("uint8", (None, 160-70-25, 320, 3))
resize_op = tf.image.resize_images(img_placeholder, (h, w), method=0)


def pre_process(image):
    # Convert to desired color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Crop 70, 25
    image_h = image.shape[0]
    image = image[70:image_h - 25, :]
    # Resize to meet the required input size for the NN
    image = cv2.resize(image, (h, w))
    # Ensure range of pixels is in [-1, 1]
    image = ( image / 255 - .5) * 2
    return image

def gen(telemetry, base_dir, batch_size):
    start = 0
    end = start + batch_size
    n = len(telemetry)

    # Continue until you have fetched all dataset images, and no more
    batch_count = 1
    while start < n:
        images, angles = [], []
        # Load from disk images for the current batch
        for i in range(batch_size):
            if start+i >= n:
                break
            name = base_dir + '/IMG/' + telemetry[start+i][0].split('/')[-1]
            center_image = cv2.imread(name)
            assert center_image is not None
            center_image = pre_process(center_image)
            center_angle = float(telemetry[start+i][3])
            images.append(center_image)
            angles.append(center_angle)

        X_batch = np.array(images)
        y_batch = np.array(angles)
        start += batch_size
        print('Batch processed', batch_count)
        batch_count += 1
        yield (X_batch, y_batch)


def create_model():
    input_tensor = Input(shape=(h, w, ch))
    if FLAGS.network == 'vgg':
        model = VGG16(input_tensor=input_tensor, include_top=False)
        x = model.output
        x = AveragePooling2D((7, 7))(x)
        model = Model(model.input, x)
    elif FLAGS.network == 'inception':
        model = InceptionV3(input_tensor=input_tensor, include_top=False)
        x = model.output
        x = AveragePooling2D((8, 8), strides=(8, 8))(x)
        model = Model(model.input, x)
    else:
        model = ResNet50(input_tensor=input_tensor, include_top=False)
    return model

def load_telemetry(fname):
    # Load telemetry from the dataset
    telemetry = []
    with open(fname) as csv_file:
        reader = csv.reader(csv_file)
        header = True
        for line in reader:
            if header:
                header = False
                continue
            telemetry.append(line)

    return telemetry

def main(_):

    # Load csv file with telemetry
    # Base directory for the dataset
    dataset_dir = FLAGS.dataset
    csv_fname = dataset_dir + '/driving_log.csv'
    telemetry = load_telemetry(csv_fname)
    print('Read', len(telemetry), 'lines from input csv file', csv_fname)

    # Load input data X and y
    images_dir = dataset_dir + '/IMG'

    train_output_file = "{}_{}_{}.p".format(FLAGS.network, 'driving', 'bottleneck_features_train')

    print("Resizing to", (w, h, ch))
    print("Saving to ...")
    print(train_output_file)

    with tf.Session() as sess:
        K.set_session(sess)
        K.set_learning_phase(1)

        model = create_model()

        print('Bottleneck training')
        train_gen = gen(telemetry, FLAGS.dataset, batch_size)
        # Second argument here might be wrong
        # bottleneck_features_train = model.predict_generator(train_gen, ceil(len(telemetry)/batch_size))
        bottleneck_features_train = model.predict_generator(train_gen, len(telemetry))
        data = {'features': bottleneck_features_train, 'labels': telemetry[3]}
        pickle.dump(data, open(train_output_file, 'wb'))

if __name__ == '__main__':
    tf.app.run()
