import pickle
import tensorflow as tf
import cv2
import numpy as np
from keras.layers import Input, Flatten, Dense
from keras.models import Model, Sequential
from matplotlib import pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', 'vgg_driving_bottleneck_features_train.p', "Bottleneck features training file (.p)")
flags.DEFINE_integer('epochs', 25, "The number of epochs.")
flags.DEFINE_integer('batch_size', 48, "The batch size.")

def load_bottleneck_data(training_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']

    return X_train, y_train


def main():
    # load bottleneck data
    X_train, y_train = load_bottleneck_data(FLAGS.training_file)

    # X_train = np.squeeze(X_train)
    model = Sequential()
    # model.add(Dense(input_shape=(fan_in,), units=fan_out, name='Dense-1'))
    model.add(Flatten(input_shape=X_train.shape[1:]))
    model.add(Dense(units=384, activation='relu', name='Dense-1'))
    model.add(Dense(units=192, name='Dense-2'))
    model.add(Dense(units=96, name='Dense-3'))
    model.add(Dense(units=48, name='Dense-4'))
    model.add(Dense(units=1, name='Dense-5'))

    model.compile(optimizer='adam', loss='mse')

    # train model
    log = model.fit(X_train,
              y_train,
              epochs=FLAGS.epochs,
              batch_size=FLAGS.batch_size,
              shuffle=True,
              validation_split=.1)

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


# parses flags and calls the `main` function above
if __name__ == '__main__':
    main()
