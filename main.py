from scipy.misc import imresize
from sklearn.model_selection import train_test_split
from etaprogress.progress import ProgressBar
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import os
import sys
import keras

# Initial shape of the images: 424x424
# Desired shape of input
shape_x, shape_y = 212, 212
# Number of channels of the images
channels = 3


def resize_image(image, target_width=shape_x, target_height=shape_y, max_zoom=0.2):
    """Zooms and crops the image randomly for data augmentation."""

    # First, let's find the largest bounding box with the target size ratio that fits within the image
    height = image.shape[0]
    width = image.shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = width if crop_vertically else int(height * target_image_ratio)
    crop_height = int(width / target_image_ratio) if crop_vertically else height

    # Now let's shrink this bounding box by a random factor (dividing the dimensions by a random number
    # between 1.0 and 1.0 + `max_zoom`.
    resize_factor = np.random.rand() * max_zoom + 1.0
    crop_width = int(crop_width / resize_factor)
    crop_height = int(crop_height / resize_factor)

    # Next, we can select a random location on the image for this bounding box.
    x0 = np.random.randint(0, width - crop_width)
    y0 = np.random.randint(0, height - crop_height)
    x1 = x0 + crop_width
    y1 = y0 + crop_height

    # Let's crop the image using the random bounding box we built.
    image = image[y0:y1, x0:x1, :]

    # Let's also flip the image horizontally with 50% probability:
    if np.random.rand() < 0.5:
        image = np.fliplr(image)

    # Now, let's resize the image to the target dimensions.
    image = imresize(image, (target_width, target_height))

    # Finally, let's ensure that the colors are represented as
    # 32-bit floats ranging from 0.0 to 1.0 (for now):
    return image.astype(np.uint8)


def create_model():
    """
    It builds a Convolutional Neural Networks using the keras library
    :return: A keras model
    """

    # We define a neural network
    model = keras.Sequential()

    # Then, we will be adding some layers

    # Layer 1
    model.add(keras.layers.Convolution2D(48, [5, 5], input_shape=(shape_x, shape_y, channels), activation='relu',
                                         name='conv1'))
    model.add(keras.layers.MaxPooling2D([3, 3], [3, 3], name='pool1'))

    # Layer 2
    model.add(keras.layers.Convolution2D(96, [5, 5], activation='relu', name='conv2'))
    model.add(keras.layers.MaxPooling2D([2, 2], [2, 2], name='pool2'))

    # Layer 3
    model.add(keras.layers.Convolution2D(192, [3, 3], activation='relu', name='conv3'))

    # Layer 4
    model.add(keras.layers.Convolution2D(192, [3, 3], activation='relu', name='conv4'))

    # Layer 5
    model.add(keras.layers.Convolution2D(384, [3, 3], activation='relu', name='conv5'))

    # Layer 6
    model.add(keras.layers.Convolution2D(384, [3, 3], activation='relu', name='conv6'))
    model.add(keras.layers.MaxPooling2D([3, 3], [3, 3], name='pool6'))

    # Layer 7
    model.add(keras.layers.Dense(2048, activation='relu', name='dense7'))
    model.add(keras.layers.Dropout(0.5, name='drop7'))

    # Layer 8
    model.add(keras.layers.Dense(2048, activation='relu', name='dense8'))
    model.add(keras.layers.Dropout(0.5, name='drop8'))
    model.add(keras.layers.Flatten(name='flat8'))

    # Layer 9
    model.add(keras.layers.Dense(37, activation='relu', name='dense9'))

    # We will use a Stochasric Gradient Descent optimizer
    sgd = keras.optimizers.SGD(lr=0.1)

    # We will build the network, which uses the Mean Square Error loss function
    model.compile(loss='mean_squared_error', optimizer=sgd)

    return model


def main():
    # Classes for images
    solutions = "data/solutions.csv"

    # Read data from the solutions file
    df = pd.read_csv(solutions, index_col=0, header=0)

    # Set the indices as labels of type=str, in order to convert the labels
    # of the dataframe to:
    # Name of file - 'nameOfImage'.jpg --> Label of dataframe - 'nameOfImage'
    df.index = df.index.map(str)

    # Get the length of the dataset
    total = len(df.index.values)
    bar = ProgressBar(total, max_width=800)

    # Create two lists
    # x: List formed by arrays of 3 dimensions of 212x212x3,
    # containing the pixels of the images
    # y: List formed by arrays of 1 dimension of 37,
    # containing the labaels of each image
    x, y = [], []
    i = 0
    print("Reading files...")
    for name in df.index.values[:]:
        image = mpimg.imread(os.path.join("data/images", name + ".jpg"))[:, :, :channels]
        image = resize_image(image)
        x.append(image)
        y.append(df.loc[name].values)

        bar.numerator = i
        print(bar, end='\r')
        sys.stdout.flush()
        i = i + 1

    x = np.asarray(x)
    y = np.asarray(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    load = input("Load a trained model? [y/N]")

    if load.lower() == 'y':
        # Load a previosly trained model from files
        json_file = open('trained_model/model.json', 'r')
        model_json = json_file.read()
        json_file.close()

        model = keras.models.model_from_json(model_json)
        model.load_weights("trained_model/weights.h5")

        sgd = keras.optimizers.SGD(lr=0.1)
        model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    else:
        # Build the network
        model = create_model()

        # Training the model
        hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

        # Histogram of the training troughout the epochs
        plt.figure(figsize=(12, 8))
        plt.plot(hist.epoch, hist.history['loss'], label='Test')
        plt.plot(hist.epoch, hist.history['val_loss'], label='Validation', linestyle='--')
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.legend()
        plt.savefig('trained_model/histogram.png')

        # Save the trained model for future use
        model_json = model.to_json()
        with open("trained_model/model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("trained_model/weights.h5")

    # Evaluate the model using accuracy
    scores = model.evaluate(x_test, y_test, batch_size=32)
    print("\nModel accuracy: {0:.2f}%".format(scores[1] * 100))


if __name__ == '__main__':
    main()
