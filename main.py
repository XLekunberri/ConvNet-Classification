import os
import numpy as np
import pandas as pd
from scipy.misc import imresize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from math import sqrt


# Number of images to use
train_size = 100
test_size = 100

# Desired shape of input. Initial shape: 424x424
shape_x, shape_y = 120, 120
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
    image = image[y0:y1, x0:x1]

    # Let's also flip the image horizontally with 50% probability:
    if np.random.rand() < 0.5:
        image = np.fliplr(image)

    # Now, let's resize the image to the target dimensions.
    image = imresize(image, (target_width, target_height))

    # Finally, let's ensure that the colors are represented as
    # 32-bit floats ranging from 0.0 to 1.0 (for now):
    return image.astype(np.float32) / 255



"""
List of names of images
list(df.index.values)

List of names of solution values
list(df.columns.values)

Value of specific label and column
df.at['100008', 'Class1.1']
"""


def main():

    # Classes for images
    solutions = "training/training_solutions.csv"

    # Read data from imagesc
    df = pd.read_csv(solutions, index_col=0, header=0, nrows=train_size)

    # Set the indices as labels of type=str
    df.index = df.index.map(str)

    img = mpimg.imread(os.path.join("training/images", "100008.jpg"))[:, :, :channels]

    new_img = resize_image(img)

    """
    plt.imshow(new_img)
    plt.axis("off")
    plt.show()
    """

    # Input layer
    # input_layer = tf.reshape(features["x"], [-1, shape_x, shape_y, channels])


    # Build the network
    model = keras.Sequential()

    model.add(keras.layers.Convolution2D(48,  [5, 5], input_shape=(shape_x, shape_y, channels), activation='relu'))
    model.add(keras.layers.MaxPooling2D([3, 3], [3, 3]))

    model.add(keras.layers.Convolution2D(96,  [5, 5], input_shape=(shape_x, shape_y, channels), activation='relu'))
    model.add(keras.layers.MaxPooling2D([2, 2], [2, 2]))

    model.add(keras.layers.Convolution2D(192, [3, 3], input_shape=(shape_x, shape_y, channels), activation='relu'))

    model.add(keras.layers.Convolution2D(192, [3, 3], input_shape=(shape_x, shape_y, channels), activation='relu'))

    model.add(keras.layers.Convolution2D(384, [3, 3], input_shape=(shape_x, shape_y, channels), activation='relu'))

    model.add(keras.layers.Convolution2D(384, [3, 3], input_shape=(shape_x, shape_y, channels), activation='relu'))
    model.add(keras.layers.MaxPooling2D([3, 3], [3, 3]))

    model.add(keras.layers.Dense(2048, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(2048, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(2048, activation='relu'))

    sgd = keras.optimizers.SGD(lr=0.1)
    def rmse(y_true, y_pred):

        # sqrt((1 / test_size) * [error for error )
        """
        :param y_true: Actual value
        :param y_pred: Predicted value
        :return: The Root Mean Squared Error of the labels
        """

    model.compile(loss='mean_squared_error', optimizer=sgd)


    # shuffle=True because it seems to get better results
    #model.fit(shuffle=True)









if __name__ == '__main__':
    main()
