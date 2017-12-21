import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.misc import imresize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Number of images to use
n_img = 10000

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
    df = pd.read_csv(solutions, index_col=0, header=0, nrows=n_img)

    # Set the indices as labels of type=str
    df.index = df.index.map(str)

    X = tf.placeholder(tf.float32, shape=[None, 212, 212, 3], name="X")

    img = mpimg.imread(os.path.join("training/images", "100008.jpg"))[:, :, :channels]

    new_img = resize_image(img)

    plt.imshow(new_img)
    plt.axis("off")
    plt.show()

    # Input layer
    # input_layer = tf.reshape(features["x"], [-1, shape_x, shape_y, channels])

    with tf.name_scope("CNN"):

        """Convolutional layers"""
        # Layer 1
        conv1 = tf.layers.conv2d(input_layer, filters=48, kernel_size=[5, 5], strides=[1, 1],
                                 padding='same', name='conv1', activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(norm1, pool_size=[3, 3], strides=[3, 3], name='pool1')

        # Layer 2
        conv2 = tf.layers.conv2d(pool1, filters=96, kernel_size=[5, 5], strides=[1, 1],
                                 padding='same', name='conv2', activation=tf.nn.relu)

        pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[2, 2], name='pool2')

        # Layer 3
        conv3 = tf.layers.conv2d(pool2, filters=192, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', name='conv3', activation=tf.nn.relu)

        # Layer 4
        conv4 = tf.layers.conv2d(conv3, filters=192, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', name='conv4', activation=tf.nn.relu)

        # Layer 5
        conv5 = tf.layers.conv2d(conv4, filters=384, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', name='conv5', activation=tf.nn.relu)

        # Layer 6
        conv6 = tf.layers.conv2d(pool1, filters=96, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', name='conv6', activation=tf.nn.relu)

        pool6 = tf.layers.max_pooling2d(conv6, pool_size=[3, 3], strides=[3, 3], name='pool6')

        """Fully conected layers"""
        # Layer 7
        full7 = tf.layers.dense(pool6, units=4*4, name='full7')

        drop7 = tf.layers.dropout(full7, rate=0.5, name='drop7')

        # Layer 8
        full8 = tf.layers.dense(drop7, units=1*1, name='full8')

        drop8 = tf.layers.dropout(full8, rate=0.5, name='drop8')

        # Layer 8
        logits = tf.layers.dense(drop8, units=1*1, name='out')






if __name__ == '__main__':
    main()
