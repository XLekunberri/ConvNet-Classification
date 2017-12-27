import os
import numpy as np
import pandas as pd
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import keras

data_path = "data/images/"

# Desired shape of input. Initial shape: 424x424
shape_x, shape_y = 212, 212
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
    # Build the network
    model = keras.Sequential()

    model.add(keras.layers.Convolution2D(48, [5, 5], input_shape=(shape_x, shape_y, channels), activation='relu',
                                         name='conv1'))
    model.add(keras.layers.MaxPooling2D([3, 3], [3, 3], name='pool1'))

    model.add(keras.layers.Convolution2D(96, [5, 5], activation='relu', name='conv2'))
    model.add(keras.layers.MaxPooling2D([2, 2], [2, 2], name='pool2'))

    model.add(keras.layers.Convolution2D(192, [3, 3], activation='relu', name='conv3'))

    model.add(keras.layers.Convolution2D(192, [3, 3], activation='relu', name='conv4'))

    model.add(keras.layers.Convolution2D(384, [3, 3], activation='relu', name='conv5'))

    model.add(keras.layers.Convolution2D(384, [3, 3], activation='relu', name='conv6'))
    model.add(keras.layers.MaxPooling2D([3, 3], [3, 3], name='pool6'))

    model.add(keras.layers.Dense(2048, activation='relu', name='dense7'))
    model.add(keras.layers.Dropout(0.5, name='drop7'))

    model.add(keras.layers.Dense(2048, activation='relu', name='dense8'))
    model.add(keras.layers.Dropout(0.5, name='drop8'))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(37, activation='relu', name='dense9'))

    sgd = keras.optimizers.SGD(lr=0.1)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    return model


def create_label(dataframe, i, j, name='ShapeLabel', showDistr=False):
    """
    Giving a dataframe with probabilities of having a label in each class, reduces the dataframe
    to one column, where it specifies which is the label of it.
    :param dataframe: Dataframe where the probability of the values of each class is stored
    :param i: Index of the first probability of the class we want to create a label from
    :param j: Index of the last probability of the class we want to create a label from
    :param name: Name of the new label, optional
    :param showDistr: Boolean to check the distribution of the label
    """

    n_probs = j - i + 1

    dataframe = dataframe.drop(dataframe.columns[j + 1:], axis=1)

    dataframe = dataframe.drop(dataframe.columns[:i], axis=1)
    print(dataframe.columns.values)

    dataframe[name] = pd.Series(np.random.random_integers(0, 0), index=dataframe.index)

    max_prob = -float('inf')
    for img in dataframe.index.values:
        i, label = 0, 0

        for shape_probs in dataframe.loc[img].values:
            if shape_probs > max_prob:
                max_prob = shape_probs
                label = i
            i += 1

        dataframe.at[img, name] = label

    dataframe = dataframe.drop(dataframe.columns[:n_probs], axis=1)

    if showDistr:
        arr = np.zeros(n_probs)

        for label in dataframe.loc[:, name].values:
            arr[label] += 1

    for n in range(n_probs):
        print("Label {}: {}%".format(n, round(arr[n] / len(dataframe.loc[:, name].values) * 100, 2)))

    return dataframe


def main():
    # Classes for images
    solutions = "data/solutions.csv"

    # Read data from imagesc
    df = pd.read_csv(solutions, index_col=0, header=0)  # , nrows=train_size)

    # Set the indices as labels of type=str
    df.index = df.index.map(str)

    total = len(df.index.values)

    X, Y = [], []
    i = 0
    for name in df.index.values[:]:
        image = mpimg.imread(os.path.join("data/training", name + ".jpg"))[:, :, :channels]
        image = resize_image(image)
        X.append(image)
        Y.append(df.loc[name].values)

        print('Fotos leidas: {0:.2f}%'.format((i / total) * 100), end="\r")
        i = i + 1

    X = np.asarray(X)
    Y = np.asarray(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

    model = create_model()

    model.fit(X_train, Y_train, epochs=10, batch_size=32)

    model_json = model.to_json()
    with open("models/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("models/model.h5")


if __name__ == '__main__':
    main()
