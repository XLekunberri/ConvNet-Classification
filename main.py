import os
import numpy as np
import pandas as pd
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
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
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    return model


def main():
    # Classes for images
    solutions = "data/solutions.csv"

    # Read data from imagesc
    df = pd.read_csv(solutions, index_col=0, header=0)

    # Set the indices as labels of type=str
    df.index = df.index.map(str)

    total = len(df.index.values)

    x, y = [], []
    i = 0
    for name in df.index.values[:]:
        image = mpimg.imread(os.path.join(data_path, name + ".jpg"))[:, :, :channels]
        image = resize_image(image)
        x.append(image)
        y.append(df.loc[name].values)

        print('Loading photos... {0:.2f}%'.format((i / total) * 100), end="\r")
        i = i + 1

    print("\nPhotos loaded: {}\n".format(i))
    x = np.asarray(x)
    y = np.asarray(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    model = create_model()

    load = input("Load a trained model? [y/N]")

    if load.lower() == 'y':
        json_file = open('trained_model/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = keras.models.model_from_json(loaded_model_json)
        loaded_model.load_weights("trained_model/weights.h5")

        sgd = keras.optimizers.SGD(lr=0.1)
        loaded_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

        scores = loaded_model.evaluate(x_test, y_test, batch_size=32)
        print("\nModel accuracy: {0:.2f}%".format(scores[1] * 100))
    else:
        hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

        plt.figure(figsize=(12, 8))
        plt.plot(hist.epoch, hist.history['loss'], label='Test')
        plt.plot(hist.epoch, hist.history['val_loss'], label='Validation', linestyle='--')
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.legend()
        plt.savefig('trained_model/histogram.png')

        model_json = model.to_json()
        with open("trained_model/model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("trained_model/weights.h5")

        scores = model.evaluate(x_test, y_test, batch_size=32)
        print("\nModel accuracy: {0:.2f}%".format(scores[1] * 100))


if __name__ == '__main__':
    main()
