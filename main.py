import pandas as pd
# import numpy as np


"""
List of names of images
list(df.index.values)

List of names of solution values
list(df.columns.values)

Value of specific label and column
df.at['100008', 'Class1.1']
"""


def main():

    # Number of images to use
    n_img = 10000

    # Classes for images
    solutions = "training/training_solutions.csv"

    # Read data from imagesc
    df = pd.read_csv(solutions, index_col=0, header=0, nrows=n_img)

    # Set the indices as labels of type=str
    df.index = df.index.map(str)

if __name__ == '__main__':
    main()
