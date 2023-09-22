import numpy as np
from sklearn import preprocessing, impute


# def handle_missing_data(S=np.ndarray):
#     return NotImplementedError


def binarize_categorical_onehot(col, unique_count):
    """
    Binarize categorical values with one-hot encoding: https://en.wikipedia.org/wiki/One-hot
    :param col: the column being converted into binary in the data
    :param unique_count: the unique number of values in the column
    :return: a one-hot encoded version of a categorical column in the index provided for the data
    """
    # create a map for each unique value found the data column
    unique_mapping = dict(zip(col, range(unique_count)))
    return np.eye(unique_count)[np.vectorize(unique_mapping.get)(col)]


def binarize_numeric_median(col):
    """
    Binarize numeric values based on the median of the column
    :param col: values to binarize
    :return:    the binarized column
    """
    # return a binarized column using the column's median
    return np.where(col > np.median(col), 1, 0)


def clean(data=np.ndarray):
    """
    Cleans a given dataset (as a np.ndarray) by checking the data types and (to-be-implemented) handling missing values.
    :param data: a np.ndarray containing the original values in a dataset
    :return:
    """
    new_data = []
    for col in range(data.shape[1]):
        column_vals = data[:, col]
        unique_vals = np.unique(data[:, col])
        # if not binary, handle categorical values (via one-hot encoding) and/or numeric (using the median)
        if len(unique_vals) > 2:
            # handle categorical column
            if data[:, col].dtype == 'object':
                binarize_categorical_onehot(column_vals, len(unique_vals))
            # handle numeric column
            else:
                binarize_numeric_median(column_vals)
        # otherwise, the data is binary; we don't changee it
        else:
            new_data.append(column_vals.reshape(-1, 1))
    return new_data