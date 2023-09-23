import pandas as pd
import numpy as np


def entropy(y):
    return -np.sum(calculate_prob(y) * np.log2(calculate_prob(y)))


def gini(y):
    return 1 - np.sum(np.square(calculate_prob(y)))


def max_error(y):
    return 1 - np.max(calculate_prob(y))


def calculate_prob(y):
    unique_y, counts = np.unique(y, return_counts=True)
    return counts / len(y)


def split_data(X, y, best_split_inx, split_val):
    """
    Given the index of an attribute, this method splits the dataset into two subsets (X,y) with the given value.
    Parameters:
    :param X: data (np.array)
    :param y: labels (np array)
    :param best_split_inx: index of the attribute (int)
    :param split_val: the attribute value used to split the data
    """
    # split the dataset based on the current attribute value
    subset_mask = X[:, best_split_inx] == split_val
    return X[subset_mask], y[subset_mask]

def calc_prediction_error(y_true, y_pred):
    """
    This method calculates the average prediction error given the true labels (y_true) and the predicted labels (y_pred).

    Parameters:
    y_true: True labels (np.array)
    y_pred: Predicted labels (np.array)
    """
    return np.mean(y_true != y_pred)


def preprocess_numerical(X):
    """
    Convert numerical attributes to binary based on median.

    Parameters:
    :param X: a set or subset possibly containing numerical attributes (np.array)
    """
    # binarize each column in X with numerical values
    for col_index in range(X.shape[1]):
        # check for numerical values
        col_values = X[:, col_index]
        if np.issubdtype(col_values.dtype, np.number):
            # get the median
            median = np.median(col_values)
            # binarize using the median
            X[:, col_index] = np.where(col_values >= median, 1, 0)
    return X


class Node:
    def __init__(self, value=None, attribute=None, children=None):
        """
      A constructor for a small Node class. This class helps build the decision tree.
      It's a tree node containing the attribute it's assinged to and the value it splits on as well as a map of its children (if any).
      """
        self.value = value
        self.attribute = attribute
        self.children = {} if children is None else children


class ID3:
    def __init__(self, max_depth: int = 6, impurity_measure: str = "entropy"):
        self.max_depth = max_depth
        self.impurity_measure = impurity_measure
        self.root = None
        self.most_common_label = None

    def fit(self, X, y):
        """
        Train the model given X, y.
        Then it will then initialize the root node and call the recursive function (build_tree) to build the tree.

        Parameters:
        :param X: attributes (np.array)
        :param y: labels (np array)
        :return:
        """
        # preprocess numerical attributes, if any
        X = preprocess_numerical(X)
        # for prediction if a value is not found
        self.most_common_label = np.unique(y)[0]
        # create the root node and start recursive process
        self.root = self.build_tree(X, y, depth=0)

    def calc_impurity(self, y):
        """
        Given a subset y of labels and an impurity measure specified by the user ('entropy', 'GINI', 'max_error'),
        this method calculates the impurity of a set. It will be repeatedly used for calculating Information Gain.

        Parameters:
        :param y: labels (np array)
        :return: calculated impurity (float)
        """
        if self.impurity_measure == "entropy":
            return entropy(y)
        elif self.impurity_measure == "me":
            return max_error(y)
        elif self.impurity_measure == "gini":
            return gini(y)
        else:
            return gini(y) #had to hardcode this for all 3, something wasn't going right

    def information_gain(self, X, y, attribute_index):
        """
        Given a feature index, this method calculates the Information Gain using the calc_impurity helper method.

        Parameters:
        :param X: Attributes (numpy array)
        :param y: labels (np array)
        :param attribute_index: index of the attribute (int)
        :return: Information Gain
        """

        # get impurity for the complete dataset
        IG_S = self.calc_impurity(y)

        # Split data based on attribute values
        unique_values = np.unique(X[:, attribute_index])

        S_i_sum = 0

        for value in unique_values:
            # get a subset of the rows where the attribute equal to current value in X
            bool_mask = X[:, attribute_index] == value
            # get the respective y subset
            y_S_i = y[bool_mask]

            # get the impurity for the current subset
            S_i_impurity = self.calc_impurity(y_S_i)

            # weigh the impurity by the size of the subset
            weight = len(y_S_i) / len(y)

            # add to the total weighted impurity sum
            S_i_sum += weight * S_i_impurity

        # return the information gain
        return IG_S - S_i_sum

    def find_best_split(self, X, y):
        """
        Given a set X, this method finds the best split by finding the attribute with the max information gain and returning its index in X.
        :param X:
        :param y:
        :return:
        """
        IGs = []
        for i in range(X.shape[1]):
            # calculate the information gain of the ith attribute
            IGs.append(self.information_gain(X, y, i))
        return np.argmax(IGs)

    def build_tree(self, X, y, depth):
        """
        To recursively build the decision tree, for each feature in X call information_gain to find the best feature to split.

        Parameters:
        :param X: attributes (np.array)
        :param y: labels (np array)
        :param depth: the current depth of the tree (int)
        """
        # base cases: all elements have same class OR no attributes remaining OR max depth reached
        unique_targets = np.unique(y)
        if len(unique_targets) == 1 or X.shape[1] == 0 or depth == self.max_depth:
            return Node(value=y[0])

        # if no base cases are met, find the next best attribute to split on and init a new node on it
        best_split_inx = self.find_best_split(X, y)
        new_node = Node(attribute=best_split_inx)

        # then continue recursively building the tree, adding children to the new node
        unique_values = np.unique(X[:, best_split_inx])
        for value in unique_values:
            X_subset, y_subset = split_data(X, y, best_split_inx, value)
            child_node = self.build_tree(X_subset, y_subset, depth + 1, print_steps)
            new_node.children[value] = child_node

        # return the new node
        return new_node

    def predict(self, X):
        """
        This method traverses the constructed tree for each example in X and returns predicted labels for y.
        Parameters:
        :param X: attributes (np.array)
        """
        predictions = []
        for sample in X:
            prediction = self.traverse_tree(sample, self.root)  # Start traversing from the root
            predictions.append(prediction)
        return np.array(predictions)

    def traverse_tree(self, sample, node):
        """
        This helper method traverses the tree to make a prediction
        :param sample: a row in X (np.array)
        :param node: a node in the tree
        :return:
        """
        if node.value is not None:
            return node.value  # If it's a leaf node, return the predicted class
        else:
            attribute_index = node.attribute
            attribute_value = sample[attribute_index]
            if attribute_value in node.children:
                child_node = node.children[attribute_value]
                return self.traverse_tree(sample, child_node)  # Recursively traverse the tree
            else:
                # Handle cases where the attribute value is not in the training data
                return self.most_common_label


def load_and_seperate_data(train_path: str, test_path: str):
    """
    This helper method reads the train and test csv files from their provided filepaths.
    It then converts them to numpy arrays and splits them into their X and y sets.
    :param train_path: the filepath for the training data
    :param test_path: the filepath for the test data
    :return: the attributes (X) and labels (y) for both train and test.
    """
    # load the data
    test = pd.read_csv(test_path, header=None).to_numpy()
    train = pd.read_csv(train_path, header=None).to_numpy()

    # seperate (X,y) for train and test sets
    X_train, y_train, = sep_X_y(train)
    X_test, y_test = sep_X_y(test)

    return X_train, y_train, X_test, y_test


def sep_X_y(data):
    """
    A helper method for spliting X and y.
    """
    return data[:, :-1], data[:, -1]


def main(train_path, test_path):
    # Separate features and labels
    X_train, y_train, X_test, y_test = load_and_seperate_data(train_path, test_path)

    # Initialize a dictionary to keep track of average errors for each impurity measure
    average_errors = {
        "entropy": [],
        "me": [],
        "gini": []
    }

    # List of impurity measures you'll use
    impurity_measures = ["entropy", "me", "gini"]


    #entropy
    print("\n ENTROPY \n")
    for depth in range(1, 7):  # Depth varies from 1 to 6
        tree = ID3(max_depth=depth, impurity_measure="entropy")
        tree.fit(X_train, y_train)

        y_pred_train = tree.predict(X_train)
        avg_error_train = calc_prediction_error(y_train, y_pred_train)

        y_pred_test = tree.predict(X_test)
        avg_error_test = calc_prediction_error(y_test, y_pred_test)

        # Append the average errors for this depth and impurity measure to the list
        average_errors["entropy"].append((depth, avg_error_train, avg_error_test))

        # Displaying average_errors; you can also print this to a file or plot it
        for measure, errors in average_errors.items():
            print(f"For {entropy}:")
            for depth, err_train, err_test in errors:
                print(f"Depth: {depth}, Training Error: {err_train}, Test Error: {err_test}")
    #me
    print("\n ME \n")

    for depth in range(1, 7):  # Depth varies from 1 to 6
        tree = ID3(max_depth=depth, impurity_measure="me")
        tree.fit(X_train, y_train)

        y_pred_train = tree.predict(X_train)
        avg_error_train = calc_prediction_error(y_train, y_pred_train)

        y_pred_test = tree.predict(X_test)
        avg_error_test = calc_prediction_error(y_test, y_pred_test)

        # Append the average errors for this depth and impurity measure to the list
        average_errors["me"].append((depth, avg_error_train, avg_error_test))

        # Displaying average_errors; you can also print this to a file or plot it
        for measure, errors in average_errors.items():
            print(f"For {max_error}:")
            for depth, err_train, err_test in errors:
                print(f"Depth: {depth}, Training Error: {err_train}, Test Error: {err_test}")

    #gini
    print("\n GINI \n")

    for depth in range(1, 17):  # Depth varies from 1 to 6
        tree = ID3(max_depth=depth, impurity_measure="gini")
        tree.fit(X_train, y_train)

        y_pred_train = tree.predict(X_train)
        avg_error_train = calc_prediction_error(y_train, y_pred_train)

        y_pred_test = tree.predict(X_test)
        avg_error_test = calc_prediction_error(y_test, y_pred_test)

        # Append the average errors for this depth and impurity measure to the list
        average_errors["gini"].append((depth, avg_error_train, avg_error_test))

        # Displaying average_errors; you can also print this to a file or plot it
        for measure, errors in average_errors.items():
            print(f"For {gini}:")
            for depth, err_train, err_test in errors:
                print(f"Depth: {depth}, Training Error: {err_train}, Test Error: {err_test}")


if __name__ == "__main__":
    main("/Users/annabell/Desktop/Homework This Week/ML_HW1_D3/ML1/bank-4/train.csv", "/Users/annabell/Desktop/Homework This Week/ML_HW1_D3/ML1/bank-4/test.csv")
