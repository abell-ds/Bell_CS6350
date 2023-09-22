import numpy as np
import preprocess
import information_gain as ig


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
    def __init__(self, data: np.ndarray, max_depth: int = 6, IG_method: str = ""):
        """
        This constructor initializes an ID3 Tree with optional max_depth an impurity method functions.
        :param data: the entire dataset; a np.ndarray in the format (X,y) where y is the last column of the data containing the instance outcomes/target class.
        :param max_depth: an optional parameter for the max depth of the decision tree; the default is 6.
        :param IG_method: an optional parameter for the method used to calculate uncertainty/impurity. the default is entropy.
        """
        print("Cleaning the data...")
        self.S = preprocess.clean(self.S)
        print("Initializing root and setting information gain calculator")
        self.root = None
        self.IG = ig.InformationGain(method=IG_method)

    def build_tree(self, data, attributes):
        """
        This is the driver function for ID3. It initializes the root of the tree and then the tree is recursively built.
        The driver method terminates when the root is returned.
        ```
        :param data:
        :param attributes:
        :param target_index:
        :return:
        """
        self.root = self._build_tree(data, attributes)

    def _find_attribute_with_best_gain(self, data):
        """
        Helper method for the recursive function; determines the best attribute to split on.
        :param data: the data needing to be split (S or a subset of S)
        :return: the best attribute in the data to make the next split.
        """
        # Recursive steps to continue splitting the dataset to buld the tree if neither base case is met
        best_gain = -1
        best_attribute = None
        for attr in data[:, :-1]:
            gain = self.info_gain_calculator.calc_information_gain(data, data[:, -1])
            if gain > best_gain:
                best_gain = gain
                best_attribute = attr
        return best_attribute

    def _build_tree(self, data, attributes):
        """
        Check for homogeneity...
            If:
                - if all rows are 1, return the root with label = +,
                - if all examples 0, Return the root with label = -.
                - if the number of attributes is 0, return the root with the label as the most common value.
        Otherwise, build the Decision Tree on an attribute A:
            Else:
                - for each possible value of A, add a new tree branch below the root corresponding to the test A = vi.
                - let S_i be the subset of examples that have the value vi for A in set S...
                    -- if S_vi is empty, add a leaf node below this new branch with the most common target value as the label
                    -- else, below this new branch add the next subtree.
        :param data: the set or a subset of S.
        :param attributes: attributes in the data.
        :return:
        """
        # get a list of the unique possible outcomes in y
        unique_targets = np.unique(data[:, -1])

        # base case: all elements have same class
        if len(unique_targets) == 1:
            return Node(value=unique_targets[0])

        # base case: no attributes remaining
        if len(attributes) == 0:
            return Node(value=np.argmax(np.bincount(data[:, -1].astype(int))))

        # if neither base case is met, init new node and find the attribute to split on
        best_attribute = self.find_attribute_with_best_gain(data)
        new_node = Node(attribute=best_attribute)

        for attribute_value in np.unique(data[:, best_attribute]):
            subset = data[data[:, best_attribute] == attribute_value]
            new_attributes = [attr for attr in attributes if attr != best_attribute]

            new_node.children[attribute_value] = self._build_tree(subset, new_attributes, -1)

        return new_node
