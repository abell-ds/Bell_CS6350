import numpy as np


def entropy(y):
    unique_y, counts = np.unique(y, return_counts=True)
    prob = counts / len(y)
    return -np.sum(prob * np.log2(prob))


def gini(y):
    unique_y, counts = np.unique(y, return_counts=True)
    prob = counts / len(y)
    return 1 - np.sum(np.square(prob))


def max_error(y):
    unique_y, counts = np.unique(y, return_counts=True)
    prob = counts / len(y)
    return 1 - np.max(prob)


class InformationGain:
    def __init__(self, S: np.ndarray, method: str = ""):
        """
        The constructor calculates the information gain for a given set.
        :param S: the entire dataset; a np.ndarray in the format (X,y) where y is the last column of the data containing the instance outcomes/target class.
        :param method: an optionally-specified parameter for the method for calculating impurity. if left unspecified (or the method is not available), "entropy" will be used.
        """
        print("initializing InformationGain constructor..."
              "\n ------------------------------ \n")
        self.methods = {"entropy": entropy, "gini": gini, "me": max_error}
        self.data = S.copy()
        self.method = self._get_impurity_method(method)
        print("calculating the impurity of the entire set."
              "\n ------------------------------\n")
        self.init_impurity = self.method(self.data[:, -1])
        self.attribute_IGs = list()
        self.get_attribute_IGs()
        self.total_gain = sum(self.attribute_IGs)
        print("calculating the information gain."
              "\n ------------------------------\n")
        self.IG = self.init_impurity - self.total_gain

    def _get_impurity_method(self, user_input):
        """
        This method will set entropy as the method if the user does not provide any input or an unavailable method.
        If the user specifies GINI or Maximum Error (not case-sensitive), entropy will be replaced with the specified method.
        :param user_input: the "method" string parameter in the InformationGain constructor
        :return:
        """
        if str.strip(str.lower(user_input)) in self.methods.keys():
            print(str(user_input) + " will be the method used for calculating impurity.")
            return self.methods[user_input]
        else:
            print("Entropy will be the method used for calculating impurity. \n "
                  "This is by default, if method was unspecified, or if the specified method is unvailable."
                  "\n ------------------------------\n")
            return entropy

    def _calculate_attribute_IG(self, col: int):
        """
        This method calculates the information gain for a single attribute and returns the calculated value.
        :param col: the index of the attribute we want to get the information gain for
        :return:
        """
        unique_values = np.unique(self.S, self.S[:, col])
        total_gain = 0

        for value in unique_values:
            subset_data = self.S[self.S[:, col] == value]
            subset_impurity = self.calc_impurity(subset_data[:, -1])
            total_gain += (len(subset_data) / len(self.S)) * subset_impurity

        attribute_IG = self.init_impurity - total_gain
        print("\t The impurity of attribute x" + str(col) + "is " + attribute_IG +
              "\n ------------------------------\n")
        return attribute_IG

    def _calculate_all_attribute_IGs(self):
        """
        This function calculates all IGs for all attributes in S.
        """
        X = self.S[:, :-1]
        for S_i in X:
            print("Calculating all attribute IGs in the set:")
            self.attribute_IGs.append(self._calculate_attribute_IG(S_i))

    def get_information_gain(self):
        """
        A "getter" method for the IG calcualted by this class.
        :return:
        """
        return self.IG
