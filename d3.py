"""
This file calls my ID3 python implemenation in the ID3 directory.
"""
import os
import subprocess
from urllib.parse import urlparse
import numpy as np
from ID3 import D3 


def run(data_url: str, data_has_header: bool, max_depth: int = 6, IG_method: str = ""):
    """
    This is the main method for a user to provide a data url and any specific preferences for running ID3.
    :param data_url: the url to fetch the data
    :param data_has_header: whether the data has a header row (so it can be skipped before running ID3.
    :param max_depth: (optional) the max depth of the result; default is 6.
    :param IG_method: (optional) the method used to calculate uncertainty/impurity (defualt is entropy)
    :return:
    """

    #get the data from a url
    subprocess.run(["wget", data_url], check=False)
    url_path = urlparse(data_url).path
    filename = os.path.basename(url_path)
    # Run ID3
    if data_has_header:
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        return D3.ID3(data, max_depth, IG_method)
    else:
        data = np.genfromtxt(filename, delimiter=',')
        return D3.ID3(data, max_depth, IG_method)
    
def __main__(url):
    tree = d3(url, True)

