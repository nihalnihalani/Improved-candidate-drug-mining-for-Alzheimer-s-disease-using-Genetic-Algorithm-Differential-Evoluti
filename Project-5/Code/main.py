## nihalnihalani google colab code


import numpy
from numpy import *
import pandas as pd
import csv
import hashlib
import re
from sklearn import *
import random
import sklearn.utils
import numpy as np
import os
from math import sqrt
import math
import time
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from sklearn import linear_model, svm, neural_network


class DrugDiscovery:
    descriptors = None
    targets = None 
    active_descriptors = None  
    X_Train = None
    X_Valid = None  
    X_Test = None
    Y_Train = None
    Y_Valid = None
    Y_Test = None
    Data = None

    def __init__(self, descriptors_file, targets_file):
        self.descriptors = self.open_descriptor_matrix(descriptors_file)
        self.targets = self.open_target_values(targets_file)