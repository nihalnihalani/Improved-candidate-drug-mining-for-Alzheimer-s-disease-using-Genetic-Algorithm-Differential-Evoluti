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
#**********************************************************************************************       
    def processData(self):
        self.descriptors, self.targets = self.removeInvalidData(self.descriptors, self.targets)
        self.descriptors, self.active_descriptors = self.removeNearConstantColumns(self.descriptors)
        # Rescale the descriptor data
        self.descriptors = self.rescale_data(self.descriptors)
        #sort data
        self.descriptors, self.targets = self.sort_descriptor_matrix(self.descriptors, self.targets)
        
#**********************************************************************************************       
    def processData(self):
        self.descriptors, self.targets = self.removeInvalidData(self.descriptors, self.targets)
        self.descriptors, self.active_descriptors = self.removeNearConstantColumns(self.descriptors)
        # Rescale the descriptor data
        self.descriptors = self.rescale_data(self.descriptors)
        #sort data
        self.descriptors, self.targets = self.sort_descriptor_matrix(self.descriptors, self.targets)
        return self.X_Train, self.X_Valid, self.X_Test, self.Y_Train, self.Y_Valid, self.Y_Test, self.data
#**********************************************************************************************
# Set up the demonstration model
    def setUpDemoModel(self):
        # featured_descriptors = [4, 8, 12, 16]  # These indices are "false", applying only to the truncated post-filter descriptor matrix.
        binary_model = zeros((50, 593))
        count = 0

        for i in range(50):
          for j in range(593):
              r = random.randint(0, 593)
              L = int(0.015 * 593)

              if r < L:
                  binary_model[i][j] = 1
                  count += 1
          if count > 5 and count < 25:
              continue
          else:
              i -= 1

        # self.binary_model = zeros((1, self.X_Train.shape[1]))
        # self.binary_model[0][featured_descriptors] = 1
        
        
