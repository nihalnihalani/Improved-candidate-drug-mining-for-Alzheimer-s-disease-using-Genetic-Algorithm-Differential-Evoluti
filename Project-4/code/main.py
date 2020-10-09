import numpy
from numpy import *
import pandas as pd
import csv
import hashlib
import re
from sklearn import *
import random
import numpy as np
import os
from math import sqrt
import math
import time
from sklearn.utils.testing import ignore_warnings
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
    binary_model = None

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
    def splitData(self):
        self.X_Train, self.X_Valid, self.X_Test, self.Y_Train, self.Y_Valid, self.Y_Test = self.simple_split(self.descriptors, self.targets)
        self.data = {'TrainX': self.X_Train, 'TrainY': self.Y_Train, 'ValidateX': self.X_Valid, 'ValidateY': self.Y_Valid,
                     'TestX': self.X_Test, 'TestY': self.Y_Test, 'UsedDesc': self.active_descriptors}
        print(str(self.descriptors.shape[1]) + " valid descriptors and " + str(self.targets.__len__()) + " molecules available.")
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
     #**********************************************************************************************
    # Create a Multiple Linear Regression object to fit our demonstration model to the data
    def runModel(self, regressor, instructions):
    	trackDesc, trackFitness, trackModel, trackDimen, trackR2train, trackR2valid, trackR2test, testRMSE, testMAE, testAccPred = self.evaluate_population(model=regressor, instructions=instructions, data=self.data,
                                                                                                                                                            population=self.binary_model, exportfile=None)
        self.outputModelInfo(trackDesc, trackFitness, trackModel, trackDimen, trackR2train, trackR2valid, trackR2test, testRMSE, testMAE, testAccPred)

    #**********************************************************************************************
    def isValidRow(self, row):
        count = 0
        for value in row:
            if value == 1:
                count += 1
            if count < 5 or count > 25:
                return False
    # def geneticModel(self, regressor, instructions, numGenerations, fileW):

    #     fileW.writerow(['Descriptor ID', 'Fitness', 'Algorithm', 'Dimen', 'R2_Train', 'R2_Valid', 'R2_Test', 'Q2', 'RMSE', 'MAE', 'Abs. Errors'])

    #     population = zeros((50,self.X_Train.shape[1]))

    #     for i in range(50):
    #         population[i] = self.getValidRow()


    #     for generation in range(numGenerations):
    #         print("Generation: ", generation)

    #         trackDesc, trackFitness, trackModel, \
    #         trackDimen, trackR2train, trackR2valid, \
    #         trackR2test, testRMSE, testMAE, \
    #         testAccPred = self.evaluate_population(model=regressor, instructions=instructions, data=self.data,population=population, exportfile=fileW)

    #         oldPopulation = population.copy()
    #         population = zeros((50, self.X_Train.shape[1]))

    #         dad, mom = self.generateParents(oldPopulation, trackFitness)
    #         child1, child2 = self.generateChildren(dad, mom)
    #         population[0] = dad
    #         population[1] = mom
    #         population[2] = child1
    #         population[3] = child2
    #         for i in range(4,50):
    #             population[i] = self.getValidRow()
    #         # self.outputModelInfo(trackDesc, trackFitness, trackModel, trackDimen, trackR2train, trackR2valid, trackR2test, testRMSE, testMAE, testAccPred)
    #         population = self.mutatePopulation(population)

    # def DifferentialEvolutionModel(self, regressor, instructions, numGenerations, fileW):

    #     fileW.writerow(['Descriptor ID', 'Fitness', 'Algorithm', 'Dimen', 'R2_Train', 'R2_Valid', 'R2_Test', 'RMSE', 'MAE', 'Abs. Errors'])
    #     population = zeros((50,self.X_Train.shape[1]))

    #     for i in range(50):
    #             population[i] = self.getValidRow()


    #     for generation in range(numGenerations):
    #         print("Epoch ", generation,"/",numGenerations)

    #         trackDesc, trackFitness, trackModel, \
    #         trackDimen, trackR2train, trackR2valid, \
    #         trackR2test, testRMSE, testMAE, \
    #         testAccPred = self.evaluate_population(model=regressor, instructions=instructions, data=self.data,population=population, exportfile=fileW)
    #         counter = 0
    #         dummy = []
    #         trail = []
    #         for i in range(50):
    #             trail.append(i)

    #         for key in trackDesc.keys():
    #             dummy.append(trackFitness[key])

    #         df = pd.DataFrame(dummy)
    #         df.columns = ['fitness']
    #         df1 = pd.DataFrame(trail)
    #         df1.columns = ['order']
    
    #         # print('Now df1!')
    #         # print(df1)
    #         df['order'] = df1



    #         df2 = df.sort_values('fitness')
