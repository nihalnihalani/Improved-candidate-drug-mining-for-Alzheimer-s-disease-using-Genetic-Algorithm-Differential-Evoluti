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
    #         order = []

    #         order=df2['order'].values.tolist()
    #         binary_model2 = population.copy()
    #         for i in range(len(order)):

    #         	a = order[i]
    #         	binary_model2[i] = population[a]

    #         population = binary_model2
    #         oldPopulation = population
    #         F = 0.7
    #         CV = 0.7
    #         for i in range(1, 50):


    #             V = zeros(593)
    #             a = random.randint(0, 49)
    #             b = random.randint(0, 49)
    #             c = random.randint(0, 49)


    #             for j in range(self.X_Train.shape[1]):
    #             	V[j] = math.floor(abs(oldPopulation[a,j] + (F * (oldPopulation[b,j]-oldPopulation[c,j]))))

    #             	rand_num = random.uniform(0, 1)
    #             	if (rand_num < CV):
    #             		population[i, j] = V[j]
    #             	else:
    #             		continue

    #         for i in range(0,50):
    #             check  = 0
    #             for j in range(0,593):
    #                 if population[i, j] == 1:
    #                     check += 1

    #             if check < 5 or check > 25:
    #                 population[i] = self.getValidRow()
    #         # print(population)
    def removeNearConstantColumns(self, data_matrix, num_unique=10):
    	useful_descriptors = [col for col in range(data_matrix.shape[1])
                              if len(set(data_matrix[:, col])) > num_unique]
        filtered_matrix = data_matrix[:, useful_descriptors]
        remaining_desc = zeros(data_matrix.shape[1])
        remaining_desc[useful_descriptors] = 1

        return filtered_matrix, where(remaining_desc == 1)[0]


    #**********************************************************************************************
    def rescale_data(self, descriptor_matrix):
        # Statistics for dataframe
        df = pd.DataFrame(descriptor_matrix)
        rescaled_matrix = (df - df.values.mean()) / (df.values.std())
        print("Rescaled Matrix is: ")
        rescaled_matrix.to_csv("rescaledmatrix.csv")
        print(rescaled_matrix)
        return rescaled_matrix
    #**********************************************************************************************
    def sort_descriptor_matrix(self, descriptors, targets):
    	# Placing descriptors and targets in ascending order of target (IC50) value.
        alldata = ndarray((descriptors.shape[0], descriptors.shape[1] + 1))
        alldata[:, 0] = targets
       	alldata[:, 1:alldata.shape[1]] = descriptors
        alldata = alldata[alldata[:, 0].argsort()]
        descriptors = alldata[:, 1:alldata.shape[1]]
        targets = alldata[:, 0]
        return descriptors, targets
    #**********************************************************************************************
    # Performs a simple split of the data into training, validation, and testing sets.
    # So how does it relate to the Data Mining Prediction?
    def simple_split(self, descriptors, targets):

        testX_indices = [i for i in range(descriptors.shape[0]) if i % 4 == 0]
        validX_indices = [i for i in range(descriptors.shape[0]) if i % 4 == 1]
        trainX_indices = [i for i in range(descriptors.shape[0]) if i % 4 >= 2]

     	TrainX = descriptors[trainX_indices, :]
        ValidX = descriptors[validX_indices, :]
        TestX = descriptors[testX_indices, :]

        TrainY = targets[trainX_indices]
        ValidY = targets[validX_indices]
        TestY = targets[testX_indices]

        return TrainX, ValidX, TestX, TrainY, ValidY, TestY

    def evaluate_population(self, model, instructions, data, population, exportfile):
    	numOfPop = population.shape[0]
        fitness = zeros(numOfPop)
        predictive = 0

        TrainX = data['TrainX']
        TrainY = data['TrainY']
        ValidateX = data['ValidateX']
        ValidateY = data['ValidateY']
        TestX = data['TestX']
        TestY = data['TestY']
        UsedDesc = data['UsedDesc']

        trackDesc, trackFitness, trackModel, trackDimen, trackR2, trackR2PredValidation, \
        trackR2PredTest, trackRMSE, trackMAE, trackAcceptPred, trackCoefficients = self.InitializeTracks()

        unfit = 1000

        for i in range(numOfPop):

            xi = list(where(population[i] == 1)[0])


            idx = hashlib.sha1(array(xi)).digest()

            # Condenses binary models to a list of the indices of active features
            X_train_masked = TrainX.T[xi].T
            X_validation_masked = ValidateX.T[xi].T
            X_test_masked = TestX.T[xi].T

            try:
                model = model.fit(X_train_masked, TrainY)
            except:
                return unfit, fitness


            # Computed predicted values
            Yhat_training = model.predict(X_train_masked)
            Yhat_validation = model.predict(X_validation_masked)
            Yhat_testing = model.predict(X_test_masked)

            # Compute R2 scores (Prediction for Validation and Test set)
            r2_train = model.score(X_train_masked, TrainY)
            r2validation = model.score(X_validation_masked, ValidateY)
            r2test = model.score(X_test_masked, TestY)
            model_rmse, num_acceptable_preds = self.calculateRMSE(TestY, Yhat_testing)
            model_mae = self.calculateMAE(TestY, Yhat_testing)

            # Calculating fitness value
            if 'dim_limit' in instructions:
                fitness[i] = self.get_fitness(xi, TrainY, ValidateY, Yhat_training, Yhat_validation, dim_limit=instructions['dim_limit'])
            else:
                fitness[i] = self.get_fitness(xi, TrainY, ValidateY, Yhat_training, Yhat_validation)

            if predictive and ((r2validation < 0.5) or (r2test < 0.5)):
                # if it's not worth recording, just return the fitness
                #print("Ending program, fitness unacceptably low: ", predictive)
                continue

            # store stats
            if fitness[i] < unfit:

                # store stats
                trackDesc[idx] = (re.sub(",", "_", str(xi)))  # Editing descriptor set to match actual indices in the original data.
                trackDesc[idx] = (re.sub(",", "_", str(UsedDesc[xi].tolist())))  # Editing descriptor set to match actual indices in the original data.

                trackFitness[idx] = self.sigfig(fitness[i])
                trackModel[idx] = instructions['algorithm'] + ' with ' + instructions['MLM_type']
                trackDimen[idx] = int(xi.__len__())

                trackR2[idx] = self.sigfig(r2_train)
                trackR2PredValidation[idx] = self.sigfig(r2validation)
                trackR2PredTest[idx] = self.sigfig(r2test)

                trackRMSE[idx] = self.sigfig(model_rmse)
                trackMAE[idx] = self.sigfig(model_mae)
                trackAcceptPred[idx] = self.sigfig(float(num_acceptable_preds) / float(Yhat_testing.shape[0]))

            # For loop ends here.

        self.write(exportfile, trackDesc, trackFitness, trackModel, trackDimen, trackR2, trackR2PredValidation, trackR2PredTest, trackRMSE, trackMAE, trackAcceptPred)

        return trackDesc, trackFitness

    #**********************************************************************************************
    def sigfig(self, x):
        return float("%.4f"%x)

    #**********************************************************************************************
    def InitializeTracks(self):
        trackDesc = {}
        trackFitness = {}
        trackAlgo = {}
        trackDimen = {}
        trackR2 = {}
        trackR2PredValidation = {}
        trackR2PredTest = {}
        trackRMSE = {}
        trackMAE = {}
        trackAcceptPred = {}
        trackCoefficients = {}

        return trackDesc, trackFitness, trackAlgo, trackDimen, trackR2, trackR2PredValidation, trackR2PredTest, \
               trackRMSE, trackMAE, trackAcceptPred, trackCoefficients
    #**********************************************************************************************
    def get_fitness(self, xi, T_actual, V_actual, T_pred, V_pred, gamma=3, dim_limit=None, penalty=0.05):
        n = len(xi)
        mT = len(T_actual)
        mV = len(V_actual)

        train_errors = [T_pred[i] - T_actual[i] for i in range(T_actual.__len__())]
        RMSE_t = sum([element**2 for element in train_errors]) / mT
        valid_errors = [V_pred[i] - V_actual[i] for i in range(V_actual.__len__())]
        RMSE_v = sum([element**2 for element in valid_errors]) / mV

        numerator = ((mT - n - 1) * RMSE_t) + (mV * RMSE_v)
        denominator = mT - (gamma * n) - 1 + mV
        fitness = sqrt(numerator/denominator)
        # Adjusting for high-dimensionality models.
        if dim_limit is not None:
            if n > int(dim_limit * 1.5):
                fitness += ((n - dim_limit) * (penalty * dim_limit))
            elif n > dim_limit:
                fitness += ((n - dim_limit) * penalty)

        return fitness

    def calculateMAE(self, experimental, predictions):
    	errors = [abs(experimental[i] - predictions[i]) for i in range(experimental.__len__())]
        return sum(errors) / experimental.__len__()
    #**********************************************************************************************
    def calculateRMSE(self, experimental, predictions):
        sum_of_squares = 0
        errors_below_1 = 0
        for mol in range(experimental.__len__()):
      








