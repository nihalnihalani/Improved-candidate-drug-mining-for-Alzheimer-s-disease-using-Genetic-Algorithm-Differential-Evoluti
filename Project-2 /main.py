import os
import csv
from process_input import process
import numpy as np
import random
import pandas as pd
from numpy import zeros
from sklearn import linear_model
from fitting_scoring import Fitting
from sklearn import neural_network
from sklearn import svm



class MainDatamining:

    def __init__(self) -> object:
        descriptors_file = "Practice_Descriptors.csv"
        targets_file = "Practice_Targets.csv"
        self.process_input = process()
        self.fitting_scoring = Fitting()
        self.descriptors = self.process_input.open_descriptor_matrix(descriptors_file)
        self.targets = self.process_input.open_target_values(targets_file)
        self.x_train = self.x_valid = self.x_test = self.y_train = self.y_valid = self.y_test = pd.DataFrame()
        self.new_population = self.max_limit = None
        self.child1 = 10000
        self.child2 = 10000
        self.indexchild1 = 0
        self.indexchild2 = 0
        self.fileW = None
        self.count = None
        self.binary_model = pd.DataFrame()
        self.data = pd.DataFrame()

    #---------------------------------------------------------------------------------------------------------

    def R_descriptors(self):
        self.descriptors, self.targets = self.process_input.removeInvalidData(self.descriptors, self.targets)
        self.descriptors, self.active_descriptors = self.process_input.removeNearConstantColumns(self.descriptors)
        # Rescale the descriptor data
        self.descriptors = self.process_input.rescale_data(self.descriptors)

    #-----------------------------------------------------------------------

    def Sort_Split_Descriptor_Matrix(self):
        self.descriptors, self.targets = self.process_input.sort_descriptor_matrix(self.descriptors, self.targets)

        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = self.process_input.simple_split(
            self.descriptors, self.targets)
        self.data = {'TrainX': self.x_train, 'TrainY': self.y_train, 'ValidateX': self.x_valid,
                     'ValidateY': self.y_valid,
                     'TestX': self.x_test, 'TestY': self.y_test, 'UsedDesc': self.active_descriptors}

        print(str(self.descriptors.shape[1]) + "(Columns) valid descriptors and " + str(
            self.targets.__len__()) + " (Rows) molecules available.")

    #-------------------------------------------------------



    def set_model(self):
        # using a loop to get the first 50 models
        self.binary_model = zeros((50, self.x_train.shape[1]))
        self.new_population = zeros((4, self.x_train.shape[1]))
        self.length = self.descriptors.shape[1]
        self.cou = 0
        l = (0.015 * self.length)
        print("length:", self.length)

        for i in range(50):
            for j in range(self.length):
                r = random.randint(0, self.length)
                if r < l:
                    self.binary_model[i][j] = 1
                    self.cou += 1
            if self.cou > 5 and self.cou < 25:
                continue
            else:
                i -= 1
            print("binary_model:\n", self.binary_model)
    #----------------------------------------------------------------------------

    def mlr_run(self):
        regressor = linear_model.LinearRegression()
        instructions = {'dim_limit': 4, 'algorithm': 'None', 'MLM_type': 'MLR'}
        self.file(instructions, regressor)

    #---------------------------------------------------------------------------

    def svm_run(self):
        print("support Vector Machine")
        svr_regressor = svm.SVR()
        instructions = {'dim_limit': 4, 'algorithm': 'None', 'MLM_type': 'SVR'}
        self.file(instructions, svr_regressor)
    #---------------------------------------------------------

    def ann_run(self):
        print("Artificial neural Network")
        ann_regressor = neural_network.MLPRegressor(hidden_layer_sizes=(556, 5, 5))
        instructions = {'dim_limit': 4, 'algorithm': 'None', 'MLM_type': 'ANN'}
        self.file(instructions, ann_regressor)

    #-------------------------------------------------------------------------------

    def file(self, instructions, regressor):
        directory = os.path.join(os.getcwd(), 'Outputs')
        output_filename = 'mlr1_Output.csv'
        file_path = os.path.join(directory, output_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        fileOut = open(file_path, 'w', newline='')  # create stream object for output file

        self.fileW = csv.writer(fileOut)
        self.fileW.writerow(
            ['Descriptor ID', 'Fitness', 'Algorithm', 'Dimen', 'R2_Train', 'R2_Valid', 'R2_Test', 'RMSE', 'MAE',
             'Prediction'])

        ideal_accuracy = 0

        for val in range(10000):
            print('val is: ', val)
            regressor.fit(self.x_train, self.y_train)

            trackDesc, trackFitness, trackModel, trackDimen, trackR2train, trackR2valid, \
            trackR2test, testRMSE, testMAE, \
            testAccPred = self.fitting_scoring.evaluate_population(model=regressor, instructions=instructions,
                                                                   data=self.data,
                                                                   population=self.binary_model, exportfile=self.fileW)
            counter = 0
            for k in trackDesc.keys():
                if trackFitness[k] < 1:
                    self.child2 = self.child1
                    self.child1 = trackFitness[k]
                    self.indexchild2 = self.indexchild1
                    self.indexchild1 = counter

                    trailacc = 100 * testAccPred[k]

                    if ideal_accuracy < trailacc:
                        bestR2train = trackR2train[k]
                        bestR2test = trackR2test[k]
                        bestR2valid = trackR2valid[k]
                        bestAcc = trailacc

                counter += 1

            old_population = self.binary_model

            self.new_population[0] = old_population[self.indexchild1]
            self.new_population[1] = old_population[self.indexchild2]

            dad = self.new_population[0]
            mom = self.new_population[1]
            child1 = zeros(self.length, dtype=int)
            child2 = zeros(self.length, dtype=int)

            new_r = random.randint(0, self.length)

            for j in range(0, new_r):
                child1[j] = dad[j]
            for j in range(0, new_r):
                child2[j] = dad[j]

            for j in range(new_r, self.length):
                child1[j] = mom[j]
                child2[j] = mom[j]

            self.new_population[2] = self.child1
            self.new_population[3] = self.child2

            binary_model = zeros((50, self.x_train.shape[1]))
            binary_model = np.concatenate((self.new_population, binary_model[4:]), axis=0)
            L = int(0.015 * self.length)

            for no1 in range(4, 50):
                for no2 in range(self.length):
                    X = random.randint(0, self.length)

                    if X < L:
                        binary_model[no1][no2] = 1
                        self.cou += 1
                if self.cou > 5 and self.cou < 25:
                    continue
                else:
                    no1 -= 1

            for n in range(50):
                for m in range(self.length):
                    A = random.randint(0, 100)

                    if A <= 0.0005 and binary_model[n][m] == 1:
                        binary_model[n][m] = 0
                    elif A <= 0.005 and binary_model[n][m] == 0:
                        binary_model[n][m] = 1

        self.__print_result(trackDesc, trackFitness, trackDimen, trackR2train, trackR2valid, testRMSE)

    #-------------------------------------------------------------

    def __print_result(self, trackDesc, trackFitness, trackDimen, trackR2train, trackR2valid, testRMSE):
        print("\n\nFitness\t\tDimension\t\t\tR_SquareTrain\t\tR_SquareValid\t\tRMSE\t\tDescriptors")
        print("==================================================================================================")
        for key in trackDesc.keys():
            print(str(trackFitness[key]) + "\t\t" + str(trackDimen[key]) + "\t\t\t\t\t" + str(trackR2train[key]) \
                  + "\t\t\t\t" + str(trackR2valid[key]) + "\t\t\t\t" + str(testRMSE[key]) + "\t\t" + str())

if __name__ == "__main__":
    dataMining = MainDatamining()

    dataMining.R_descriptors()

    dataMining.Sort_Split_Descriptor_Matrix()

    dataMining.set_model()

    dataMining.mlr_run()
    #dataMining.svm_run()
    #dataMining.ann()
