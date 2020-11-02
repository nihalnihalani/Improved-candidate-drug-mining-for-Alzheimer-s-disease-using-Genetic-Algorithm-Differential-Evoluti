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
            
#**********************************************************************************************

    def getValidRow(self):
        numDescriptors=self.X_Train.shape[1]
        validRow = zeros((1,numDescriptors))
        count = 0
        while (count < 5) or (count > 25):
            
            for i in range(numDescriptors):
                rand = round(random.uniform(0,100),2)
                if rand < 1.5:
                    validRow[0][i] = 1
                    count +=1
        return validRow

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

    #           a = order[i]
    #           binary_model2[i] = population[a]

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
    #               V[j] = math.floor(abs(oldPopulation[a,j] + (F * (oldPopulation[b,j]-oldPopulation[c,j]))))

    #               rand_num = random.uniform(0, 1)
    #               if (rand_num < CV):
    #                   population[i, j] = V[j]
    #               else:
    #                   continue

    #         for i in range(0,50):
    #             check  = 0
    #             for j in range(0,593):
    #                 if population[i, j] == 1:
    #                     check += 1

    #             if check < 5 or check > 25:
    #                 population[i] = self.getValidRow()
    #         # print(population)        

#**********************************************************************************************

  # def BPSO(self, regressor, instructions, numGenerations, fileW, data):

  #       def create_initial_velocity(velocity):
  #           for i in range(50):
  #               for j in range(593):
  #                   velocity[i][j] = random.uniform(0, 1)

  #           return velocity

    def DE_BinaryParticleSwarmOptimization(self, regressor, instructions, numGenerations, fileW, data):



        def initial_velocity(velocity):
            for i in range(50): 
                for j in range(593):
                    velocity[i][j] = random.uniform(0, 1)
            return velocity

#**********************************************************************************************

        def initial_population(velocity):
            population = np.zeros((50, 593))
            Lambda=0.01
            for i in range(50):
                for j in range(593):
                    if velocity[i][j] <= Lambda:
                        population[i][j] = 1
                    else:
                        population[i][j] = 0

            return population

#**********************************************************************************************
        
        def create_initial_local_best_matrix(population, fitness):

            fitness = list(fitness.values())
            local_best_matrix = population
            local_fitness = fitness
            return local_best_matrix, local_fitness

#**********************************************************************************************

        def create_initial_global_best_row(local_best_matrix, local_fitness):
            global global_best_row
            global global_best_row_fitness

            # global_best_row = np.zeros(593)
            global_best_row_fitness = 200.00

            idx = local_fitness.index(min(local_fitness))

            if local_fitness[idx] < global_best_row_fitness:
                global_best_row = local_best_matrix[idx]
                global_best_row_fitness = local_fitness[idx]

                print('New Global best row has fitness: ', global_best_row_fitness)


            return global_best_row, global_best_row_fitness  

#**********************************************************************************************
       
        def UpdateNewLocalBestMatrix(population, fitness, local_best_matrix, local_fitness):


            fitness = list(fitness.values())

            for i in range(50):
                if fitness[i] < local_fitness[i]:
                    local_best_matrix[i] = population[i]

                    # update the local fitness since local best matrix was changed
                    local_fitness[i] = fitness[i]
            return local_best_matrix, local_fitness

#**********************************************************************************************
        
        def update_global_best_row(local_best_matrix, local_fitness):
            global global_best_row
            global global_best_row_fitness

            idx = local_fitness.index(min(local_fitness))

            if local_fitness[idx] < global_best_row_fitness:
                global_best_row = local_best_matrix[idx]
                global_best_row_fitness = local_fitness[idx]

                print('New Update in global best row with fitness: ', global_best_row_fitness)
            return global_best_row,global_best_row_fitness

#**********************************************************************************************

        def create_new_population(population, velocity, initial_local_best_matrix, alpha):
            new_population = np.zeros((50, 593))
            p = 0.5*(1+alpha)
            beta = 0.004
            for i in range(50):
                
                for j in range(593):
                    if (alpha < velocity[i,j] and velocity[i,j] <= p):
                        new_population[i,j] = initial_local_best_matrix[i,j]
                    elif (p < velocity[i,j] and velocity[i,j] <= (1-beta)):
                        new_population[i,j] = global_best_row[j]
                    elif ((1-beta) < velocity[i,j]) and (velocity[i,j] <=1):
                        new_population[i,j] = 1 - population[i,j]
                    else:
                        new_population[i,j] = population[i,j]
                if self.isValidRow(new_population[i]) == False:
                    new_population[i] = self.getValidRow()

                
            return new_population
#**********************************************************************************************

        def new_velocity(velocity, population):
            F = 0.7
            for i in range(50):
                velocity[i] = population[i]
                
                a = random.randint(0,49)
                b = random.randint(0,49)
                c = random.randint(0,49)
              
                for j in range(0, 593):
                    velocity[i,j] = math.floor(abs(population[c, j] + (F * (population[b, j] - population[a, j]))))

                CR = 0.7
                
                Random = random.uniform(0, 1)
                if (Random < CR):
                    velocity[i,j] = population[i,j]
                else:
                    continue
            return velocity

#**********************************************************************************************       

        def update_population(population, velocity, init_local_best_matrix, local_fitness, regressor, instructions,data,fileW, numGenerations):
            alpha = 0.5

            for i in range(1,numGenerations+1):
                
                velocity = new_velocity(velocity, population)             
                population = create_new_population(population, velocity, init_local_best_matrix, alpha)
                
                self.trackDesc, trackFitness = self.evaluate_population(model = regressor, instructions = instructions, data = self.data, population = population, exportfile = fileW)

                init_local_best_matrix, local_fitness = UpdateNewLocalBestMatrix(population, trackFitness, init_local_best_matrix, local_fitness)
                global_best_row, global_best_row_fitness =  update_global_best_row(init_local_best_matrix, local_fitness)
                
                alpha = alpha - (0.17 / numGenerations)
                
                print(f'End of generation number: {i}')
        
#**********************************************************************************************
        
        fileW.writerow(['Descriptor ID', 'Fitness', 'Algorithm', 'Dimention', 'R2_Train', 'R2_Valid', 'R2_Test', 'RMSE', 'MAE', 'Accuracy'])
        
        velocity = zeros((50,self.X_Train.shape[1]))
        velocity = initial_velocity(velocity)
        population = initial_population(velocity)        
        

        self.trackDesc, self.trackFitness  = self.evaluate_population(model=regressor, instructions=instructions, data=self.data, population=population, exportfile=fileW)
 
        global_best_row = np.zeros(593)
        global_best_row_fitness = 2000
        
        
        init_local_best_matrix, init_local_fitness = create_initial_local_best_matrix(population, self.trackFitness)
        global_best_row, global_best_row_fitness = create_initial_global_best_row(init_local_best_matrix, init_local_fitness)  
        #this is the main recurring function      
        update_population(population, velocity, init_local_best_matrix, init_local_fitness, \
                                        regressor, instructions, self.data, fileW, numGenerations)
                                        
        

#**********************************************************************************************        
    def outputModelInfo(self,trackDesc, trackFitness, trackModel, trackDimen, trackR2train, trackR2valid, trackR2test, testRMSE, testMAE, testAccPred): 
        print("\n\nFitness\t\tDimension\t\t\tR_SquareTrain\t\tR_SquareValid\t\tRMSE\t\tDescriptors")
        print("========================================================================")
        
        for key in trackDesc.keys():
            print(str(trackFitness[key]) + "\t\t" + str(trackDimen[key]) + "\t\t\t\t\t" + str(trackR2train[key])             + "\t\t\t\t" + str(trackR2valid[key]) + "\t\t\t\t" + str(testRMSE[key]) + "\t\t" + str(trackDesc[key]))
            
#**********************************************************************************************
# try to optimize this code if possible
    def open_descriptor_matrix(self,fileName):
        preferred_delimiters = [';', '\t', ',', '\n']

        preferred_delimiters = [';', '\t', ',', '\n']
        file=fileName
        dataArray=np.genfromtxt(file,delimiter=',')

        if (min(dataArray.shape) == 1):  # flatten arrays of one row or column
            return dataArray.flatten(order='C')
        else:
            return dataArray

#************************************************************************************
#Try to optimize this code if possible
    def open_target_values(self,fileName):
        preferred_delimiters = [';', '\t', ',', '\n']

        with open(fileName, mode='r') as csvfile:
            # Dynamically determining the delimiter used in the input file
            row = csvfile.readline()
            delimit = ','
            for d in preferred_delimiters:
                if d in row:
                    delimit = d
                    break

            csvfile.seek(0)
            datalist = csvfile.read().split(delimit)
            if ' ' in datalist:
                datalist = datalist[0].split(' ')

        for i in range(datalist.__len__()):
            datalist[i] = datalist[i].replace('\n', '')
            try:
                datalist[i] = float(datalist[i])
            except:
                datalist[i] = datalist[i]

        try:
            datalist.remove('')
        except ValueError:
            no_empty_strings = True

        return datalist  
#**********************************************************************************************   
    def removeInvalidData(self, descriptors, targets):
        # Numpy to df and series
        descriptors_df = pd.DataFrame(descriptors)
        targets_series = pd.Series(targets)

        # Junk to NaN
        descriptors_df = descriptors_df.apply(pd.to_numeric, errors='coerce')

        # Get indexes of rows with any NaN values
        descriptor_rows_with_nan = [index for index, row in descriptors_df.iterrows() if row.isnull().any()]

        # Drop rows with any NaN values
        descriptors_df = descriptors_df.drop(descriptor_rows_with_nan)
        targets_series = targets_series.drop(descriptor_rows_with_nan)
        delCount = len(descriptor_rows_with_nan)
        print("Dropped ", delCount, " rows containing any junk values.")

        # Drop columns that have more than 20 junks
        numJunkPerCol_Series = descriptors_df.isna().sum()
        delCount = numJunkPerCol_Series[numJunkPerCol_Series > 20].count()
        descriptors_df = descriptors_df.drop(numJunkPerCol_Series[numJunkPerCol_Series > 20].index, axis=1)
        print("Dropped ", delCount, " columns containing more than 20 junk values.")

        # change NaN to 0
        print("Converting remaining junk values to 0...")
        descriptors_df = descriptors_df.fillna(0)

        # drop columns containing all zeros
        tempLen = len(descriptors_df.columns)
        descriptors_df = descriptors_df.loc[:, descriptors_df.ne(0).any(axis=0)]
        delCount = tempLen - len(descriptors_df.columns)
        print("Dropped ", delCount, " columns containing all zeros.")

        # df and series to numpy for return
        descriptors = descriptors_df.to_numpy()
        targets = targets_series.to_numpy()

        return descriptors, targets
    
#**********************************************************************************************
    # Removes constant and near-constant descriptors.
    # But I think also does that too for real data.
    # So for now take this as it is

