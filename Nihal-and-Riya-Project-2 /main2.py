import numpy as np
from numpy import zeros
from sklearn import linear_model, svm, neural_network

import os
import csv
import process_input
import fitting_scoring
import random

# ------------------------------------------------------------------------------------------------
descriptors_file = "Practice_Descriptors.csv"
targets_file = "Practice_Targets.csv"

# ------------------------------------------------------------------------------------------------
input = process_input.process()
fit = fitting_scoring.Fitting()

descriptors = input.open_descriptor_matrix(descriptors_file)
targets = input.open_target_values(targets_file)

# ------------------------------------------------------------------------------------------------
# Step 2
# Filter out molecules with NaN-value descriptors and descriptors with little or no variance
descriptors, targets = input.removeInvalidData(descriptors, targets)
descriptors, active_descriptors = input.removeNearConstantColumns(descriptors)
# Rescale the descriptor data
descriptors = input.rescale_data(descriptors)

# ------------------------------------------------------------------------------------------------
# Step 3
descriptors, targets = input.sort_descriptor_matrix(descriptors, targets)
X_Train, X_Valid, X_Test, Y_Train, Y_Valid, Y_Test = input.simple_split(descriptors, targets)
data = {'TrainX': X_Train, 'TrainY': Y_Train, 'ValidateX': X_Valid, 'ValidateY': Y_Valid,
            'TestX': X_Test, 'TestY': Y_Test, 'UsedDesc': active_descriptors}

def MLR():
    print("Multiple Linear Regression")


    print(str(descriptors.shape[1]) + " valid descriptors and " + str(targets.__len__()) + " molecules available.")

    binary_model = zeros((50, X_Train.shape[1]))
    new_population = zeros((4, X_Train.shape[1]))

    max_limit = descriptors.shape[1]

    count = 0
    L = int(0.015 * max_limit)
    mean1 = 10000
    mean2 = 10000

    index_of_mean1 = 0
    index_of_mean2 = 0
    counter = 0

    print("max_limit:", max_limit)

    for i in range(50):
        for j in range(max_limit):
            r = random.randint(0, max_limit)

            if r < L:
                binary_model[i][j] = 1
                count += 1
        if count > 5 and count < 25:
            continue
        else:
            i -= 1

    print(binary_model)

    regressor = linear_model.LinearRegression()
    instructions = {'dim_limit': 4, 'algorithm': 'GA', 'MLM_type': 'MLR'}

    directory = os.path.join(os.getcwd(), 'Outputs')
    output_filename = 'Nihal_MLR.csv'
    file_path = os.path.join(directory, output_filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fileOut = open(file_path, 'w', newline='')  # create stream object for output file
    fileW = csv.writer(fileOut)
    fileW.writerow(
        ['Descriptor ID', 'Fitness', 'Algorithm', 'Dimen', 'R2', 'R2_Valid', 'R2_Test', 'RMSE', 'MAE', 'Prediction'])

    best_accuracy = 0

    for k in range(10000):
        print('Training no:', k)
        regressor.fit(X_Train, Y_Train)

        trackDesc, trackFitness, trackModel, \
        trackDimen, trackR2train, trackR2valid, \
        trackR2test, testRMSE, testMAE, \
        testAccPred = fit.evaluate_population(model=regressor, instructions=instructions, data=data,
                                              population=binary_model, exportfile=fileW)

        counter = 0
        for key in trackDesc.keys():
            if trackFitness[key] < mean1:
                mean2 = mean1
                mean1 = trackFitness[key]
                index_of_mean2 = index_of_mean1
                index_of_mean1 = counter
                print('Previous fitness value is:', mean2)
                print('Updated Fitness value is:', mean1)
                print("Acceptable Predictions From Testing Set:")
                print("\t" + str(100 * testAccPred[key]) + "% of predictions")

                trailacc = 100 * testAccPred[key]
                if best_accuracy < trailacc:
                    bestR2train = trackR2train[key]
                    bestR2test = trackR2test[key]
                    bestR2valid = trackR2valid[key]
                    bestAcc = trailacc

                print('Current best Accuracy is: ', trailacc)

            counter += 1

        old_population = binary_model

        new_population[0] = old_population[index_of_mean1]
        new_population[1] = old_population[index_of_mean2]

        dad = new_population[0]
        mom = new_population[1]
        child1 = zeros(max_limit, dtype=int)
        child2 = zeros(max_limit, dtype=int)

        new_r = random.randint(0, max_limit)

        for j in range(0, new_r):
            child1[j] = dad[j]

        for j in range(new_r, max_limit):
            child1[j] = mom[j]

        for j in range(0, new_r):
            child2[j] = dad[j]

        for j in range(new_r, max_limit):
            child2[j] = mom[j]

        new_population[2] = child1
        new_population[3] = child2

        binary_model = zeros((50, X_Train.shape[1]))
        binary_model = np.concatenate((new_population, binary_model[4:]), axis=0)
        L = int(0.015 * max_limit)

        for num1 in range(4, 50):
            for num2 in range(max_limit):
                X = random.randint(0, max_limit)

                if X < L:
                    binary_model[num1][num2] = 1
                    count += 1
            if count > 5 and count < 25:
                continue
            else:
                num1 -= 1

        for n in range(50):
            for m in range(max_limit):
                X = random.randint(0, 100)

                # So basically we will flip values only if the X is 0. And chances of getting 0 are very less

                if X <= 0.0005 and binary_model[n][m] == 1:
                    binary_model[n][m] = 0
                elif X <= 0.005 and binary_model[n][m] == 0:
                    binary_model[n][m] = 1

 # ------------------------------------------------------------------------------------------------
# Step 8

# ------------------------------------------------------------------------------------------------
# Step 6
# Create a Multiple Linear Regression object to fit our demonstration model to the data
def SVM():
    # ------------------------------------------------------------------------------------------------
    # Step 5
    # Set up the demonstration model


    binary_model = zeros((50,X_Train.shape[1]))
    new_population = zeros((4,X_Train.shape[1]))

    max_limit = descriptors.shape[1]

    count = 0
    L = int(0.015 * max_limit)
    mean1 = 10000
    mean2 = 10000

    index_of_mean1 = 0
    index_of_mean2 = 0
    counter = 0

    print("max_limit:", max_limit)

    for i in range(50):
        for j in range(max_limit):
            r = random.randint(0, max_limit)

            if r < L:
                binary_model[i][j] = 1
                count += 1
        if count > 5 and count < 25:
            continue
        else:
            i -= 1

    print(binary_model)

    regressor = svm.SVR()
    instructions = {'dim_limit': 4, 'algorithm': 'GA', 'MLM_type': 'SVR'}

    directory = os.path.join(os.getcwd(), 'Outputs')
    output_filename = 'Nihal_SVR.csv'
    file_path = os.path.join(directory, output_filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fileOut = open(file_path, 'w', newline='')  # create stream object for output file
    fileW = csv.writer(fileOut)
    fileW.writerow(
        ['Descriptor ID', 'Fitness', 'Algorithm', 'Dimen', 'R2', 'R2_Valid', 'R2_Test', 'RMSE', 'MAE', 'Prediction'])

    best_accuracy = 0

    for k in range(10000):
        print('Training')
        regressor.fit(X_Train, Y_Train)

        trackDesc, trackFitness, trackModel,trackDimen, trackR2train, trackR2valid,trackR2test, testRMSE, testMAE,testAccPred = fit.evaluate_population(model=regressor, instructions=instructions, data=data,population=binary_model, exportfile=fileW)

        counter = 0
        for key in trackDesc.keys():
            if trackFitness[key] < mean1:
                mean2 = mean1
                mean1 = trackFitness[key]
                index_of_mean2 = index_of_mean1
                index_of_mean1 = counter
                print('Previous fitness value is:', mean2)
                print('Updated Fitness value is:', mean1)
                print("Acceptable Predictions From Testing Set:")
                print("\t" + str(100 * testAccPred[key]) + "% of predictions")

                trailacc = 100 * testAccPred[key]
                if best_accuracy < trailacc:
                    bestR2train = trackR2train[key]
                    bestR2test = trackR2test[key]
                    bestR2valid = trackR2valid[key]
                    bestAcc = trailacc

                print('Current best Accuracy is: ', trailacc)

            counter += 1

        old_population = binary_model

        new_population[0] = old_population[index_of_mean1]
        new_population[1] = old_population[index_of_mean2]

        dad = new_population[0]
        mom = new_population[1]
        child1 = zeros(max_limit, dtype=int)
        child2 = zeros(max_limit, dtype=int)

        new_r = random.randint(0, max_limit)

        for j in range(0, new_r):
            child1[j] = dad[j]

        for j in range(new_r, max_limit):
            child1[j] = mom[j]

        for j in range(0, new_r):
            child2[j] = dad[j]

        for j in range(new_r, max_limit):
            child2[j] = mom[j]

        new_population[2] = child1
        new_population[3] = child2

        binary_model = zeros((50, X_Train.shape[1]))
        binary_model = np.concatenate((new_population, binary_model[4:]), axis=0)
        L = int(0.015 * max_limit)

        for num1 in range(4, 50):
            for num2 in range(max_limit):
                X = random.randint(0, max_limit)

                if X < L:
                    binary_model[num1][num2] = 1
                    count += 1
            if count > 5 and count < 25:
                continue
            else:
                num1 -= 1

        for n in range(50):
            for m in range(max_limit):
                X = random.randint(0, 100)

                # So basically we will flip values only if the X is 0. And chances of getting 0 are very less

                if X <= 0.0005 and binary_model[n][m] == 1:
                    binary_model[n][m] = 0
                elif X <= 0.005 and binary_model[n][m] == 0:
                    binary_model[n][m] = 1
# ------------------------------------------------------------------------------------------------
# Step 8
def ANN():
    print("#########################################################################################")
    print("\nArtificial Neural Network")

    print(str(descriptors.shape[1]) + " valid descriptors and " + str(targets.__len__()) + " molecules available.")

    # print(X_Train[0:5, 0:20])

    # ------------------------------------------------------------------------------------------------
    # Step 5
    # Set up the demonstration model

    binary_model = zeros((50, X_Train.shape[1]))
    new_population = zeros((4, X_Train.shape[1]))

    max_limit = descriptors.shape[1]

    count = 0
    L = int(0.015 * max_limit)
    mean1 = 10000
    mean2 = 10000

    index_of_mean1 = 0
    index_of_mean2 = 0
    counter = 0

    print("max_limit:", max_limit)

    for i in range(50):
        for j in range(max_limit):
            r = random.randint(0, max_limit)

            if r < L:
                binary_model[i][j] = 1
                count += 1
        if count > 5 and count < 25:
            continue
        else:
            i -= 1

    print(binary_model)

    regressor = neural_network.MLPRegressor(hidden_layer_sizes=(50, 8, 8))
    instructions = {'dim_limit': 4, 'algorithm': 'None', 'MLM_type': 'ANN'}

    # regressor = linear_model.LinearRegression()
    # instructions = {'dim_limit': 4, 'algorithm': 'None', 'MLM_type':'MLR'}

    directory = os.path.join(os.getcwd(), 'Outputs')
    output_filename = 'Nihal_MLR.csv'
    file_path = os.path.join(directory, output_filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fileOut = open(file_path, 'w', newline='')  # create stream object for output file
    fileW = csv.writer(fileOut)
    fileW.writerow(
        ['Descriptor ID', 'Fitness', 'Algorithm', 'Dimen', 'R2', 'R2_Valid', 'R2_Test', 'RMSE', 'MAE', 'Prediction'])

    best_accuracy = 0

    for k in range(1000):
        print('Training')
        regressor.fit(X_Train, Y_Train)

        trackDesc, trackFitness, trackModel, \
        trackDimen, trackR2train, trackR2valid, \
        trackR2test, testRMSE, testMAE, \
        testAccPred = fit.evaluate_population(model=regressor, instructions=instructions, data=data,
                                              population=binary_model, exportfile=fileW)

        counter = 0
        for key in trackDesc.keys():
            if trackFitness[key] < mean1:
                mean2 = mean1
                mean1 = trackFitness[key]
                index_of_mean2 = index_of_mean1
                index_of_mean1 = counter
                print('Previous fitness value is:', mean2)
                print('Updated Fitness value is:', mean1)
                print("Acceptable Predictions From Testing Set:")
                print("\t" + str(100 * testAccPred[key]) + "% of predictions")

                trailacc = 100 * testAccPred[key]
                if best_accuracy < trailacc:
                    bestR2train = trackR2train[key]
                    bestR2test = trackR2test[key]
                    bestR2valid = trackR2valid[key]
                    bestAcc = trailacc

                print('Current best Accuracy is: ', trailacc)

            counter += 1

        old_population = binary_model

        new_population[0] = old_population[index_of_mean1]
        new_population[1] = old_population[index_of_mean2]

        dad = new_population[0]
        mom = new_population[1]
        child1 = zeros(max_limit, dtype=int)
        child2 = zeros(max_limit, dtype=int)

        new_r = random.randint(0, max_limit)

        for j in range(0, new_r):
            child1[j] = dad[j]

        for j in range(new_r, max_limit):
            child1[j] = mom[j]

        for j in range(0, new_r):
            child2[j] = dad[j]

        for j in range(new_r, max_limit):
            child2[j] = mom[j]

        new_population[2] = child1
        new_population[3] = child2

        binary_model = zeros((50, X_Train.shape[1]))
        binary_model = np.concatenate((new_population, binary_model[4:]), axis=0)
        L = int(0.015 * max_limit)

        for num1 in range(4, 50):
            for num2 in range(max_limit):
                X = random.randint(0, max_limit)

                if X < L:
                    binary_model[num1][num2] = 1
                    count += 1
            if count > 5 and count < 25:
                continue
            else:
                num1 -= 1

        for n in range(50):
            for m in range(max_limit):
                X = random.randint(0, 100)

                # So basically we will flip values only if the X is 0. And chances of getting 0 are very less

                if X <= 0.0005 and binary_model[n][m] == 1:
                    binary_model[n][m] = 0
                elif X <= 0.005 and binary_model[n][m] == 0:
                    binary_model[n][m] = 1



    

while True:
    MLR()
    SVM()
    ANN()
    break








