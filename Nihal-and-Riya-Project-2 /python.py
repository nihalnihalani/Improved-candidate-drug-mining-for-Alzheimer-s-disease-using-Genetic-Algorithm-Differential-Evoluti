from numpy import zeros
from sklearn import linear_model
import pandas as pd
import fitting_scoring
import process_input
from sklearn import svm
from sklearn import neural_network
import random
import numpy
import os
import csv
from numpy import *
import csv
import numpy as np
import pandas as pd
import math
from pandas import DataFrame

# ------------------------------------------------------------------------------------------------
descriptors_file = "Practice_Descriptors.csv"
targets_file = "Practice_Targets.csv"

input = process_input.process()
fit = fitting_scoring.Fitting()


def function_step1(descriptors_file, targets_file):
    descriptors = input.open_descriptor_matrix(descriptors_file)
    print('Original matrix dimensions : ', descriptors.shape)
    targets = input.open_target_values(targets_file)
    return descriptors, targets


def function_step2(descriptors, targets):
    descriptors, targets = process_input.removeInvalidData(descriptors, targets)
    print()
    print(targets)
    print()
    print('--------------------step 1 ends-------------------------')
    descriptors, active_descriptors = process_input.removeNearConstantColumns(descriptors)

    print('After removing invalid datas descriptor are as follows: ')
    print(descriptors)
    print()
    print('Now descriptor dimensions are ', descriptors.shape)
    # Rescale the descriptor data
    descriptors = process_input.rescale_data(descriptors)
    print('------------------------Rescaled matrix is below--------------------')
    print('Rescaled value of Xes is:')
    print(descriptors)

    print('Rescaled matrix dimenstions are:', descriptors.shape)
    print('------------------------------------------------------')
    return descriptors, targets, active_descriptors


def function_step3(descriptors, targets):
    descriptors, targets = process_input.sort_descriptor_matrix(descriptors, targets)
    return descriptors, targets


def function_step4(descriptors, targets):
    X_Train, X_Valid, X_Test, Y_Train, Y_Valid, Y_Test = process_input.simple_split(descriptors, targets)
    data = {'TrainX': X_Train, 'TrainY': Y_Train, 'ValidateX': X_Valid, 'ValidateY': Y_Valid,
            'TestX': X_Test, 'TestY': Y_Test, 'UsedDesc': active_descriptors}
    print(str(descriptors.shape[1]) + " valid descriptors and " + str(targets.__len__()) + " molecules available.")
    return X_Train, X_Valid, X_Test, Y_Train, Y_Valid, Y_Test, data


def function_step5():
    binary_model = zeros((50, X_Train.shape[1]))

    L = (0.015 * 593)
    '''min1 = 20000
    min2 = 20000
    indexMin1 = indexMin2 = 0
    counter = 0'''

    # ------------------------------------------------------------------------------------------------

    def ValidRow(binary_model):
        for i in range(50):
            cc = 0
            for j in range(593):
                r = random.randint(0, 593)
                if r < L:
                    binary_model[i][j] = 1
                    cc += 1
            if cc < 5 and cc > 25:
                i -= 1
            else:
                continue
        return binary_model

    binary_model = ValidRow(binary_model)

    '''featured_descriptors = [4, 8, 12, 16]  # These indices are "false", applying only to the truncated post-filter descriptor matrix.
    binary_model = zeros((1, X_Train.shape[1]))
    binary_model[0][featured_descriptors] = 1'''
    return binary_model


def function_step6():
    regressor = linear_model.LinearRegression()
    # regressor = svm.SVR()
    # regressor = neural_network.MLPRegressor(hidden_layer_sizes=(200,))

    instructions = {'dim_limit': 4, 'algorithm': 'None', 'MLM_type': 'MLR'}
    for gen in range(50):
        regressor.fit(X_Train, Y_Train)
        print('This is MLR!')
        trackDesc, trackFitness, trackModel, \
        trackDimen, trackR2train, trackR2valid, \
        trackR2test, testRMSE, testMAE, \
        testAccPred = fitting_scoring.evaluate_population(model=regressor, instructions=instructions, data=data,
                                                          population=binary_model, exportfile=None)

        print('========================================================================')
        counter = 0
        shawn = []
        trail = []
        for i in range(50):
            trail.append(i)

        for key in trackDesc.keys():
            shawn.append(trackFitness[key])

        df = pd.DataFrame(shawn)
        df.columns = ['fitness']
        print(df)
        df1 = pd.DataFrame(trail)
        df1.columns = ['order']

        print('Now df1!')
        print(df1)
        df['order'] = df1

        print('now after append')
        print(df)

        df2 = df.sort_values('fitness')
        print('After sorting!')
        print(df2)

        order = []

        order = df2['order'].values.tolist()
        print(order)

        # savetxt('binary_model.csv', binary_model, delimiter=',')

        binary_model2 = binary_model.copy()

        for i in range(len(order)):
            # for j in range(593):
            a = order[i]
            binary_model2[i] = binary_model[a]

        print(binary_model2)
        # savetxt('binary_model2.csv', binary_model2, delimiter=',')

        pop = binary_model2

        Oldpop = pop

        for i in range(1, 50):
            V = numpy.arange(593)

            a = random.randint(1, 49)
            b = random.randint(1, 49)
            c = random.randint(1, 49)

            F = 0.7

            for j in range(0, 593):
                V[j] = math.floor(abs(Oldpop[a, j] + (F * (Oldpop[b, j] - Oldpop[c, j]))))

            CV = 0.7
            CV = random.randint(0, 1)

            for k in range(0, 593):
                Random = random.uniform(0, 1)
                if (Random < CV):
                    pop[i, k] = V[k]
                else:
                    continue

        print(pop.shape)

        def ValidRow(binary_model):
            L = (0.015 * 593)
            while True:
                cc = 0
                for j in range(593):
                    r = random.randint(0, 593)
                    if r < L:
                        binary_model[j] = 1
                        cc += 1
                if cc < 5 or cc > 25:
                    continue
                else:
                    break
            return binary_model

        for i in range(0, 50):
            check = 0
            for j in range(0, 593):
                if pop[i, j] == 1:
                    check += 1

            if check < 5 or check > 25:
                pop[i] = ValidRow(pop[i])

        print(pop)

    return regressor, instructions, trackDesc, trackFitness, trackModel, trackDimen, trackR2train, trackR2valid, trackR2test, testRMSE, testMAE, testAccPred


def function_step7():
    print('\n\nFitness')
    min1 = 2000


descriptors1, targets1 = function_step1(descriptors_file, targets_file)
print()
print('Original descriptors are as follow:')
print()
print(descriptors1)
print()
print('Targets are as below:')
print()
print(targets1)
print()

print('___Function1 done____')

descriptors, targets, active_descriptors = function_step2(descriptors1, targets1)
print()
print('------------------------step 2 ends-------------------------')

descriptors, targets = function_step3(descriptors, targets1)
print('After sorting descriptor matrix is : ')
print(descriptors)
print()
print('after sorting targets are:')
print(targets)
print('------------------------step 3 ends-------------------------')
print()
X_Train, X_Valid, X_Test, Y_Train, Y_Valid, Y_Test, data = function_step4(descriptors, targets)
print()
print('------------------------step 4 ends-------------------------')

binary_model = function_step5()
print('------------------------step 5 ends-------------------------')

regressor, instructions, trackDesc, trackFitness, trackModel, trackDimen, trackR2train, trackR2valid, trackR2test, testRMSE, testMAE, testAccPred = function_step6()
print('------------------------step 6 ends-------------------------')
function_step7()
print('------------------------step 7 ends-------------------------')