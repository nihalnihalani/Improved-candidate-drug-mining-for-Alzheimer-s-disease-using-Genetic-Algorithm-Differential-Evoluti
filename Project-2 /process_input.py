from numpy import *
import numpy
import pandas as pd
import csv
from numpy.core.multiarray import ndarray

# ------------------------------------------------------------------------------------------------
from pandas import DataFrame, Series, to_numeric


class process:
    def rescale_data(self, descriptor_matrix):

        # You have already written code for this.
        df = DataFrame(descriptor_matrix)
        rescaled_matrix = (df - df.values.mean()) / (df.values.std())
        print("Rescaled Matrix is: ")
        rescaled_matrix.to_csv("rescaledmatrix.csv")
        print(rescaled_matrix)
        return rescaled_matrix

    # ------------------------------------------------------------------------------------------------
    # What do we need to sort the data?

    def sort_descriptor_matrix(self, descriptors, targets):
        # Placing descriptors and targets in ascending order of target (IC50) value.
        alldata = ndarray((descriptors.shape[0], descriptors.shape[1] + 1))
        alldata[:, 0] = targets
        alldata[:, 1:alldata.shape[1]] = descriptors
        alldata = alldata[alldata[:, 0].argsort()]
        descriptors = alldata[:, 1:alldata.shape[1]]
        targets = alldata[:, 0]

        return descriptors, targets

    # ------------------------------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------------------------------

    # try to optimize this code if possible

    def open_descriptor_matrix(self, fileName):
        preferred_delimiters = [';', '\t', ',', '\n']

        with open(fileName, mode='r') as csvfile:
            # Dynamically determining the delimiter used in the input file
            row = csvfile.readline()

            delimit = ','
            for d in preferred_delimiters:
                if d in row:
                    delimit = d
                    break

            # Reading in the data from the input file
            csvfile.seek(0)
            datareader = csv.reader(csvfile, delimiter=delimit, quotechar=' ')
            dataArray = array([row for row in datareader if row != ''], order='C')

        if (min(dataArray.shape) == 1):  # flatten arrays of one row or column
            return dataArray.flatten(order='C')
        else:
            return dataArray

    #************************************************************************************
    #Try to optimize this code if possible

    def open_target_values(self, fileName):
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
    # Removes constant and near-constant descriptors.
    # But I think also does that too for real data.
    # So for now take this as it is

    def removeNearConstantColumns(self, data_matrix, num_unique=10):
        useful_descriptors = [col for col in range(data_matrix.shape[1])
                              if len(set(data_matrix[:, col])) > num_unique]
        filtered_matrix = data_matrix[:, useful_descriptors]

        remaining_desc = zeros(data_matrix.shape[1])
        remaining_desc[useful_descriptors] = 1
        # print('Type is: ...............')
        # print(type(where(remaining_desc == 1)[0]))

        return filtered_matrix, where(remaining_desc == 1)[0]

    # ------------------------------------------------------------------------------------------------
    # part 1: Removes all rows with junk (ex: NaN, etc). Note that the corresponding IC50 value should be deleted too
    # Part 2: Remove columns with 20 junks or more. Otherwise the junk should be replaced with zero
    # Part 3: remove all columns that have zero in every cell

    def removeInvalidData(self, descriptors, targets):

        # Write your code in here
        #Converting Numpy to pandas dataframe and series
        descriptors_df = DataFrame(descriptors)
        target1 = Series(targets)

        #converting junk values to NaN
        descriptors_df = descriptors_df.apply(to_numeric, errors='coerce')

        #Look for any NaN value row index
        nan_desc_row = [index for index, row in descriptors_df.iterrows() if row.isnull().any()]

        #Dropping rows with junk values
        descriptors_df = descriptors_df.drop(nan_desc_row)
        target1 = target1.drop(nan_desc_row)
        del_rows = len(nan_desc_row)
        print("Number of deleted junk rows are: ", del_rows)

        #Dropping columns with more than 20 junk values
        junk_col = descriptors_df.isna().sum()
        del_cols = junk_col[junk_col >= 20].count()
        descriptors_df = descriptors_df.drop(
            junk_col[junk_col >= 20].index, axis=1)
        print("Number of deleted junk columns are: ", del_cols)

        #converting all remaining junk colmns to zero's
        print("Converting junk values to 0 ")
        descriptors_df = descriptors_df.fillna(0)

        #dropping columns which has zeros
        Len_count = len(descriptors_df.columns)
        descriptors_df = descriptors_df.loc[:, (descriptors_df != 0).any(axis=0)]
        del_cols = Len_count - len(descriptors_df.columns)
        print("Number of columns dropped with each cell zero: ", del_cols)

        descriptors = descriptors_df.to_numpy()
        targets = target1.to_numpy()

        return descriptors, targets


# ------------------------------------------------------------------------------------------------
