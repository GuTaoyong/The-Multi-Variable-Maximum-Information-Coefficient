import csv
import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

min_x = -math.pi
max_x = math.pi

# def generate_data(function, volume, dimension, min_x, max_x, noise):  # TESTED
#     data_x = list()
#     data_y = list()
#     for i in range(0, volume):
#         x = list()
#         x_with_noise = list()
#         for j in range(0, dimension):
#             # print((max_x[j] - min_x[j]) * random.random() + min_x[j])
#             x.append((max_x[j] - min_x[j]) * random.random() + min_x[j])
#             # print(x[j] + (-1 + 2 * random.random())* noise / 2 * (max_x[j] - min_x[j]))
#             x_with_noise.append(
#                 x[j] + (-1 + 2 * random.random()) * noise / 2 * (max_x[j] - min_x[j]))
#         if (function == 'random'):
#             y = random.random()
#         else:
#             y = eval(function)(x_with_noise)
#         data_x.append(x)
#         data_y.append(y)
#     return data_x, data_y


def data2csv(data, path):  # TESTED
    with open(path, "w+", newline='') as file:
        csv_file = csv.writer(file)
        for i in range(0, len(data)):
            row = data[i]
            csv_file.writerow(row)


def csv2data(path):  # TESETED
    data = list()
    with open(path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row_data = list()
            for i in range(0, len(row)):
                row_data.append(float(row[i]))
            data.append(row_data)
    return data



def generate_data(dimension, function, data_volume, noise, correlation):
    data_x = list()
    data_y = list()
    data = list()
    if correlation !='uc':
        dimension = dimension - 1
        for i in range(0, data_volume):
            x = list()
            y = 0
            for j in range(0, dimension):
                x.append((max_x - min_x) * random.random() + min_x)
            # print(x)
            if correlation == 'ac':
                y = eval(function)(x)
            if correlation == 'pc':
                y = eval(function)(x[0:len(x) - 1])
            data_x.append(x)
            data_y.append(y)
        max_noise = (max(data_y) - min(data_y)) * noise
        for i in range(0, data_volume):
            data_y[i] = data_y[i] + (-1 + 2 * random.random()) * max_noise
            data.append(data_x[i])
            data[i].append(data_y[i])
    if correlation =='uc':
        for i in range(0, data_volume):
            x = list()
            for j in range(0, dimension):
                x.append((max_x - min_x) * random.random() + min_x)
            data.append(x)
    return data


def la(x):
    y = 0
    for i in range(0, len(x)):
        y = y + x[i]
    return y


def pa(x):
    y = 0
    for i in range(0, len(x)):
        y = y + math.pow(x[i], 2)
    return y


def ea(x):
    y = 0
    for i in range(0, len(x)):
        y = y + math.pow(2, x[i])
    return y


def lm(x):
    y = 1
    for i in range(0, len(x)):
        y = y * x[i]
    return y


def pm(x):
    y = 1
    for i in range(0, len(x)):
        y = y * math.pow(x[i], 2)
    return y


def em(x):
    y = 1
    for i in range(0, len(x)):
        y = y * math.pow(2, x[i])
    return y


def sa(x):
    y = 0
    for i in range(0, len(x)):
        y = y + x[i]
    y = math.sin(y)
    return y


def draw_point(data):
    data_volume = len(data)
    x = list()
    y = list()
    for i in range(0, data_volume):
        x.append(data[i][0])
        y.append(data[i][1])
    x = np.array(x)
    y = np.array(y)
    # print(x)
    # print(y)
    plt.scatter(x, y)
    plt.show()



def result2txt(M_matrix, MMIC, MMIC_division, path):
    with open(path, 'a') as f:
        seperate_line = '-------------------------------------------------------------------------------------------------------------------------------'
        f.write('MMIC:\t' + str(MMIC)  + '\nMMIC division:\n')
        for i in range(0, len(MMIC_division)):
            f.write(str(MMIC_division[i]))
        f.write('\nM division:\n')
        for i in range(0, len(M_matrix)):
            f.write('Division Count:\t[')
            for j in range(0,len(M_matrix[i][2])-1):
                f.write(str(M_matrix[i][2][j]) + ',')
            f.write(str(M_matrix[i][2][len(M_matrix[i][2])-1]) + ']\tEntropy:' + str(M_matrix[i][0]) + '\n')
        f.write(seperate_line + '\n')

def information2txt(dimension,data_volume,function,noise,correlation,path):
    with open(path, 'a') as f:
        f.write('Dimension:\t' + str(dimension) +'\nData volume:\t' + str(data_volume) + '\nFunction:\t\t' + function
                + '\nNoise:\t\t' + str(noise) + '\nCorrelation:\t' + correlation + '\n')


def r2_calculate(dimension, function, data_volume, noise, correlation):
    path = '../dataset/' + str(dimension) + '_' + str(data_volume) + '_'  + str(function) + '_' +   str(
                            noise) +  '_' +  str(correlation) +  '.csv'
    data = csv2data(path)
    data_x = list()
    data_y = list()
    data_y_noiseless = list()
    for i in range(0,data_volume):
        x = data[i][0:dimension-1]
        y = data[i][dimension-1]
        data_x.append(x)
        data_y.append(float(y))

        if correlation == 'ac':
            y_noiseless = eval(function)(x)
        if correlation == 'pc':
            y_noiseless = eval(function)(x[0:len(x) - 1])
        # print(y,y_noiseless)
        data_y_noiseless.append(y_noiseless)
    r2 = r2_score(data_y,data_y_noiseless)
    return r2
