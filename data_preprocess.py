# ！usr/bin/env python
#!-*-coding:utf-8 -*-
#!@Time:    2019/12/6 10:34
#!@Author:  Gu Taoyong
#!@File:    .py

'''
库依赖
'''

import pandas as pd
from minepy import MINE
import numpy as np
import csv
import operator
import matplotlib.pyplot as plt

'''
内部函数依赖
'''
from src import greedy_strategy as gs
from src import information_entropy_calculate as iec
from src import function_data_generate as fdg

'''
函数
'''

def get_column_name(path):
    csv_data = pd.read_csv(path, low_memory=False)
    csv_df = pd.DataFrame(csv_data)
    column_name = list(csv_df.columns)
    return column_name

'''
get_column_name
即时测试
'''

# def get_index

def get_index(column_name, select_name):
    index = list()
    for i in range(0, len(select_name)):
        match_flag = False
        for j in range(0, len(column_name)):
            if (select_name[i] == column_name[j]):
                match_flag = True
                index.append(j)
        if not match_flag:
            print(select_name[i] + ' is not in the column.')
            exit(-1)
    return index


def get_mic_matrix(mic_path):
    mic_data = pd.read_csv(mic_path, low_memory=False, header=None)
    mic_matrix = pd.DataFrame(mic_data).values
    return mic_matrix


def sort_mic(mic_path, column_name, select_name):
    index = 0
    match_flag = False
    # print(select_name)
    for i in range(0, len(column_name)):
        # print(column_name[i])
        if (select_name == column_name[i]):
            match_flag = True
            index = i
    if not match_flag:
        print(str(select_name) + ' is not in the column.')
        exit(-1)
    # print(index)
    mic_data = pd.read_csv(mic_path, low_memory=False, header=None)
    mic_matrix = pd.DataFrame(mic_data).values
    select_mic = mic_matrix[:, index]
    # print(select_mic)
    column_name_mic = list()
    # print(len(column_name))
    for i in range(0, len(column_name)):
        if(select_mic[i] != 0) & (select_mic[i] != 1):
            column_name_mic.append([column_name[i], float(select_mic[i])])
    column_name_mic.sort(key=operator.itemgetter(1), reverse=True)
    # print(column_name_mic)
    return column_name_mic


def generate_data_column_name(data_path, x_column_name, y_column_name, generate_data_path):
    csv_data = pd.read_csv(data_path, low_memory=False)
    csv_df = pd.DataFrame(csv_data)
    column_name = list(csv_df.columns)
    x_index = get_index(column_name,x_column_name)
    y_index = get_index(column_name,y_column_name)
    # print(x_index,y_index)
    index = x_index
    index.append(y_index[0])
    # print(index)
    data = csv_df.iloc[:, index]
    data.dropna(axis=0,how='any',inplace = True)
    # print(data)
    data.to_csv(generate_data_path,index = False,header =True)
    return


'''
测试脚本
'''
#
# DATA_VOLUME = 1000
# X_MAX = 10
# X_MIN = -10
#
# data = fdg.generate_data('random', DATA_VOLUME, 0.4, X_MIN, X_MAX)
# # print(data)
#
# for i in range(1, len(data) - 1):
#     plt.scatter(data[i][0], data[i][1],c='b')
# plt.show()
#
# data = [[[1,2],[3,4]],[[5,6],[7,8]]]
# data = np.array(data)
# print(data.shape)
#
# def Calculate_mutual_information(data, division):
#     dimension = data.ndim
#     for i in range (dimension):
#         pass
#
# import pandas as pd
# data = pd.read_csv('../Dataset/WHO2.csv', sep=",", header=None)
# # print(data.shape[1])
# column_name = []
# for i in range(1, data.shape[1] - 1):
#     # print(data[i][0])
#     column_name.append(data[i][0])
# # print(column_name)
#
# row_name = []
# for i in range(1, data.shape[0] - 1):
#     row_name.append(data[0][i])
# # print(row_name)
#
# data = data.drop(index=0)
# data = data.drop(axis=1, columns=[0, 1, 2])
#
# # print(data.shape)
# # print(data.index)
# # print(data.columns)
#
# population_data = data.loc[:, 4:17]
# # print(population_data.index)
# # print(population_data.columns)
#
# # print(population_data.loc[1,5])
# population_data = population_data.dropna(axis=0, how='any')
# # print(population_data)
# # print(population_data.index.values)
#
# population_data.to_csv('data.csv')
#
#
# # for i in range(0,popul1ation_data.index.values.shape[0]):
# #     print(row_name[population_data.index.values[i]-1])
#
# # MIC CALCULATION
# mic = np.eye(population_data.shape[1])
# array_population_data = population_data.values
# from minepy import MINE
# for i in range(population_data.shape[1]):
#     for j in range(population_data.shape[1]):
#         mine = MINE(alpha=0.6, c=15, est="mic_approx")
#         mine.compute_score(array_population_data[:][i],array_population_data[:][j])
#         mic[i][j] = np.around(mine.mic(),decimals=2)
#
# # print(mic)
#
# import csv
#
# with open("mic.csv","w") as csvfile:
#     writer = csv.writer(csvfile)
#     row = column_name
#     row.insert(0, '')
#     writer.writerow(row)
#     for i in range(population_data.shape[1]):
#         row = mic[i][:]
#         row = row.tolist()
#         print(row)
#         # print(type(row))
#         # row.insert(0,column_name[i])
#         # writer.writerow(row)
#
#
#
