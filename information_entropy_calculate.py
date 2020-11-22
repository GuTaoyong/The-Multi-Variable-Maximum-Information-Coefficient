# ！usr/bin/env python
#!-*-coding:utf-8 -*-
#!@Time:    2019/7/23 9:49
#!@Author:  Gu Taoyong
#!@File:    .py

# import numpy as np
import pandas as pd
import math
import copy
from src import general
from numba import jit

# read data
# data = pd.read_csv('data.csv', sep=",", header=None)
# data.drop(index=0, inplace=True)
# data.drop(axis=1, columns=0, inplace=True)
# data = data.loc[:, 1:3]

# read partition of the data
# partition = list()
# partition.append([20, 40, 60, 80])
# partition.append([1000, 3000, 5000, 7000])
# partition.append([20, 40, 60])

# initialize the count list
# print(data[(data[1]<50) & (data[3]<50)])
# data_temp = data[data[1]<50]
# data_temp = data_temp[data_temp[3]<float('inf')]
# print(data_temp)
# decode: index -> index list


def decode(dimension, partition_count, index):  # TESTED
    '''
    code、decode函数目的是为了去除将维度和各维度长度信息
    :param dimension: int  输入数据的维度
    :param partition_count: list   每一维度数据分组的数量，可以为1（不分组）
    :param index: int   编码
    :return:list    长度为维度的编码
    '''
    reverse_partition_count = copy.deepcopy(partition_count)
    reverse_partition_count.reverse()
    index_list = list()
    index_list.append(index % reverse_partition_count[0])

    for i in range(1, dimension):
        index = index // reverse_partition_count[i - 1]
        index_list.append(index % reverse_partition_count[i])
    index_list.reverse()
    return index_list


'''
decode测试
'''
# for code in range(0,8):
# code_list = decode(2,[2,4],3)
# print(code_list)


def code(dimension, partition_count, index_list):  # TESTED
    '''
    :param dimension: int   数据维度
    :param partition_count:list    每一维度数据分组的数量，可以为1（不分组）
    :param index_list:list  长度为维度的编码
    :return:int   编码
    '''
    index = index_list[0]
    for i in range(0, dimension - 1):
        index = index * partition_count[i + 1] + index_list[i + 1]
    return index


'''
code测试
'''
# code_list = [1,0,3]
# code = code(3,[2,0,4],code_list)
# print(code,code_list)


def get_dimension(data):  # TESTED
    '''
    :param data: list   数据
    :return: int    数据维度
    '''
    dimension = data.shape[1]
    return dimension


def get_data_volume(data):  # TESTED
    '''
    :param data: 数据
    :return: int    数据元组数量
    '''
    data_volume = data.shape[0]
    return data_volume


def get_partition_count(partition):  # TESETED
    '''
    :param     partition:   list    分割
    :return:   partition_count:    list    每一维度数据分组的数量，可以为1（不分组）
                total_partition_count:   分组数量的乘机
    '''
    total_partition_count = 1
    dimension = len(partition)
    partition_count = list()
    for i in range(0, dimension):
        # 由于存在左右两端点（-inf，inf），分割数量为断点数量-1
        partition_count.append(len(partition[i]) - 1)
        total_partition_count = total_partition_count * partition_count[i]
    return partition_count, total_partition_count


def get_total_partition_count(partition):  # TESTED
    '''
    :param partition:
    :return:
    '''
    total_partition_count = 1
    dimension = len(partition)
    for i in range(0, dimension):
        total_partition_count = total_partition_count * (len(partition[i]) - 1)
    return total_partition_count


def set_inf(partition):  # TESETED
    '''
    在每一个维度的分割中加入左右端点-inf和inf
    :param partition:   list    分割
    :return: partition  list    分割
    '''
    dimension = len(partition)
    for i in range(0, dimension):
        partition[i].append(float('inf'))
        partition[i].insert(0, float('-inf'))
    return partition


def count(data, partition):  # TESTED
    '''

    :param data:
    :param partition:
    :return:
    '''
    # partition = set_inf(partition)
    dimension = get_dimension(data)
    partition_count, total_partition_count = get_partition_count(partition)
    index_count = list()
    # data_list = list()
    for index in range(0, total_partition_count):
        index_list = decode(dimension, partition_count, index)
        data_temp = pd.DataFrame(data)
        # print(data_temp.iloc[index,:])
        # print(index)
        # print(partition_count)
        # print('index_list')
        # print(index_list)
        for j in range(0, dimension):
            # print('test')
            # print(data_temp.shape)
            # print(partition)
            # print(j,index_list[j])
            # print(dimension)
            # print(index_list)
            data_temp = data_temp[(data_temp[j] >= partition[j][index_list[j]]) & (
                data_temp[j] < partition[j][index_list[j] + 1])]
            # print(partition[j][index_list[j]])
            # print(partition[j][index_list[j]+1])
            # print(partition[j][index_list[j]])
            # print(partition[j][index_list[j]+1])
            # if (data_temp.empty == False):
            #     data_list = data_list + data_temp.index.tolist()
        index_count.append(data_temp.shape[0])
    # data_list = sorted(data_list)
    # print(data_list)
    return index_count


def entropy_calculate(data, partition):  # TESTED
    '''

    :param data:
    :param partition:
    :return:
    '''
    data_volume = get_data_volume(data)
    entropy = 0

    # print('data')
    # print(data)
    # print('partition')
    # print(partition)

    index_count = count(data, partition)
    # print(partition)
    # print(index_count)
    total_partition_count = len(index_count)
    proportion = list()
    for i in range(0, total_partition_count):
        proportion.append(index_count[i] / data_volume)
    for i in range(0, total_partition_count):
        if (proportion[i] != 0):
            entropy = entropy - proportion[i] * math.log(proportion[i], 2)
    return entropy


def combine(data_x, data_y):  # TESTED
    '''

    :param data_x:
    :param data_y:
    :return:
    '''
    row_count = data_x.shape[0]
    list_data_x = data_x.values.tolist()
    list_data_y = data_y.values.tolist()
    data_list = list()
    for i in range(0, row_count):
        row = list_data_x[i] + list_data_y[i]
        data_list.append(row)
    data = pd.DataFrame(data_list)
    return data


# TESTED
def information_coefficient_calculate(
        data_x, partition_x, data_y, partition_y):
    '''

    :param data_x:
    :param partition_x:
    :param data_y:
    :param partition_y:
    :return:
    '''
    # print(data_x)
    # print(partition_x)
    # print(data_y)
    # print(partition_y)
    entropy_x = entropy_calculate(data_x, partition_x)
    entropy_y = entropy_calculate(data_y, partition_y)

    data = combine(data_x, data_y)
    # print(data)
    partition = partition_x + partition_y
    # print(partition)
    total_partition_count_x = get_total_partition_count(partition_x)
    total_partition_count_y = get_total_partition_count(partition_y)
    regularization = math.log(
        min(total_partition_count_x, total_partition_count_y), 2)
    entropy_xy = entropy_calculate(data, partition)
    information_coefficient = (
        entropy_x + entropy_y - entropy_xy) / regularization
    '''
    print results
    '''
    # print('information entropy of X:', entropy_x)
    # print('information entropy of Y:', entropy_y)
    # print('information entropy of XY:', entropy_xy)
    # print('regularization of XY:', regularization)
    print('information coefficient:', information_coefficient)
    return information_coefficient

@jit
def simplified_information_coefficient_calculate(data, partition):  # TESTED
    '''
    默认data的前dimension-1列为data_x，最后一列为data_y
    默认partition的前dimension-1列为partition_x，最后一列为partition_y
    :param data:
    :param partition:
    :return:
    '''

    data_x = data.loc[:, 0:data.shape[1] - 2]
    data_y = pd.DataFrame(data.loc[:, data.shape[1] - 1].values.tolist())
    # print(data_x)
    # print(data_y)
    partition_x = partition[0:len(partition) - 1]
    partition_y = [partition[len(partition) - 1]]
    # print(partition_x)
    # print(partition_y)
    entropy_x = entropy_calculate(data_x, partition_x)
    entropy_y = entropy_calculate(data_y, partition_y)

    data = combine(data_x, data_y)
    # print(data)
    partition = partition_x + partition_y
    # print(partition)
    total_partition_count_x = get_total_partition_count(partition_x)
    total_partition_count_y = get_total_partition_count(partition_y)
    # print(total_partition_count_x)
    # print(total_partition_count_y)
    # if (min(total_partition_count_x, total_partition_count_y)) == 0:
    #     print('partition == 0')
    #     return 0
    regularization = math.log(
        min(total_partition_count_x, total_partition_count_y), 2)
    # prevent total_partition_count_x == 1 or total_partition_count_y == 1
    if (regularization == 0):
        regularization = 1
    entropy_xy = entropy_calculate(data, partition)
    information_coefficient = (
        entropy_x + entropy_y - entropy_xy) / regularization
    # print('Information entropy of X:', entropy_x)
    # print('Information entropy of Y:', entropy_y)
    # print('Information entropy of XY:', entropy_xy)
    # print('Regularization of XY:', regularization)
    # print(partition)
    # format_partition = general.format_partition(partition)
    # print('partition: ',str(format_partition))
    print('information coefficient:', information_coefficient)
    # general.print_dividing_line()
    return information_coefficient

# calculate the count list using recursive method
# k = 0
# count_list[0] = data.shape[0]

# global count_list
# count_list = np.zeros(total_partition_count)
