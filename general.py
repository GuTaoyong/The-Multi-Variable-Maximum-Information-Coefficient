import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
def print_test_for():
    print('Test For')


def print_dividing_line(): #
    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')

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

def format_partition(partition): # TESTED
    format_partition = []
    for i in range(0,len(partition)):
        format_partition.append([])
        for j in range(0,len(partition[i])):
            format_partition[i].append(round(partition[i][j],2))
    return format_partition


def partition2choices(dimension, parttion):
    choices = []
    for i in range(0, dimension):
        for j in range(0, len(parttion[i])):
            choices.append([i, parttion[i][j]])
    return choices


def choices2partition(dimension, choices):  # TESTED
    partition = []
    for i in range(0, dimension):
        partition.append([])
    for i in range(0, len(choices)):
        partition[choices[i][0]].append(choices[i][1])
    for i in range(0, dimension):
        partition[i].sort()
    return partition

# partition =[[-float('inf'), -0.5890536162879023, -0.2234996182411872, 0.030284304359210146, float('inf')], [-float('inf'), -0.07195561138583853, 2.102061624859638, float('inf')], [-float('inf'), -0.006628727425010521, float('inf')]]
# format_partition = format_partition(partition)
# print(format_partition)