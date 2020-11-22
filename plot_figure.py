# ！usr/bin/env python
#!-*-coding:utf-8 -*-
#!@Time:    2019/12/11 9:34
#!@Author:  Gu Taoyong
#!@File:    .py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from src import information_entropy_calculate as iec

# demonstrate the relationship between MMIC and R2
# result_path = '../output/MMIC_R2.csv'
# csv_data = pd.read_csv(result_path)
# result = pd.DataFrame(csv_data)
#
# dimension = [2, 3, 4]
# correlation = ['ac']
# color = ['b', 'r', 'g']
# for i in range(0, len(dimension)):
#     scatter_data = result.loc[result['dimension'].isin(
#         [dimension[i]]) & result['correlation'].isin(correlation)]
#     # print(dimension[i])
#     if dimension[i] == 2:
#         scatter_data = scatter_data.loc[result['data_volume'].isin([100])]
#     if dimension[i] == 3:
#         scatter_data = scatter_data.loc[(result['noise'].isin([0, 0.2]) & result['data_volume'].isin(
#             [100])) | (~result['noise'].isin([0, 0.2]) & result['data_volume'].isin([200]))]
#     if dimension[i] == 4:
#         scatter_data = scatter_data.loc[(result['noise'].isin([0, 0.2]) & result['data_volume'].isin(
#             [100])) | (~result['noise'].isin([0, 0.2]) & result['data_volume'].isin([200]))]
#         print(scatter_data)
#         scatter_data = scatter_data.loc[~(scatter_data['noise'].isin(
#             [0]) & scatter_data['function'].isin(['pa']))]
#         print(scatter_data)
#     scatter_data = scatter_data.loc[:, ['MMIC', 'R2']]
#     # print(scatter_data)
#     x = scatter_data.loc[:, 'R2']
#     y = scatter_data.loc[:, 'MMIC']
#     plt.scatter(x, y, c=color[i], label='dimension=' + str(i + 2))
# x = np.arange(0, 1, 0.01)
# y = x
# plt.plot(x, y, c='k')
# plt.xlabel('R2')
# plt.ylabel('MMIC')
# plt.legend()
# plt.show()
#
# figure_save_path = '../figure/' + 'dimension_compare_R2_MMIC.png'
# plt.savefig(figure_save_path,dpi = 300)
# plt.show()

# # demonstrate the relationship between test partition count and dominating test partition count
# result_path = '../output/Time_Complex.csv'
# csv_data = pd.read_csv(result_path)
# result = pd.DataFrame(csv_data)
#
# scatter_data = result
# x = scatter_data.loc[:,'test partition count']
# y = scatter_data.loc[:,'dominating test partition count']
#
# plt.scatter(x,y,c='b')
# plt.xlabel('partition count')
# plt.ylabel('dominating partition count')
# # plt.legend()
#
# figure_save_path = '../figure/' + 'dominating_partition_rate.png'
# plt.savefig(figure_save_path,dpi = 300)
# plt.show()

# demonstrate the function
# dimension = 3
# data_volume = 200
# function = 'pa'
# noise = 0
# correlation = 'ac'
#
# data_path = '../dataset/' + str(dimension) + '_' + str(data_volume) + '_' + str(
#     function) + '_' + str(noise) + '_' + str(correlation) + '.csv'
# data = pd.read_csv(data_path,header=None)
# scatter_data = data
#
# fig = plt.figure()
# if dimension == 2:
#     x = scatter_data.iloc[:,0].values.reshape(1,data_volume)
#     y = scatter_data.iloc[:,1].values.reshape(1,data_volume)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.scatter(x,y)
# if dimension == 3:
#     x1 = scatter_data.iloc[:,0].values.reshape(1,data_volume)
#     x2 = scatter_data.iloc[:,1].values.reshape(1,data_volume)
#     y = scatter_data.iloc[:,2].values.reshape(1,data_volume)
#     x1_min = min(x1[0])
#     x1_max = max(x1[0])
#     x2_min = min(x2[0])
#     x2_max = max(x2[0])
#     y_min = min(y[0])
#     y_max = max(y[0])
#     # print(x1_min)
#     ax = Axes3D(fig)
#     ax.scatter(x1, x2, y)
#     ax.set_xlabel('x1')
#     ax.set_ylabel('x2')
#     ax.set_zlabel('y')
# # figure_save_path = '../figure/' + str(dimension) + '_' + str(data_volume) + '_' + str(
# #     function) + '_' + str(noise) + '_' + str(correlation) +'.png'
# # fig.savefig(figure_save_path,dpi = 300)
# fig.show()

# demonstrate partition
# demonstrate the function
# dimension = 2
# data_volume = 100
# function = 'pa'
# noise = 0
# correlation = 'ac'
#
# data_path = '../dataset/' + str(dimension) + '_' + str(data_volume) + '_' + str(
#     function) + '_' + str(noise) + '_' + str(correlation) + '.csv'
# data = pd.read_csv(data_path,header=None)
# scatter_data = data
#
# fig = plt.figure()
# if dimension == 2:
#     partition = [[ -2.7837557501417862, -2.3821651862420077, -1.8764956554141468, -1.5574757084749977, -1.4566417782416938,
#      -0.6239895720476736, -0.10022984192852274, 0.43582263470706706, 0.8108309197359527, 1.3181928506995826,
#      1.782564875101087, 2.0367760005579028, 2.4903090466386444, 2.7551130438925395],[ 3.250648762864821]]
#     x = scatter_data.iloc[:,0].values.reshape(1,data_volume)
#     y = scatter_data.iloc[:,1].values.reshape(1,data_volume)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.scatter(x,y,color='r')
#     x_min = min(x[0])
#     x_max = max(x[0])
#     y_min = min(y[0])
#     y_max = max(y[0])
#     # for i in range(0,len(partition[0])):
#     for i in (2,10):
#         plt.plot([partition[0][i],partition[0][i]],[y_min,y_max],color='b',alpha=0.5)
#         # plt.plot([0, y_min], [0, y_max], color='b')
#     for i in range(0,len(partition[1])):
#         plt.plot([x_min,x_max],[partition[1][i],partition[1][i]],color='g',alpha=0.5)
#
#
# if dimension == 3:
#     partition = [[-2.4336730245061826, -1.0101321232495755, 1.5614787104825196, 2.495477062379809],
#                  [-2.1071028065617696, 2.001383761193602, 2.3913251444018306],
#                  [7.687870856943967]]
#     x1 = scatter_data.iloc[:,0].values.reshape(1,data_volume)
#     x2 = scatter_data.iloc[:,1].values.reshape(1,data_volume)
#     y = scatter_data.iloc[:,2].values.reshape(1,data_volume)
#     ax = Axes3D(fig)
#     ax.scatter(x1, x2, y,color='r')
#     ax.set_xlabel('x1')
#     ax.set_ylabel('x2')
#     ax.set_zlabel('y')
#
#     rs = 50
#     cs = 50
#     x1_min = min(x1[0])
#     x1_max = max(x1[0])
#     x2_min = min(x2[0])
#     x2_max = max(x2[0])
#     y_min = min(y[0])
#     y_max = max(y[0])
#     # print(x1_min)
#     x2 = np.arange(x2_min, x2_max, (x2_max - x2_min) / 100).reshape(1, 100)
#     y = np.arange(y_min,y_max,(y_max - y_min)/100).reshape(1,100)
#     x2,y = np.meshgrid(x2,y)
#     for i in range(0, len(partition[0])):
#         x1 = np.array([partition[0][i]]*100).reshape(1,100)
#         ax.plot_wireframe(x1, x2, y, rstride=rs, cstride=cs,alpha =0.5, color = 'b')
#
#     x1 = np.arange(x1_min, x1_max, (x1_max - x1_min) / 100).reshape(1, 100)
#     y = np.arange(y_min, y_max, (y_max - y_min) / 100).reshape(1, 100)
#     x1, y = np.meshgrid(x1, y)
#     for i in range(0, len(partition[1])):
#         x2 = np.array([partition[1][i]] * 100).reshape(1, 100)
#         ax.plot_wireframe(x1, x2, y, rstride=rs, cstride=cs, alpha=0.5,color = 'g')
#
#     x1 = np.arange(x1_min, x1_max, (x1_max - x1_min) / 100).reshape(1, 100)
#     x2 = np.arange(x2_min, x1_max, (x2_max - x1_min) / 100).reshape(1, 100)
#     x1, x2 = np.meshgrid(x1, x2)
#     for i in range(0, len(partition[2])):
#         y = np.array([partition[2][i]] * 100).reshape(1, 100)
#         ax.plot_wireframe(x1, x2, y, rstride=rs, cstride=cs, alpha=0.5,color = 'c')
# figure_save_path = '../figure/' + 'partition_' +  str(dimension) + '_' + str(data_volume) + '_' + str(
#     function) + '_' + str(noise) + '_' + str(correlation) +'.png'
# fig.savefig(figure_save_path,dpi = 300)
# fig.show()
#
# def plot_3D(data,partition):
#     x1 = data.iloc[:, 0].values.reshape(1, data_volume)
#     x2 = data.iloc[:, 1].values.reshape(1, data_volume)
#     y = data.iloc[:, 2].values.reshape(1, data_volume)
#     ax = Axes3D(fig)
#     ax.scatter(x1, x2, y, color='r')
#     ax.set_xlabel('x1')
#     ax.set_ylabel('x2')
#     ax.set_zlabel('y')
#
#     rs = 50
#     cs = 50
#     x1_min = min(x1[0])
#     x1_max = max(x1[0])
#     x2_min = min(x2[0])
#     x2_max = max(x2[0])
#     y_min = min(y[0])
#     y_max = max(y[0])
#     # print(x1_min)
#     x2 = np.arange(x2_min, x2_max, (x2_max - x2_min) / 100).reshape(1, 100)
#     y = np.arange(y_min, y_max, (y_max - y_min) / 100).reshape(1, 100)
#     x2, y = np.meshgrid(x2, y)
#     for i in range(0, len(partition[0])):
#         x1 = np.array([partition[0][i]] * 100).reshape(1, 100)
#         ax.plot_wireframe(x1, x2, y, rstride=rs, cstride=cs, alpha=0.5, color='b')
#
#     x1 = np.arange(x1_min, x1_max, (x1_max - x1_min) / 100).reshape(1, 100)
#     y = np.arange(y_min, y_max, (y_max - y_min) / 100).reshape(1, 100)
#     x1, y = np.meshgrid(x1, y)
#     for i in range(0, len(partition[1])):
#         x2 = np.array([partition[1][i]] * 100).reshape(1, 100)
#         ax.plot_wireframe(x1, x2, y, rstride=rs, cstride=cs, alpha=0.5, color='g')
#
#     x1 = np.arange(x1_min, x1_max, (x1_max - x1_min) / 100).reshape(1, 100)
#     x2 = np.arange(x2_min, x1_max, (x2_max - x1_min) / 100).reshape(1, 100)
#     x1, x2 = np.meshgrid(x1, x2)
#     for i in range(0, len(partition[2])):
#         y = np.array([partition[2][i]] * 100).reshape(1, 100)
#         ax.plot_wireframe(x1, x2, y, rstride=rs, cstride=cs, alpha=0.5, color='c')
#
#     fig.show()

def plot_2D(data,partition,x_index,y_index):
    figure_2D = plt.figure()
    data_volume = data.shape[0]
    x = data.iloc[:, x_index].values.reshape(1, data_volume)
    # print(x)
    y = data.iloc[:, y_index].values.reshape(1, data_volume)
    plt.scatter(x, y, color='r')
    x_min = min(x[0])
    x_max = max(x[0])
    y_min = min(y[0])
    y_max = max(y[0])
    for i in range(0, len(partition[x_index])):
        plt.plot([partition[x_index][i], partition[x_index][i]], [y_min, y_max], color='b', alpha=0.5)
    for i in range(0, len(partition[y_index])):
        plt.plot([x_min, x_max], [partition[y_index][i], partition[y_index][i]], color='g', alpha=0.5)
    figure_save_path = '../figure/temp' + str(x_index) + str(y_index) + '.png'
    figure_2D.savefig(figure_save_path, dpi=300)

def plot_3D(data,partition):
    figure_3D = plt.figure()
    data_volume = iec.get_data_volume(data)
    x1 = data.iloc[:, 0].values.reshape(1, data_volume)
    x2 = data.iloc[:, 1].values.reshape(1, data_volume)
    y = data.iloc[:, 2].values.reshape(1, data_volume)
    ax = Axes3D(figure_3D)
    ax.scatter(x1, x2, y, color='r')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')

    rs = 50
    cs = 50
    x1_min = min(x1[0])
    x1_max = max(x1[0])
    x2_min = min(x2[0])
    x2_max = max(x2[0])
    y_min = min(y[0])
    y_max = max(y[0])
    # print(x1_min)
    x2 = np.arange(x2_min, x2_max, (x2_max - x2_min) / 100).reshape(1, 100)
    y = np.arange(y_min, y_max, (y_max - y_min) / 100)[0:100].reshape(1, 100)
    x2, y = np.meshgrid(x2, y)
    for i in range(0, len(partition[0])):
        x1 = np.array([partition[0][i]] * 100).reshape(1, 100)
        ax.plot_wireframe(x1, x2, y, rstride=rs, cstride=cs, alpha=0.5, color='b')

    x1 = np.arange(x1_min, x1_max, (x1_max - x1_min) / 100).reshape(1, 100)
    y = np.arange(y_min, y_max, (y_max - y_min) / 100)[0:100].reshape(1, 100)
    x1, y = np.meshgrid(x1, y)
    for i in range(0, len(partition[1])):
        x2 = np.array([partition[1][i]] * 100).reshape(1, 100)
        ax.plot_wireframe(x1, x2, y, rstride=rs, cstride=cs, alpha=0.5, color='g')

    x1 = np.arange(x1_min, x1_max, (x1_max - x1_min) / 100).reshape(1, 100)
    x2 = np.arange(x2_min, x2_max, (x2_max - x2_min) / 100).reshape(1, 100)
    x1, x2 = np.meshgrid(x1, x2)
    for i in range(0, len(partition[2])):
        y = np.array([partition[2][i]] * 100).reshape(1, 100)
        ax.plot_wireframe(x1, x2, y, rstride=rs, cstride=cs, alpha=0.5, color='c')
    # figure_save_path = '../figure/temp.png'
    # figure_3D.savefig(figure_save_path, dpi=300)
    figure_3D.show()
    # return figure