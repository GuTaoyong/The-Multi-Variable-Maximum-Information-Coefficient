from src import information_entropy_calculate as iec
import math
import numpy as np
import copy

# greedy()

# max_iteration = 10
# partition_proportion = data_count ^ 0.6
alpha = 0.6
beta = 2
gamma = 1
theta = 1


def print_dividing_line():
    print('-------------------------------------------------------------------------------------------------------------------------------')


def greedy_MMIC(data):

    '''

    :param data:
    :return:
    '''
    print('MMIC calculated by greedy stepwise strategy:')
    print('alpha = ' + str(alpha))
    print('beta = ' +  str(beta))
    print('gamma = ' +  str(gamma))
    print('theta = ' +  str(theta))

    data_volume = iec.get_data_volume(data)

    dimension = iec.get_dimension(data)
    max_preset_partition_count = dimension * beta * math.log(data_volume, 2)
    max_potential_dividing_point_count = math.pow(data_volume, alpha)

    # partition_count_list = get_preset_partition_count(
    #     dimension, max_preset_partition_count)
    partition_count_list, optimal_partition_count_list_compare = get_preset_partition_count_with_optimization(
        dimension, max_preset_partition_count)
    # print(partition_count_list)

    optimal_partition_count_list = partition_count_optimize(partition_count_list,max_preset_partition_count, max_potential_dividing_point_count)
    # print(optimal_partition_count_list)

    dimension_selection = list()
    dimension_exclusion = list()

    for i in range(dimension):
        dimension_selection.append(i)
    optimal_partition_count_list = partition_count_optimize_with_dimension_selection(optimal_partition_count_list,dimension_selection,dimension_exclusion)

    print('Data Volome:', data_volume)
    print('Dimension:', dimension)
    print('Maximum partition:', max_preset_partition_count)
    print('Maximum Potential Dividing Point:',
          max_potential_dividing_point_count)
    print('Maximum Testing partition Count:', len(partition_count_list))
    print('Dominating Maximum Testing partition Count:',len(optimal_partition_count_list))
    # print(optimal_partition_count_list)
    print_dividing_line()
    # print(optimal_partition_count_list)

    M_matrix = list()
    for i in range(0,len(optimal_partition_count_list)):
        partition, entropy = greedy_information_entropy_calculate(data, optimal_partition_count_list[i])
        M_matrix.append([entropy,partition,optimal_partition_count_list[i]])
        print('partition Count:',M_matrix[i][2],'Entropy:',M_matrix[i][0])
        # print(partition,entropy)
    MMIC = 0
    MMIC_partition = list()
    for i in range(0,len(M_matrix)):
        if M_matrix[i][0] > MMIC:
            MMIC = M_matrix[i][0]
            MMIC_partition = M_matrix[i][1]
    print_dividing_line()
    print('MMIC partition:',MMIC_partition, 'MMIC:', MMIC)
    print_dividing_line()
    return M_matrix, MMIC, MMIC_partition

def partition_count_optimize_with_dimension_selection(partition_count_list,dimension_selection,dimension_exclusion):
    dimension = len(partition_count_list[0])
    optimal_flag = [True] * len(partition_count_list)
    optimal_partition_count_list = list()
    for i in range(0,len(partition_count_list)):
        for j in range(0,dimension):
            if (j in dimension_selection) & (partition_count_list[i][j]==1):
                optimal_flag[i] = False
                continue
            if (j in dimension_exclusion) & (partition_count_list[i][j]>1):
                optimal_flag[i] = False
    for i in range(0, len(partition_count_list)):
        if optimal_flag[i]:
            optimal_partition_count_list.append(partition_count_list[i])
            # print(partition_count_list[i])
    return optimal_partition_count_list


def partition_count_optimize(partition_count_list, max_preset_partition_count, max_potential_dividing_point_count): # TESTED
    dimension = len(partition_count_list[0])
    # print(dimension)
    optimal_partition_count_list = list()
    optimal_flag = [True] * len(partition_count_list)
    for i in range(0, len(partition_count_list)):
        if math.pow(partition_count_list[i][dimension - 1], 2) > max_preset_partition_count:
            # print(partition_count_list[i])
            # print(partition_count_list[i])
            optimal_flag[i] = False
        for j in range(0,dimension):
            if partition_count_list[i][j] > max_potential_dividing_point_count:
                optimal_flag[i] = False
                continue
    # for i in range(0, len(partition_count_list)):
    #     if optimal_flag[i]:
    #         print(partition_count_list[i])
    for i in range(0, len(partition_count_list)):
        if optimal_flag[i]:
            for j in range(0, i):
                if optimal_flag[j]:
                    if (partition_count_list[i][dimension - 1]
                            == partition_count_list[j][dimension - 1]):
                        temp_optimal_flag_i = False
                        temp_optimal_flag_j = False
                        for k in range(0, dimension - 1):
                            if (partition_count_list[i][k]
                                    > partition_count_list[j][k]):
                                temp_optimal_flag_i = True
                            if (partition_count_list[i][k]
                                    < partition_count_list[j][k]):
                                temp_optimal_flag_j = True
                        if not temp_optimal_flag_i:
                            optimal_flag[i] = False
                        if not temp_optimal_flag_j:
                            optimal_flag[j] = False
    print_dividing_line()
    for i in range(0, len(partition_count_list)):
        if optimal_flag[i]:
            optimal_partition_count_list.append(partition_count_list[i])
            # print(partition_count_list[i])
    return optimal_partition_count_list


def get_preset_partition_count_with_optimization(
        dimension, max_preset_partition_count): # TESTED
    partition_count_list = list()
    optimal_partition_count_list = list()

    recurse_preset_partition_count(
        partition_count_list, 0, dimension, [], max_preset_partition_count)
    # print(partition_count_list)
    optimal_flag = [True] * len(partition_count_list)

    for i in range(0, len(partition_count_list)):
        if math.pow(partition_count_list[i][dimension - 1], 2) > max_preset_partition_count:
            # print(partition_count_list[i])
            # print(partition_count_list[i])
            optimal_flag[i] = False
    # for i in range(0, len(partition_count_list)):
    #     if optimal_flag[i]:
    #         print(partition_count_list[i])
    for i in range(0, len(partition_count_list)):
        if optimal_flag[i]:
            for j in range(0, i):
                if optimal_flag[j]:
                    if (partition_count_list[i][dimension - 1]
                            == partition_count_list[j][dimension - 1]):
                        temp_optimal_flag_i = False
                        temp_optimal_flag_j = False
                        for k in range(0, dimension - 1):
                            if (partition_count_list[i][k]
                                    > partition_count_list[j][k]):
                                temp_optimal_flag_i = True
                            if (partition_count_list[i][k]
                                    < partition_count_list[j][k]):
                                temp_optimal_flag_j = True
                        if not temp_optimal_flag_i:
                            optimal_flag[i] = False
                        if not temp_optimal_flag_j:
                            optimal_flag[j] = False
    # print_dividing_line()
    for i in range(0,len(partition_count_list)):
        if optimal_flag[i]:
            optimal_partition_count_list.append(partition_count_list[i])
            # print(partition_count_list[i])
    return partition_count_list, optimal_partition_count_list


def get_preset_partition_count(dimension, max_preset_partition_count):  # TESTED
    partition_count_list = list()
    recurse_preset_partition_count(
        partition_count_list, 0, dimension, [], max_preset_partition_count)
    print(partition_count_list)
    return partition_count_list


def recurse_preset_partition_count(partition_count_list, current_dimension, dimension, partition_prelist, max_preset_partition_count):  # TESTED
    if current_dimension == dimension:
        # print('point 2')
        if (partition_prelist[len(partition_prelist) - 1]) > 1:
            for i in range(0, len(partition_prelist) - 1):
                if partition_prelist[i] > 1:
                    partition_count_list.append(partition_prelist)
                    # print(partition_prelist)
                    return
        # return
    else:
        # print(partition_prelist,max_preset_partition_count)
        for i in range(1, math.ceil(max_preset_partition_count)):
            current_partition_prelist = copy.deepcopy(partition_prelist)
            # print(current_partition_prelist)
            current_partition_prelist.append(i)
            # print(current_partition_prelist)
            recurse_preset_partition_count(
                partition_count_list,
                current_dimension + 1,
                dimension,
                current_partition_prelist,
                max_preset_partition_count / i)


def greedy_information_entropy_calculate(data, preset_partition_count): # TESTED
    # print('preset_partition_count in greedy_information_entropy_calculate')
    # print(preset_partition_count)
    partition, potential_partition = initialize_partition_with_preset(
        data, preset_partition_count)
    dimension = iec.get_dimension(data)
    for i in range(0, theta):
        for current_dimension in range(0, dimension):
            partition, potential_partition = get_max_gain_partition(
                data, partition, potential_partition, current_dimension, i)
    entropy = iec.simplified_information_coefficient_calculate(data,partition)
    return partition, entropy


# partition_count = pow(data_volume, ALPHA)
# TESTED
def get_potential_partition_with_dimension(data, current_dimension, alpha):
    data_volume = iec.get_data_volume(data)
    max_partition_count = int(math.pow(data_volume, alpha) + 0.5)
    partition_proportion = list()
    data_column = np.array(data.loc[:, current_dimension].values.tolist())
    for i in range(0, max_partition_count):
        partition_proportion.append(100 * (i + 1) / (max_partition_count + 1))
        # print((i+1)/(max_partition_count+1))
    potential_partition = np.percentile(
        data_column, partition_proportion).tolist()
    return potential_partition


def get_potential_partition(data, alpha):  # TESTED
    data_volume = iec.get_data_volume(data)
    max_partition_count = int(math.pow(data_volume, alpha) + 0.5)
    dimension = iec.get_dimension(data)
    # partition_proportion = list()
    potential_partition = list()
    for i in range(0, dimension):
        potential_partition.append([])
        data_column = np.array(data.loc[:, i].values.tolist())
        partition_proportion = list()
        for j in range(0, max_partition_count):
            partition_proportion.append(
                100 * (j + 1) / (max_partition_count + 1))
        # print((i+1)/(max_partition_count+1))
        potential_partition_temp = np.percentile(
            data_column, partition_proportion).tolist()
        # print(len(potential_partition_temp))
        potential_partition[i] = copy.deepcopy(potential_partition_temp)
    return potential_partition


def get_max_gain_partition(
        data,
        partition,
        potential_partition,
        current_dimension,
        out_iteration): # TESTED
    # remove -inf and inf
    # print('Initial partition:', partition)
    max_iteration = gamma * len(partition[current_dimension]) - 2
    remove_index = 0
    insert_index = 0
    # print('data.shape')
    # print(data.shape)
    # print('len(partition)')
    # print(len(partition))
    original_entropy = iec.simplified_information_coefficient_calculate(
        data, partition)
    max_entropy = original_entropy
    for iteration in range(0, max_iteration):
        replace = False
        # print('Iteraion:', out_iteration)
        # print('Dimension:', current_dimension)
        # print('Iteraion in Dimension:', iteration)
        # max_entropy_in_iteration = list()
        for i in range(1, len(partition[current_dimension]) - 1):
            partition_after_remove = copy.deepcopy(partition)
            partition_after_remove[current_dimension].remove(
                partition[current_dimension][i])
            for j in range(0, len(potential_partition[current_dimension])):
                partition_after_insert = copy.deepcopy(partition_after_remove)
                partition_after_insert[current_dimension].append(
                    potential_partition[current_dimension][j])
                partition_after_insert[current_dimension].sort()
                entropy = iec.simplified_information_coefficient_calculate(
                    data, partition_after_insert)
                # print(partition_after_insert)
                # print(entropy)
                if entropy > max_entropy:
                    max_entropy = entropy
                    # max_entropy_in_iteration = entropy
                    remove_index = i
                    insert_index = j
                    replace = True
        if not replace:
            # print('No Replacement.')
            break
        else:
            temp_remove = partition[current_dimension][remove_index]
            # print(remove_index)
            # print(temp_remove)
            temp_insert = potential_partition[current_dimension][insert_index]
            partition[current_dimension].remove(temp_remove)
            partition[current_dimension].append(temp_insert)
            partition[current_dimension].sort()
            # print('partition:', partition)
            # print('entropy:', max_entropy)
            potential_partition[current_dimension].remove(temp_insert)
            potential_partition[current_dimension].append(temp_remove)
            potential_partition[current_dimension].sort()
            # print('Potential partition:',len(potential_partition[out_iteration]))
        # print_dividing_line()
        # print(partition)
        # print(max_entropy)
    return partition, potential_partition


def get_max_partition(data, partition, potential_partition, current_dimension):  # TESTED
    information_coefficient = 0
    for i in range(0, len(potential_partition)):
        current_partition = copy.deepcopy(partition)
        current_partition[current_dimension].append(potential_partition[i])
        current_partition[current_dimension].sort()
        # print(partition)
        # print_dividing_line()
        # print('Current partition:', current_partition)
        temp_information_coefficient = iec.simplified_information_coefficient_calculate(
            data, current_partition)
        if temp_information_coefficient > information_coefficient:
            information_coefficient = temp_information_coefficient
            partition_return = copy.deepcopy(current_partition)
    return partition_return, temp_information_coefficient

# def initialize_partition(data): # TESTED
#     partition = list()
#     dimension = iec.get_dimension(data)
#     for i in range(0,dimension):
#         partition.append([])
#     # set Y 1/2 point for initialization
#     potential_partition_y = get_potential_partition(data,dimension-1,0.6)
#     # print(potential_partition_y)
#     partition[dimension-1].append(potential_partition_y[int(len(potential_partition_y)/2)])
#     iec.set_inf(partition)
#     return partition


def initialize_partition_with_preset(data, preset_partition_count):  # TESTED
    partition = list()
    potential_partition = get_potential_partition(data, alpha)
    dimension = iec.get_dimension(data)
    # for i in range(0,dimension):
    #     print(len(potential_partition[i]))
    for i in range(0, dimension):
        partition.append([])
    # print(potential_partition_y)
    print(preset_partition_count)
    for i in range(0, dimension):
        if preset_partition_count[i] > 1:
            for j in range(1, preset_partition_count[i]):
                temp = potential_partition[i][int(
                    j * len(potential_partition[i]) / preset_partition_count[i])]
                partition[i].append(temp)
                potential_partition[i].remove(temp)
        # partition[i].tolist()
    iec.set_inf(partition)
    # print('partition in initialize_partition_with_preset')
    # print(partition)
    return partition, potential_partition
