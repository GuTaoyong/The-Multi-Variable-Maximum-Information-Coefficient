# !/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import math
import random
import numpy as np
import pandas as pd
import copy
from matplotlib import pyplot as plt
from src import information_entropy_calculate as iec
from src import greedy_strategy as gs
from src import function_data_generate as fdg
from src import general as g
from src import plot_figure as pt

# max partition count coefficient
alpha = 0.6
# potential partition position coefficient
beta = 2
# exploration coefficient
delta = 1
# exploitation coefficient
epsilon = 0.1
# early stop coefficient
zeta = 0.3


# eta = 0.5

data= None
data_volume= None
dimension= None
max_preset_partition_count= None
max_potential_dividing_point_count= None
exploration_budget= None
max_round_count= None
exploitation_budget= None
available_choices= None
available_choices_count= None
information_coefficient_list= None
MMIC= None
MMIC_partition=None
round_mean_information_coefficient = None
max_search_depth = None

class State(object):
    """
    蒙特卡罗树搜索的游戏状态，记录在某一个Node节点下的状态数据，包含当前的游戏得分、当前的游戏round数、从开始到当前的执行记录。
    需要实现判断当前状态是否达到游戏结束状态，支持从Action集合中随机取出操作。
    """
    def __init__(self):
        self.current_value = 0.0
        # For the first root node, the index is 0 and the game should start
        # from 1
        self.current_round_index = 0
        self.cumulative_choices = list()
        self.available_choices = available_choices
        self.cumulative_choices_count = 0
        for i in range(0, dimension):
            self.cumulative_choices.append([i, float('inf')])
            self.cumulative_choices.append([i, float('-inf')])

    def get_current_value(self):
        return self.current_value

    def set_current_value(self, value):
        self.current_value = value

    def get_current_round_index(self):
        return self.current_round_index

    def set_current_round_index(self, turn):
        self.current_round_index = turn

    def get_cumulative_choices(self):
        return self.cumulative_choices

    def set_cumulative_choices(self, choices):
        self.cumulative_choices = choices
        self.available_choices = list(
            choice for choice in available_choices if choice not in self.cumulative_choices)
        # print('len(self.cumulative_choices)')
        # print(len(self.cumulative_choices))
        # print('len(self.available_choices)')
        # print(len(self.available_choices))

    def is_terminal(self):
        # The round index starts from 1 to max round number
        return self.current_round_index == max_round_count

    def compute_reward(self):
        global MMIC, MMIC_partition
        partition = g.choices2partition(
            dimension, self.cumulative_choices)
        ic = iec.simplified_information_coefficient_calculate(
            data, partition)
        if (ic > MMIC):
            MMIC = ic
            MMIC_partition = partition
        self.set_current_value(ic)
        return ic

    def get_next_state_with_random_choice(self):
        '''
        修订
        :return:
        '''
        # 直接随机得到分割
        random_choice = random.choice(
            [choice for choice in self.available_choices])
        # 先随机维度，在随机分割点（维度均衡）
        # random_dimension = random.choice(range(0,dimension))
        # random_partition = random.choice(available_choices[random_dimension])
        # random_choice = [random_dimension,random_partition]
        next_state = State()
        next_state.set_current_round_index(self.current_round_index + 1)
        next_cumulative_choices = copy.deepcopy(self.cumulative_choices)
        next_cumulative_choices.append(random_choice)
        next_state.set_cumulative_choices(next_cumulative_choices)
        return next_state

    def __repr__(self):
        return "State: {}, value: {}, round: {}, \nchoices: {}\n " .format(
            hash(self),
            self.current_value,
            self.current_round_index,
            g.format_partition(
                g.choices2partition(
                    dimension,
                    self.cumulative_choices)))


class Node(object):
    """
    蒙特卡罗树搜索的树结构的Node，包含了父节点和直接点等信息，还有用于计算UCB的遍历次数和quality值，还有游戏选择这个Node的State。
    """

    def __init__(self):
        self.parent = None
        self.children = []
        self.visit_times = 0
        self.quality_value = 0.0
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_children(self):
        return self.children

    def get_visit_times(self):
        return self.visit_times

    def set_visit_times(self, times):
        self.visit_times = times

    def visit_times_add_one(self):
        self.visit_times += 1

    def get_quality_value(self):
        return self.quality_value

    def set_quality_value(self, value):
        self.quality_value = value

    def quality_value_add_n(self, n):
        self.quality_value += n

    def is_all_expand(self):
        return len(self.children) == available_choices_count

    def add_children(self, sub_node):
        sub_node.set_parent(self)
        self.children.append(sub_node)

    def __repr__(self):
        return "Node: {}, Q/N: {}/{}, state: {}".format(
            hash(self), self.quality_value, self.visit_times, self.state)


def tree_policy(node):
    """
    蒙特卡罗树搜索的Selection和Expansion阶段，传入当前需要开始搜索的节点（例如根节点），根据exploration/exploitation算法返回最好的需要expend的节点，注意如果节点是叶子结点直接返回。
    基本策略是先找当前未选择过的子节点，如果有多个则随机选。如果都选择过就找权衡过exploration/exploitation的UCB值最大的，如果UCB值相等则随机选。
    """
    # Check if the current node is the leaf node
    # print('tree_policy')
    # print(node)
    while node.get_state().is_terminal() == False:
        if node.is_all_expand():
            # print('is_all_expand')
            node = best_child(node, True)
        else:
            # Return the new sub node
            # print('sub_node')
            sub_node = expand(node)
            # search_depth = search_depth + 1
            return sub_node
    # Return the leaf node
    return node


def default_policy(node, round):
    """
    蒙特卡罗树搜索的Simulation阶段，输入一个需要expand的节点，随机操作后创建新的节点，返回新增节点的reward。注意输入的节点应该不是子节点，而且是有未执行的Action可以expend的。
    基本策略是随机选择Action。
    """
    # Get the state of the game
    total_final_state_reward = 0
    round_exploitation_budget = math.floor(
        exploitation_budget *
        math.log(
            max_round_count -
            round +
            1,
            2))
    print('round_exploitation_budget',round_exploitation_budget)
    for i in range(0, round_exploitation_budget):
        current_state = node.get_state()
        # Run until the game over
        search_depth = 0
        while current_state.is_terminal() == False & search_depth < max_search_depth:
            # Pick one random action to play and get next state
            search_depth = search_depth + 1
            current_state = current_state.get_next_state_with_random_choice()
        final_state_reward = current_state.compute_reward()
        if (final_state_reward) != 0:
            information_coefficient_list[round].append(
                final_state_reward)
        total_final_state_reward = total_final_state_reward + final_state_reward
    mean_final_state_reward = total_final_state_reward / round_exploitation_budget
    return mean_final_state_reward


def expand(node):
    """
    输入一个节点，在该节点上拓展一个新的节点，使用random方法执行Action，返回新增的节点。注意，需要保证新增的节点与其他节点Action不同。
    """
    tried_sub_node_states = [
        sub_node.get_state() for sub_node in node.get_children()
    ]
    # print('tried_sub_node_states')
    # print(tried_sub_node_states)
    new_state = node.get_state().get_next_state_with_random_choice()
    # Check until get the new state which has the different action from others
    while new_state in tried_sub_node_states:
        new_state = node.get_state().get_next_state_with_random_choice()
    # print('new_state')
    # print(new_state)
    sub_node = Node()
    sub_node.set_state(new_state)
    # print('sub_node')
    # print(sub_node)
    node.add_children(sub_node)
    return sub_node


def best_child(node, is_exploration):
    """
    使用UCB算法，权衡exploration和exploitation后选择得分最高的子节点，注意如果是预测阶段直接选择当前Q值得分最高的。
    """
    # TODO: Use the min float value
    best_score = -sys.maxsize
    best_sub_node = None
    # Travel all sub nodes to find the best one
    # print(len(node.get_children()))
    for sub_node in node.get_children():
        # Ignore exploration for inference
        if is_exploration:
            # C = 1 / math.sqrt(2.0)
            C = 0.0 # 不使用UCB
        else:
            C = 0.0
        # UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
        left = sub_node.get_quality_value() / sub_node.get_visit_times()
        right = 2.0 * math.log(node.get_visit_times()) / \
            sub_node.get_visit_times()
        score = left + C * math.sqrt(right)
        if score > best_score:
            best_sub_node = sub_node
            best_score = score
    return best_sub_node


def backup(node, reward):
    """
    蒙特卡洛树搜索的Backpropagation阶段，输入前面获取需要expend的节点和新执行Action的reward，反馈给expend节点和上游所有节点并更新对应数据。
    """
    # Update util the root node
    while node is not None:
        # Update the visit times
        node.visit_times_add_one()
        # Update the quality value
        node.quality_value_add_n(reward)
        # Change the node to the parent node
        node = node.parent


def monte_carlo_tree_search(node, round):
    """
    实现蒙特卡洛树搜索算法，传入一个根节点，在有限的时间内根据之前已经探索过的树结构expand新节点和更新数据，然后返回只要exploitation最高的子节点。
    蒙特卡洛树搜索包含四个步骤，Selection、Expansion、Simulation、Backpropagation。
    前两步使用tree policy找到值得探索的节点。
    第三步使用default policy也就是在选中的节点上随机算法选一个子节点并计算reward。
    最后一步使用backup也就是把reward更新到所有经过的选中节点的节点上。
    进行预测时，只需要根据Q值选择exploitation最大的节点即可，找到下一个最优的节点。
    """
    # computation_budget = math.floor(max_potential_dividing_point_count/2)
    # Run as much as possible under the computation budget
    for i in range(0, exploration_budget):
        # 1. Find the best node to expand
        expand_node = tree_policy(node)
        # 2. Random run to add node and get reward
        reward = default_policy(expand_node, round)
        # 3. Update all passing nodes with reward
        backup(expand_node, reward)
    # N. Get the best next node
    best_next_node = best_child(node, False)
    return best_next_node


def result_save(path):
    with open(path, 'a') as f:
        # f.write(str(information_coefficient_list)+'\n')
        # f.write('Data Volome:' +  str(data_volume) + '\n')
        # f.write('Dimension:'+ str(dimension) + '\n')
        f.write('alpha: ' + str(alpha) +'\n')
        f.write('beta: ' + str(beta) + '\n')
        f.write('delta: ' + str(delta) + '\n')
        f.write('epsilon: ' + str(epsilon) + '\n')
        f.write('maximum partition:'+ str(max_preset_partition_count) + '\n')
        f.write('maximum potential partition point:' + str(max_potential_dividing_point_count) + '\n')
        f.write('maximum round count: ' + str(max_round_count) + '\n')
        f.write('exploration budget: '+ str(exploration_budget) + '\n')
        f.write('exploitation budget: ' +  str(exploitation_budget) + '\n')
        f.write('round mean information coefficient: ' + str(round_mean_information_coefficient) + '\n')
        f.write('MMIC:\t' + str(MMIC)  + '\nMMIC partition:\n')
        f.write(str(MMIC_partition))
    # for i in range(0,max_round_count):
    #     break
    return


def global_var_init4test(function, m,n , noise, type):
    '''

    :param function:
    :param m: volume
    :param n: dimension
    :param noise:
    :param type:
    :param path:
    :return:
    '''
    path = '../dataset/' + str(n) + '_' + str(m) + \
           '_' + str(function) + '_' + str(noise) + '_' + type + '.csv'
    global data, data_volume, dimension, max_preset_partition_count, max_potential_dividing_point_count, \
        exploration_budget, max_round_count, exploitation_budget, available_choices, available_choices_count, \
        information_coefficient_list, MMIC, MMIC_partition,round_mean_information_coefficient,max_search_depth

    data = pd.DataFrame(fdg.csv2data(path))
    data_volume = iec.get_data_volume(data)
    dimension = iec.get_dimension(data)
    # max_preset_partition_count = dimension * beta * math.log(data_volume, 2)
    # max_potential_dividing_point_count = math.pow(data_volume, alpha)
    # data_volume = iec.get_data_volume(data)
    # dimension = iec.get_dimension(data)
    max_preset_partition_count = dimension * beta * math.log(data_volume, 2)
    max_potential_dividing_point_count = math.pow(data_volume, alpha)
    exploration_budget = math.ceil(max_potential_dividing_point_count * delta)
    max_round_count = math.ceil(
        max_preset_partition_count / (2 ** (dimension - 1)) + (dimension - 1))
    exploitation_budget = math.ceil(
        max_potential_dividing_point_count * epsilon)
    max_search_depth = 5

    print('Data Volome:', data_volume)
    print('Dimension:', dimension)
    print('Maximum partition:', max_preset_partition_count)
    print('Maximum Potential Dividing Point:',
          max_potential_dividing_point_count)
    print('maximum round count: ', max_round_count)
    print('exploration budget: ', exploration_budget)
    print('exploitation budget: ', exploitation_budget)
    g.print_dividing_line()

    available_choices = g.partition2choices(
        dimension, gs.get_potential_partition(data, alpha))
    # print(available_choices)
    available_choices_count = 1
    for i in range(0, len(available_choices)):
        available_choices_count = available_choices_count * \
            len(available_choices[i])

    information_coefficient_list = []
    partition_list = []
    for i in range(0, max_round_count):
        information_coefficient_list.append([])
        partition_list.append([])
    MMIC = 0
    MMIC_partition = []
    round_mean_information_coefficient = []

def global_var_init(d):
    '''
    :param function:
    :param m: volume
    :param n: dimension
    :param noise:
    :param type:
    :param path:
    :return:
    '''
    global data, data_volume, dimension, max_preset_partition_count, max_potential_dividing_point_count, \
        exploration_budget, max_round_count, exploitation_budget, available_choices, available_choices_count, \
        information_coefficient_list, MMIC, MMIC_partition,round_mean_information_coefficient,max_search_depth
    data = d
    data_volume = iec.get_data_volume(data)
    dimension = iec.get_dimension(data)
    max_preset_partition_count = dimension * beta * math.log(data_volume, 2)
    max_potential_dividing_point_count = math.pow(data_volume, alpha)
    # data_volume = iec.get_data_volume(data)
    # dimension = iec.get_dimension(data)
    max_preset_partition_count = dimension * beta * math.log(data_volume, 2)
    max_potential_dividing_point_count = math.pow(data_volume, alpha)
    exploration_budget = math.ceil(max_potential_dividing_point_count * delta)
    max_round_count = math.ceil(
        max_preset_partition_count / (2 ** (dimension - 1)) + (dimension - 1))
    exploitation_budget = math.ceil(
        max_potential_dividing_point_count * epsilon)
    max_search_depth = 5

    print('data volome:', data_volume)
    print('Dimension:', dimension)
    print('maximum partition:', max_preset_partition_count)
    print('maximum potential partition point:',
          max_potential_dividing_point_count)
    print('maximum round count: ', max_round_count)
    print('exploration budget: ', exploration_budget)
    print('exploitation budget: ', exploitation_budget)
    g.print_dividing_line()

    available_choices = g.partition2choices(
        dimension, gs.get_potential_partition(data, alpha))
    # print(available_choices)
    available_choices_count = 1
    for i in range(0, len(available_choices)):
        available_choices_count = available_choices_count * \
            len(available_choices[i])

    information_coefficient_list = []
    partition_list = []
    for i in range(0, max_round_count):
        information_coefficient_list.append([])
        partition_list.append([])
    MMIC = 0
    MMIC_partition = []
    round_mean_information_coefficient = []

def MCTS_MMIC4test(data,reuslt_save_path):
    # Create the initialized state and initialized node
    # global MMIC,MMIC_partition
    init_state = State()
    init_node = Node()
    init_node.set_state(init_state)
    current_node = init_node
    # Set the rounds to play
    round = 0
    while round < max_round_count:
        print("Play round: {}".format(round + 1))
        current_node = monte_carlo_tree_search(current_node, round)
        print("Choose node: {}".format(current_node))
        g.print_dividing_line()
        round = round + 1
    current_state = current_node.get_state()
    # partition = g.choices2partition(dimension,current_state.get_cumulative_choices())
    # pt.plot_3D(data,partition)
    # iec.simplified_information_coefficient_calculate(data,partition)
    print(MMIC)
    print(MMIC_partition)

    for i in range(0, max_round_count):
        round_mean_information_coefficient.append(
            np.mean(information_coefficient_list[i]))
    # print(information_coefficient_list)
    print(round_mean_information_coefficient)
    result_save(reuslt_save_path)
    plt.scatter(range(0, max_round_count), round_mean_information_coefficient)
    plt.show()


def MCTS_MMIC(data,reuslt_save_path):
    # Create the initialized state and initialized node
    # global MMIC,MMIC_partition
    global_var_init(data)
    init_state = State()
    init_node = Node()
    init_node.set_state(init_state)
    current_node = init_node
    # Set the rounds to play
    round = 0
    while round < max_round_count:
        print("Play round: {}".format(round + 1))
        current_node = monte_carlo_tree_search(current_node, round)
        print("Choose node: {}".format(current_node))
        g.print_dividing_line()
        round = round + 1
    # current_state = current_node.get_state()
    # partition = g.choices2partition(dimension,current_state.get_cumulative_choices())
    # pt.plot_3D(data,partition)
    # iec.simplified_information_coefficient_calculate(data,partition)
    print(MMIC)
    print(MMIC_partition)

    for i in range(0, max_round_count):
        round_mean_information_coefficient.append(
            np.mean(information_coefficient_list[i]))
    # print(information_coefficient_list)
    # print(round_mean_information_coefficient)
    result_save(reuslt_save_path)
    # plt.scatter(range(0, max_round_count), round_mean_information_coefficient)
    # plt.show()

def __main__():
    dimension_type = [4]
    data_volume_type = [200]
    function_type = ['la','lm', 'pa','pm','ea','em','sa']
    # function_type = ['pm', 'ea', 'em', 'sa']
    noise_type = [0, 0.2, 0.4, 0.6, 0.8]
    correlation_type = ['ac']
    for n in dimension_type:
        for m in data_volume_type:
            for f in function_type:
                for noise in noise_type:
                    for type in correlation_type:
                        path = '../dataset/' + str(n) + '_' + str(m) + \
                               '_' + str(f) + '_' + str(noise) + '_' + type + '.csv'
                        reuslt_save_path = '../output/MCTS/classic/'  + str(n) + '_' + str(
                            m) + '_' + str(f) + '_' + str(noise) + '_' + type + '.txt'
                        global_var_init4test(function=f, m=m, n=n, noise=noise, type=type)
                        MCTS_MMIC4test(data,reuslt_save_path)

if __name__ == '__main__':
    __main__()


