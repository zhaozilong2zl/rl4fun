# -*- coding: utf-8 -*-
# @Author  :
# @FileName: Qlearning.py
# @Time    : 2024/1/13 14:23
import random

import numpy as np


class Qlearning:
    def __init__(self,alpha, gamma, epsilon, ep_decay, final_epsilon, chos, node_num):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = ep_decay
        self.final_epsilon = final_epsilon
        self.chos = chos
        self.node_num = node_num
        self.Q_table = np.zeros([node_num,node_num])

    # take_action的逻辑是关键
    # 1，没有遍历完其他点之前不能end
    # 2，不能去往已经经过的点
    # 所以state不仅是当前在哪个stop，还需要包含历史经过的信息，即state = [now_state,stops]
    def take_action(self,state):
        q = np.copy(self.Q_table[state[0],:])
        q[state[1]] = -np.inf
        if (len(state[1])==self.node_num):
            action = state[1][0]
        elif np.random.random() < self.epsilon:
            action = random.choice([x for x in range(self.node_num) if x not in state[1]])
        else:
            action = np.argmax(q)
        return action

        # if not remaining_stops:  # 如果所有城市都经过了
        #     action = 0
        # elif np.random.random() < self.epsilon:  # 在没有经过的stop里面随机选一个作为下一个目的地
        #     # action = np.random.randint(self.n_action)
        #     action = random.choice(remaining_stops)
        # else:
        #     # action = np.argmax(self.Q_table[state])
        #     max_value = max(self.Q_table[state[0]][remaining_stops])
        #     max_idx = [i for i,value in enumerate(self.Q_table[state[0]][remaining_stops]) if value==max_value]
        #     action = remaining_stops[random.choice(max_idx)]
        #     # if len(max_idx)==1:
        #     #     action = remaining_stops[np.argmax(self.Q_table[state[0]][remaining_stops])]
        #     # else:
        #     #     action = remaining_stops[random.choice(max_idx)]
        #     # action = remaining_stops[np.argmax(self.Q_table[state[0]][remaining_stops])]
        # return action

    def update(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1[0]].max() - self.Q_table[s0[0], a0]
        self.Q_table[s0[0], a0] += self.alpha * td_error

        if self.epsilon > self.final_epsilon:
            self.epsilon *= self.epsilon_decay

        # print(self.epsilon)













