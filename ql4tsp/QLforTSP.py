# -*- coding: utf-8 -*-
# @Author  :
# @FileName: QLforTSP.py
# @Time    : 2024/1/13 14:41

import random
import numpy as np
from itertools import combinations
import copy
import matplotlib.pyplot as plt

class tsp:
    def __init__(self,map_size,node_num,end_rew):
        self.node_num = node_num
        self.map_size = map_size
        self.end_rew = end_rew
        self.node_pos = self.generate_node_pos()  # 随机生成node_num个站点
        self.node_pos_dict = {cnt:pos for cnt,pos in zip(range(len(self.node_pos)), self.node_pos)}
        self.dist_dict = self.cal_dist(node_pos=self.node_pos)
        self.stops = []  # 按顺序记录依次经过了哪些stop
        self.rew_mat = self.reward_matrix(node_num)
        # self.render()

    # def generate_node_pos(self):  # 生成整数坐标
    #     all_pos = [(x,y) for x in range(self.map_size[0]) for y in range(self.map_size[1])]
    #     uni_pos = random.sample(all_pos,self.node_num)
    #     return uni_pos

    def generate_node_pos(self):  # 生成浮点坐标
        coordinates_set = set()

        while len(coordinates_set) < self.node_num:
            x = random.uniform(0, self.map_size[0])
            y = random.uniform(0, self.map_size[1])
            coordinates_set.add((x, y))

        coordinates_list = list(coordinates_set)
        return coordinates_list

    def cal_dist(self,node_pos):
        distances_dict = {}
        for pair in combinations(node_pos, 2):
            distance = self.calculate_distance(pair[0], pair[1])
            distances_dict[pair] = distance
            distances_dict[(pair[1],pair[0])] = distance  # 对称
        return distances_dict

    def calculate_distance(self, point1, point2):
        # 计算两点之间的欧几里得距离
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def reward_matrix(self,node_num):
        rew_mat = np.zeros([node_num,node_num+1])  # 最后一列放到end的reward，end和start是一个点
        for cnt_x in range(node_num):  # row
            for cnt_y in range(node_num):  # col  # 可考虑对称，减少计算量
                if cnt_x==cnt_y:  # 下一步不能返回自己
                    rew_mat[cnt_x][cnt_y] = np.NAN
                else:
                    rew_mat[cnt_x][cnt_y] = -self.dist_dict.get((self.node_pos[cnt_x], self.node_pos[cnt_y]))
                    # rew_mat[cnt_x][cnt_y] = self.end_rew -1*self.dist_dict.get((self.node_pos_dict.get(cnt_x),self.node_pos_dict.get(cnt_y)))
        # for cnt in range(node_num):
        #     # if cnt==self.stops[0]:
        #     #     rew_mat[cnt][node_num] = np.NAN
        #     # else:
        #     rew_mat[cnt][node_num] = self.end_rew + rew_mat[cnt][0]

        return rew_mat

    def step(self,action):
        state = self.stops[-1]
        self.stops.append(action)
        # action 是从当前位置出发选择的下一个节点
        next_state = [action,self.stops]
        # done = True if (next_state[0]==self.stops[0]) else False
        done = (len(self.stops) == (self.node_num+1))
        # if done:
        #     # reward = self.rew_mat[state][self.node_num]  # 结束奖励不是到0，是到node_num+1
        #     reward = self.end_rew
        # else:

        # dense reward:
        reward = self.rew_mat[state][next_state[0]]

        # # sparse reward
        # reward = 0
        # if done:  # 完成时的奖励是整条路径的总长度
        #     for cnt in range(len(self.stops)-1):
        #         reward += self.rew_mat[self.stops[cnt]][self.stops[cnt+1]]
        # else:
        #     reward = 0

        # if done:
        #     reward += self.rew_mat[state][next_state[0]]

        return next_state, reward, done

    def reset(self):
        self.stops = []
        first_stop = np.random.randint(self.node_num)
        self.stops.append(first_stop)
        return [first_stop,self.stops]

    def render(self, return_img=False):

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        # Show stops
        # end_x = self.node_pos[self.stops[0]][0]
        # end_y = self.node_pos[self.stops[0]][1]
        # self.x = np.concatenate([np.array([x[0] for x in self.node_pos]) , np.array([end_x])],axis=0)
        # self.y = np.concatenate([np.array([x[1] for x in self.node_pos]) , np.array([end_y])],axis=0)
        self.x = np.array([x[0] for x in self.node_pos])
        self.y = np.array([x[1] for x in self.node_pos])
        ax.scatter(self.x, self.y, c="red", s=50)

        # Show START
        xy = self.node_pos_dict.get(0)
        xytext = xy[0]*(1 + 0.1), xy[1]*(1 - 0.05)
        ax.annotate("START", xy=xy, xytext=xytext, weight="bold")

        # Show itinerary
        # self.stops = list(range(self.node_num))+[0]
        ax.plot(self.x[self.stops+[self.stops[0]]], self.y[self.stops+[self.stops[0]]], c="blue", linewidth=1, linestyle="--")

        if return_img:
            # From https://ndres.me/post/matplotlib-animated-gifs-easily/
            fig.canvas.draw_idle()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image
        else:
            plt.show()

# tsp = tsp([10,10],8,100)
# print(tsp.node_pos)
# print(tsp.node_pos_dict)
# print(tsp.dist_dict.get((tsp.node_pos[0],tsp.node_pos[3])))
# print(tsp.dist_dict)