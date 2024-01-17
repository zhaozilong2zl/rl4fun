# -*- coding: utf-8 -*-
# @Author  :
# @FileName: main.py
# @Time    : 2024/1/13 14:22

import matplotlib.pyplot as plt
import numpy as np
import rl_utils
from Qlearning import Qlearning
from QLforTSP import tsp
from tqdm import tqdm

chos = 1  # chos： 1 随机初始化地图； 0 导入固定地图
node_num = 20  # stop个数
map_size = [1,1]
end_rew = max(map_size)  # 结束奖励
num_episodes = 5e3  # 训练次数

env = tsp(map_size=map_size,node_num=node_num,end_rew=end_rew)  # TSP env
agent = Qlearning(alpha=0.2, gamma=0.926, epsilon=0.5,ep_decay=0.999, final_epsilon=1e-10, chos=chos, node_num=node_num)
return_list = []  # 记录每一条序列的回报
for i in range(10):  # 显示10个进度条
    # tqdm的进度条功能
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done = env.step(action)
                episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                agent.update(state, action, reward, next_state)
                state = next_state
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

env.render()  # 可视化最后一个episode的轨迹

episodes_list = list(range(len(return_list)))

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Q-learning on {}'.format('TSP'))
plt.show()

