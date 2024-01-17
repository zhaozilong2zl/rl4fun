# -*- coding: utf-8 -*-
# @Author  : 邹子理
# @FileName: main_DDQN.py
# @Time    : 2024/1/15 14:55

import matplotlib.pyplot as plt
import numpy as np
import rl_utils
from Qlearning import Qlearning
from QLforTSP import tsp
from tqdm import tqdm
from DoubleDQN import *
# from DoubleDQNforTSP import tsp

node_num = 7  # stop个数
map_size = [1, 1]
end_rew = max(map_size)  # 结束奖励
num_episodes = 3e3  # 训练次数

lr = 1e-2
hidden_dim = 64
gamma = 0.98
epsilon = 0.1
target_update = 25
buffer_size = 5000
minimal_size = 1000
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env = tsp(map_size=map_size, node_num=node_num, end_rew=end_rew)  # TSP env
agent = DQN(node_num=node_num,
            state_dim=1,
            hidden_dim=hidden_dim,
            action_dim=node_num,
            learning_rate=lr,
            gamma=gamma,
            epsilon=epsilon,
            target_update=target_update,
            device=device,
            dqn_type='DoubleDQN')

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)

return_list, max_q_value_list = train_DQN(agent, env, num_episodes,
                                          replay_buffer, minimal_size,
                                          batch_size)

episodes_list = list(range(len(return_list)))
env.render()
# frames_list = list(range(len(max_q_value_list)))
# plt.plot(frames_list, max_q_value_list)
# plt.axhline(0, c='orange', ls='--')
# plt.axhline(10, c='red', ls='--')
# plt.xlabel('Frames')
# plt.ylabel('Q value')
# plt.title('DoubleDQN on {}'.format("TSP"))
# plt.show()

# return_list = []  # 记录每一条序列的回报
# for i in range(10):  # 显示10个进度条
#     # tqdm的进度条功能
#     with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
#         for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
#             episode_return = 0
#             state = env.reset()
#             done = False
#             while not done:
#                 action = agent.take_action(state)
#                 next_state, reward, done = env.step(action)
#                 episode_return += reward  # 这里回报的计算不进行折扣因子衰减
#                 agent.update(state, action, reward, next_state)
#                 state = next_state
#             return_list.append(episode_return)
#             if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
#                 pbar.set_postfix({
#                     'episode':
#                         '%d' % (num_episodes / 10 * i + i_episode + 1),
#                     'return':
#                         '%.3f' % np.mean(return_list[-10:])
#                 })
#             pbar.update(1)
#
# env.render()  # 可视化最后一个episode的轨迹
#
# episodes_list = list(range(len(return_list)))
#
mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DoubleDQN on {}'.format('TSP'))
plt.show()
