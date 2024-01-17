# -*- coding: utf-8 -*-
# @Author  : 邹子理
# @FileName: DoubleDQN.py
# @Time    : 2024/1/15 14:28

import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQN:
    ''' DQN算法,包括Double DQN '''
    def __init__(self,node_num,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 learning_rate,
                 gamma,
                 epsilon,
                 target_update,
                 device,
                 dqn_type='DoubleDQN'):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device

        self.node_num = node_num

    # take_action的逻辑是关键
    # 1，没有遍历完其他点之前不能end
    # 2，不能去往已经经过的点
    # 所以state不仅是当前在哪个stop，还需要包含历史经过的信息，即state = [now_state,stops]
    def take_action(self, state, stops):
        # if len(stops) == self.node_num:
        #     action = stops[0]
        if np.random.random() < self.epsilon:
            action = random.choice([x for x in range(self.node_num) if x not in stops])
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action_value = self.q_net(state)
            action_value[stops] = -np.inf  # 防止选到曾经经过的stop
            action = action_value.argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).view(-1, 1).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        stops = transition_dict['stops']  # 注意这里stops不是都是同一个长度（node_num）！！！
        # print(states)
        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        if self.dqn_type == 'DoubleDQN': # DQN与Double DQN的区别
            q_result = self.q_net(next_states)
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else: # DQN的情况
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size,
              batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state[0],state[1])
                    # print(state[1])
                    # print(state[0])
                    # max_q_value = agent.max_q_value(
                    #     state[0]) * 0.005 + max_q_value * 0.995  # 平滑处理
                    # max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                    # action_continuous = dis_to_con(action, env,
                    #                                agent.action_dim)
                    # next_state, reward, done, _ = env.step([action_continuous])
                    next_state, reward, done = env.step(action)
                    # if done:
                    #     break
                    # if done:
                    #     # print(len(state[1]))
                    #     print(len(next_state[1]))
                    #     print(next_state[0])
                    #     # print("2")
                    # print(next_state[1])
                    replay_buffer.TSP_add(state[0], action, reward, next_state[0], done, state[1])
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d, b_stops = replay_buffer.TSP_sample(
                            batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d,
                            'stops': b_stops
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list, max_q_value_list
















