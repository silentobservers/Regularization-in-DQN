# Author: Chengjia Lei


import gym
from RL_brain import DQNPrioritizedReplay
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np



gym.envs.register(
            id='MountainCar-default-v0',
            entry_point='gym.envs.classic_control:MountainCarEnv',
            # tags={'wrapper_config.TimeLimit.max_episode_steps': 500.0},
            reward_threshold=-110,
            kwargs={'index': 1},
        )
env = gym.make('MountainCar-default-v0')


MEMORY_SIZE = 10000 #100000
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
with tf.variable_scope('DQN'):
    RL_natural = DQNPrioritizedReplay(
        n_actions=env.action_space.n, n_features=np.shape(env.observation_space)[0], memory_size=MEMORY_SIZE,
        e_greedy_increment=0.0001, sess=sess, prioritized=False,
    )


sess.run(tf.global_variables_initializer())



def train(RL, steps_limit):
    # env.render()
    steps_num = 0
    solved = False
    sumreward = 1
    account = 0
    sess.run(tf.global_variables_initializer())
    RL.initiate_common_par()
    observation = env.reset()
    episodes_reward = []
    episodes = []
    done = False
    action = 0
    for _ in range(MEMORY_SIZE):
        if done:
            observation = env.reset()
        # env.render()
        # if _ % 1 == 0:
        action = env.action_space.sample()
        observation_, reward, done, info = env.step(action)
        if done:
            reward=0
        else:
            reward=-1
        RL.store_transition(observation, action, reward, observation_)
            # if done:
            #     print("fff")

        # print(account)
        # for i_episode in range(15):
    done = True
    # env._max_episode_steps = 200
    while not solved:
        if done:
            observation = env.reset()
            sumreward = -1
            done = False
        # env.render()
        action = RL.choose_action(observation, initial = False)
        observation_, reward, done, info = env.step(action)
        steps_num += 1
        if done:
            reward=0
        else:
            reward=-1
        sumreward+= reward
        RL.store_transition(observation, action, reward, observation_)
        RL.learn()
        observation = observation_

        if done:
            account+=1
            # print('episode ', i_episode, ' episode steps', episode_steps, 'total steps', np.sum(steps))
            episodes_reward.append(sumreward)
            print('times:', account, 'sumreward:', sumreward)
        if steps_num == steps_limit:
            solved = True
    print('Model trained completely!')
    # env_e.close()
    # env._max_episode_steps = 10000
    done = True
    gather=[]
    print('Evaluate!')
    for index in range(80, 141, 1):
        index = index / 100.0
        gym.envs.register(
            id='MountainCar-evaluate-v0',
            entry_point='gym.envs.classic_control:MountainCarEnv',
            tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
            reward_threshold=-110.0,
            kwargs={'index': index},
        )
        env_r = gym.make('MountainCar-evaluate-v0')
        sumreward = 0
        episodes_reward = []
        for _ in range(100):
            while True:
                if done:
                    observation = env_r.reset()
                    sumreward = 0

                    # if _ % 1 == 0:
                # env_r.render()
                action = RL.choose_action(observation, initial = False)
                observation_, reward, done, info = env_r.step(action)
                sumreward += reward
                if done:
                    # print('episode ', i_episode, ' episode steps', episode_steps, 'total steps', np.sum(steps))
                    episodes_reward.append(sumreward)
                    print('weight(kg)', 0.2*index, 'times:', _+1, 'sumreward:', sumreward)
                    break
                observation = observation_
        if index == 0.8:
            gather = [index, np.mean(episodes_reward)]
        else:
            gather = np.vstack((gather, [index, np.mean(episodes_reward)]))
        # env_r.close()
    return gather

log = []
inx_log = []
for _ in range(0, 20):
    if _ == 0:
        log = train(RL_natural, 50000)
    else:
        log = np.c_[log, train(RL_natural, 50000)[:, 1]]
        print(np.shape(log))
# log = train(RL_natural, 100000)
x_axis = log[:, 0]
index = np.mean(log[:, 1:], 0)
# print(np.shape(np.reshape(index, [1, 10])))
# print(np.shape(log[:, 1:]))
index = np.reshape(index, [1, 20])
y_data = np.r_[index, log[:, 1:]]
y_data = y_data.T[np.lexsort(-y_data[::-1,:])].T
y_mean = np.mean(y_data[1:, 0:10], 1)
# y_std = log[:, 2]
y_std = np.std(y_data[1:, 0:10], 1)
y_max = np.max(y_data[1:, 0:10], 1)
y_min = np.min(y_data[1:, 0:10], 1)
np.save('x_axis.npy', x_axis)
np.save('y_mean.npy', y_mean)
np.save('y_std.npy', y_std)
np.save('y_max.npy', y_max)
np.save('y_min.npy', y_min)
plt.plot(x_axis*0.2, y_mean, c='tomato', label='DQN_baseline')
# plt.fill_between(x_axis, y_mean + y_std/2, y_mean - y_std/2, facecolor='tomato', alpha=0.3)
plt.fill_between(x_axis*0.2, y_max, y_min, facecolor='tomato', alpha=0.3)
plt.legend(loc='best')
plt.ylabel('Rewards')
plt.xlabel('Car Weight')
plt.title('Mountain Car')
plt.grid()
plt.show()

