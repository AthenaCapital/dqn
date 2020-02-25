from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.utils import common

from data import *
from stock_trading_env import *


window_size = 10
network_shape = (128,128,)

num_eval_episodes = 5

replay_buffer_max_length = 100000
learning_rate = 1e-3

checkpoint_dir = 'checkpoint'
log_path = checkpoint_dir + '/log.csv'
policy_dir = 'policy'


env = gym_wrapper.GymWrapper(StockTradingEnv(df=train_data, window_size=window_size))

train_env = tf_py_environment.TFPyEnvironment(env)
eval_env = tf_py_environment.TFPyEnvironment(env)


def eval_policy(policy):
    total_reward = 0
    for _ in range(num_eval_episodes):
        time_step = eval_env.reset()
        episode_reward = 0
        while not time_step.is_last():
            time_step = eval_env.step(policy.action(time_step).action)
            episode_reward += time_step.reward
        total_reward += episode_reward
    avg_reward = total_reward / num_eval_episodes
    return avg_reward.numpy()[0]


q_net = q_network.QNetwork(train_env.observation_spec(), train_env.action_spec(), fc_layer_params=network_shape)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

global_step = tf.compat.v1.train.get_or_create_global_step()

agent = dqn_agent.DqnAgent(train_env.time_step_spec(), train_env.action_spec(), q_network=q_net, optimizer=optimizer, td_errors_loss_fn=common.element_wise_squared_loss, train_step_counter=global_step)
agent.initialize()
