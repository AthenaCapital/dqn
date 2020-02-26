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

num_eval_episodes = 30

network_shape = (128, 128,)
learning_rate = 1e-3

tau = 0.1
gradient_clipping = 1

policy_dir = 'policy'


train_py_env = gym_wrapper.GymWrapper(StockTradingEnv(df=train_data, window_size=window_size))
eval_py_env = gym_wrapper.GymWrapper(StockTradingEnv(df=eval_data, window_size=window_size, eval=True))

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


def eval_policy(policy, render=False):
    total_reward = 0
    for _ in range(num_eval_episodes):
        time_step = eval_env.reset()
        episode_reward = 0
        while not time_step.is_last():
            time_step = eval_env.step(policy.action(time_step).action)
            episode_reward += time_step.reward

        if render:
            eval_py_env.render()

        total_reward += episode_reward

    eval_env.close()
    eval_py_env.close()

    avg_reward = total_reward / num_eval_episodes

    return avg_reward.numpy()[0]


q_net = q_network.QNetwork(train_env.observation_spec(), train_env.action_spec(), fc_layer_params=network_shape)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

global_step = tf.compat.v1.train.get_or_create_global_step()

agent = dqn_agent.DqnAgent(train_env.time_step_spec(), train_env.action_spec(), q_network=q_net, optimizer=optimizer, target_update_tau=tau, td_errors_loss_fn=common.element_wise_squared_loss, gradient_clipping=gradient_clipping, train_step_counter=global_step)
agent.initialize()
