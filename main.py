import tensorflow as tf
import pandas as pd

import gym
import gym_anytrading

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.utils import common


window_size = 10
fc_layer_params = (128,128,)

collect_steps_per_iteration = 1
num_eval_episodes = 1

replay_buffer_max_length = 100000
learning_rate = 1e-3

checkpoint_dir = 'checkpoint'
policy_dir = 'policy'

data_path = 'data/AAPL.csv'
log_path = 'log/log.csv'

env_name = 'stocks-v0'


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward

        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def render_env(py_environment, tf_environment, policy, num_episodes=10):
    for _ in range(num_episodes):
        py_environment.reset()
        time_step = tf_environment.reset()

        while not time_step.is_last():
            py_environment.render()
            action_step = policy.action(time_step)
            time_step = tf_environment.step(action_step.action)

    py_environment.close()
    tf_environment.close()


df = pd.read_csv(data_path)

env = gym_wrapper.GymWrapper(gym.make(env_name, df=df, window_size=window_size, frame_bound=(window_size, len(df))))

train_env = tf_py_environment.TFPyEnvironment(env)
eval_env = tf_py_environment.TFPyEnvironment(env)


q_net = q_network.QNetwork(train_env.observation_spec(), train_env.action_spec(), fc_layer_params=fc_layer_params)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

global_step = tf.compat.v1.train.get_or_create_global_step()

agent = dqn_agent.DqnAgent(train_env.time_step_spec(), train_env.action_spec(), q_network=q_net, optimizer=optimizer, td_errors_loss_fn=common.element_wise_squared_loss, train_step_counter=global_step)
agent.initialize()
