import argparse
import shutil
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import policy_saver
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from main import *


num_iterations = 10000
batch_size = 64

log_interval = 100
eval_interval = 1000


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--init', default=False, action='store_true')
arg_parser.add_argument('--iter', type=int, default=num_iterations)
arg_parser.add_argument('--batch', type=int, default=batch_size)

args = arg_parser.parse_args()
init = args.init
num_iterations = args.iter
batch_size = args.batch


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=train_env.batch_size, max_length=replay_buffer_max_length)

collect_driver = dynamic_step_driver.DynamicStepDriver(train_env, agent.collect_policy, observers=[replay_buffer.add_batch], num_steps=collect_steps_per_iteration)

if init:
    shutil.rmtree(checkpoint_dir)
train_checkpointer = common.Checkpointer(ckpt_dir=checkpoint_dir, max_to_keep=1, agent=agent, policy=agent.policy, replay_buffer=replay_buffer, global_step=global_step)
train_checkpointer.initialize_or_restore()

tf_policy_saver = policy_saver.PolicySaver(agent.policy)


collect_driver.run()

dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)

iterator = iter(dataset)


if init:
    steps = [agent.train_step_counter.numpy()]
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('Average Return: {}'.format(avg_return))
    returns = [avg_return]

    tf_policy_saver.save(policy_dir)
else:
    log = pd.read_csv(log_path)
    steps = log['step'].tolist()
    returns = log['avg return'].tolist()
    print('Average Return: {}'.format(returns[-1]))


agent.train = common.function(agent.train)

for _ in range(num_iterations):
    for _ in range(collect_steps_per_iteration):
        collect_driver.run()

    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step {}: loss = {}'.format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)

        if avg_return > max(returns):
            tf_policy_saver.save(policy_dir)

        steps.append(step)
        returns.append(avg_return)

        print('Average Return: {}'.format(avg_return))


train_checkpointer.save(global_step)

log = pd.DataFrame({'step': steps, 'avg return': returns})
log.to_csv(log_path)
