from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import matplotlib.pyplot as plt

from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import policy_saver
from tf_agents.trajectories import trajectory

from main import *


replay_buffer_max_length = 1000000
collect_steps_per_iteration = 10

checkpoint_dir = 'checkpoint'

num_iterations = 5000
batch_size = 256

log_interval = 100
eval_interval = 1000

log_path = checkpoint_dir + '/log.csv'
plot_path = checkpoint_dir + '/plot.png'


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--iter', type=int, default=num_iterations)
arg_parser.add_argument('--batch', type=int, default=batch_size)
args = arg_parser.parse_args()

num_iterations = args.iter
batch_size = args.batch


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=train_env.batch_size, max_length=replay_buffer_max_length)

collect_driver = dynamic_step_driver.DynamicStepDriver(train_env, agent.collect_policy, observers=[replay_buffer.add_batch], num_steps=collect_steps_per_iteration)

train_checkpointer = common.Checkpointer(ckpt_dir=checkpoint_dir, max_to_keep=1, agent=agent, policy=agent.policy, replay_buffer=replay_buffer, global_step=global_step)
train_checkpointer.initialize_or_restore()

tf_policy_saver = policy_saver.PolicySaver(agent.policy)


collect_driver.run()

dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)

iterator = iter(dataset)


try:
    df = pd.read_csv(log_path, index_col='step')
except:
    df = pd.DataFrame()
    df.index.name = 'step'


agent.train = common.function(agent.train)

for _ in range(num_iterations):
    collect_driver.run()

    experience, unused_info = next(iterator)
    loss = agent.train(experience).loss.numpy()

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step {}: loss = {}'.format(step, loss))

        df.loc[step, 'loss'] = loss

    if step % eval_interval == 0:
        avg_reward = eval_policy(agent.policy)

        print('average reward: {:.6f}'.format(avg_reward))

        if not 'avg reward' in df or avg_reward > max(df['avg reward'].dropna()):
            tf_policy_saver.save(policy_dir)

        df.loc[step, 'avg reward'] = avg_reward

train_checkpointer.save(global_step)


df.to_csv(log_path)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(df['loss'])
plt.xlabel('step')
plt.title('loss')

plt.subplot(1, 2, 2)
plt.plot(df['avg reward'].dropna())
plt.xlabel('step')
plt.title('avg reward')

plt.tight_layout()
plt.savefig(plot_path)
plt.show()
