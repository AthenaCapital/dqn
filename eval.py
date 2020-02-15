import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from main import *


saved_policy = tf.compat.v2.saved_model.load(policy_dir)

print('Average Return: {}'.format(compute_avg_return(eval_env, saved_policy, num_eval_episodes)))

log = pd.read_csv(log_path)
steps = log['step'].tolist()
returns = log['avg return'].tolist()

plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.show()


render_env(env, eval_env, saved_policy)
