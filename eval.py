from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from main import *


policy = tf.compat.v2.saved_model.load(policy_dir)
print('average reward: {}'.format(eval_policy(policy)))

"""
for _ in range(5):
    time_step = env.reset()
    while not time_step.is_last():
        time_step = env.step(policy.action(time_step).action)
    env.render()
"""
