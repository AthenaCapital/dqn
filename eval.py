from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
from main import *


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--render', default=False, action='store_true')
args = arg_parser.parse_args()

render = args.render


policy = tf.compat.v2.saved_model.load(policy_dir)

print('average reward: {:.6f}'.format(eval_policy(policy, render=render)))
