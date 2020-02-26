from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd


train_data_path = 'data/train.csv'
eval_data_path = 'data/eval.csv'


train_data = pd.read_csv(train_data_path, index_col=0)
eval_data = pd.read_csv(eval_data_path, index_col=0)
