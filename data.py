from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd

train_data_path = 'data/train.csv'
test_data_path = 'data/test.csv'


train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
