from __future__ import absolute_import, division, print_function, unicode_literals

import random
from datetime import time
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gym
from gym import spaces


class Actions(Enum):
    Sell = 0
    Buy = 1


class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class StockTradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size):
        self.df = df
        self.window_size = window_size

        self.date_times, self.prices, self.features = self.process_data()
        self.observation_shape = (self.window_size, self.features.shape[1])
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)

        self.start_tick = None
        self.end_tick = None
        self.done = None
        self.current_tick = None
        self.last_trade_tick = None
        self.position = None
        self.position_history = None
        self.holdings = None
        self.total_reward = None


    def reset(self):
        self.start_tick, self.end_tick = self.get_tick_window()
        self.done = False
        self.current_tick = self.start_tick
        self.last_trade_tick = self.current_tick - 1
        self.position = Positions.Long
        self.position_history = [self.position]
        self.holdings = 1
        self.total_reward = 0
        return self.get_observation()


    def step(self, action):
        self.done = False
        self.current_tick += 1

        if self.current_tick == self.end_tick:
            self.done = True

        trade = (action == Actions.Buy.value and self.position == Positions.Short) or (action == Actions.Sell.value and self.position == Positions.Long)

        step_reward = self.calculate_reward(trade)
        self.total_reward += step_reward

        if trade:
            self.position = self.position.opposite()
            self.last_trade_tick = self.current_tick

        self.position_history.append(self.position)

        observation = self.get_observation()
        info = dict(
            total_reward = self.total_reward,
            position = self.position.value
        )

        return observation, step_reward, self.done, info


    def get_observation(self):
        return self.features[(self.current_tick - self.window_size):self.current_tick]


    def render(self, mode='human'):

        plt.plot(self.prices[self.start_tick - self.window_size:self.end_tick])

        short_ticks = []
        long_ticks = []
        for i in range(len(self.position_history)):
            if self.position_history[i] == Positions.Short:
                short_ticks.append(self.start_tick + i)
            else:
                long_ticks.append(self.start_tick + i)

        plt.plot(short_ticks, self.prices[short_ticks], 'r.')
        plt.plot(long_ticks, self.prices[long_ticks], 'g.')

        plt.suptitle("total reward: {:.6f}".format(self.total_reward))

        plt.show()


    def process_data(self):
        features = self.df
        date_times = pd.to_datetime(features.pop('DateTime'))
        prices = features.loc[:, 'Close']

        features_mean = features.mean(axis=0)
        features_std = features.std(axis=0)
        features = (features - features_mean) / features_std

        return date_times, prices, features

    def get_tick_window(self):
        market_open_indexes = []
        for i in range(0, len(self.date_times), 30):
            if self.date_times[i].time() == time(hour=9, minute=30, second=0):
                market_open_indexes.append(i)

        market_open = random.randrange(len(market_open_indexes))
        market_open_index = market_open_indexes[market_open]

        if market_open == len(market_open_indexes) - 1:
            market_close_index = len(self.date_times) - 1
        else:
            market_close_index = market_open_indexes[market_open + 1] - 1

        return market_open_index + self.window_size, market_close_index


    def calculate_reward(self, trade):
        last_trade_holdings = self.holdings
        if trade or self.done:
            current_price = self.prices[self.current_tick]
            last_trade_price = self.prices[self.last_trade_tick]
            if self.position == Positions.Long:
                self.holdings *= current_price / last_trade_price
            else:
                self.holdings *= last_trade_price / current_price
        return self.holdings - last_trade_holdings


    def max_reward(self):
        current_tick = self.start_tick
        last_trade_tick = current_tick - 1
        holdings = 1

        while current_tick <= self.end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self.end_tick and self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self.end_tick and self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            current_price = self.prices[current_tick - 1]
            last_trade_price = self.prices[last_trade_tick]
            if position == Positions.Long:
                holdings *= current_price / last_trade_price
            else:
                holdings *= last_trade_price / current_price

            last_trade_tick = current_tick - 1

        return holdings - 1
