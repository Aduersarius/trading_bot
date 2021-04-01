import os
import logging

import numpy as np

from tqdm import tqdm

from .utils import (
    format_currency,
    format_position
)
from .ops import (
    get_state
)


def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=10):
    total_profit = 0
    data_length = len(data)-1
    #data = list(data["close"])
    agent.inventory = []
    avg_loss = []

    state = get_state(data, 0, window_size)

    for t in tqdm(range(1, data_length-window_size), desc='Episode {}/{}'.format(episode, ep_count)):
        reward = 0
        next_state = get_state(data, t, window_size)

        # select an action
        action = agent.act(state)

        # BUY
        if action == 1:
            agent.inventory.append(data["close"][t])

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data["close"][t] - bought_price
            reward = delta  # max(delta, 0)
            total_profit += delta

        # HOLD
        else:
            pass

        done = (t == data_length - window_size)
        agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    if episode % 10 == 0:
        agent.save(episode)

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, window_size, debug):
    total_profit = 0
    #data = list(data["close"])
    data_length = len(data)-1

    history = []
    agent.inventory = []
    
    state = get_state(data, 0, window_size)

    for t in range(1, data_length-window_size):
        reward = 0
        next_state = get_state(data, t, window_size)
        
        # select an action
        action = agent.act(state, is_eval=True)

        # BUY
        if action == 1:
            agent.inventory.append(data["close"][t])

            history.append((data["close"][t], "BUY"))
            if debug:
                logging.debug("Buy at: {}".format(format_currency(data["close"][t])))
        
        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data["close"][t] - bought_price
            reward = delta #max(delta, 0)
            total_profit += delta

            history.append((data["close"][t], "SELL"))
            if debug:
                logging.debug("Sell at: {} | Position: {}".format(
                    format_currency(data["close"][t]), format_position(data["close"][t] - bought_price)))
        # HOLD
        else:
            history.append((data["close"][t], "HOLD"))

        done = (t == data_length - window_size)
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state

    return total_profit, history
