import numpy as np
import pickle

import pandas as pd
import sys
from collections import defaultdict



def get_load(id, timewindow, data):
    """Get current load, reqs per second
    args:
        id: the current request id
        timewindow: the time window to estimate current load
        data: the record data of reqs coming timestamp
    returns:
        current load
    """
    req = data.iloc[id]
    req_start = req['start']
    idx = id - 1
    cnt = 1
    while idx >= 0:
        reqi_start = data.iloc[idx]['start']
        if reqi_start > req_start - 6000:
            cnt += 1
            idx -= 1
        else:
            break
    return cnt







"""
思路：（蒙特卡罗控制）初始一个random的policy：load->bs (state->action)
policy evaluation
policy improvement
"""
def get_policy(Q, nA, epsilon=0.1):
    """According to Q function to produce epsilon-greedy policy
    args:
        Q: Q function, Q[state][action] = value
        nA: the number of actions
        epsilon: for a little chance escaping greedy
    return:
        a policy function
    """
    def policy_fn(observation):
        action_probs = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        action_probs[best_action] += (1.0 - epsilon)
        return action_probs
    
    return policy_fn


def mc_control_epsilon_greedy(data, episode_num, discount_factor=1.0, epsilon=0.1,):
    """
    args:
        data: data[(state, action)] = a list of latency reward
    return:
        a good policy in the form of Q function
    """
    supported_bs = [1, 2, 4, 8, 16, 32, 64] # load is state, bs is action
    supported_load = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Keeps track of sum and count of returns for each
    # state-action pair to calculate an average
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final action-value function
    # state->(actions->values)
    # Initally, all (state-action) owns zero value, so
    # the inital policy according to this Q function
    # is equally a random policy
    Q = defaultdict(lambda: np.zeros(len(supported_bs)))

    policy = get_policy(Q, len(supported_bs))   # The epsilon policy derived from Q function

    for episode_i in range(1, episode_num+1):
        if episode_i % 500 == 0:
            print('\r episode: {}/{}'.format(episode_i, episode_num), end='')
            sys.stdout.flush()
        # Pick a random state
        state = np.random.choice(supported_load)
        probs = policy(state)
        action_id = np.random.choice(np.arange(len(probs)), p=probs)
        action = 2**action_id
        sa_pair = (state, action)
        reward = np.random.choice(data[sa_pair])
        returns_sum[sa_pair] += reward
        returns_count[sa_pair] += 1
        Q[state][action_id] = returns_sum[sa_pair] / returns_count[sa_pair]

    return Q, policy


def train():
    data = defaultdict(list)
    raw_data = pd.read_csv('../data/latency/resnet152.csv')
    for idx, row in raw_data.iterrows():
        data[(row['load'], row['bs'])].append(row['reward'])
    Q, policy = mc_control_epsilon_greedy(data, 20*len(raw_data))
    print(Q)
    save_Q(dict(Q))

def save_Q(Q_func):
    with open('Q_func.pkl', 'wb') as f:
        pickle.dump(Q_func, f, pickle.HIGHEST_PROTOCOL)

train()