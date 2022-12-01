import logging
import math
import os

import numpy as np

from examples.models.rule_model.random import RandomActor


def init_logger(filename):
    """initialize logger config

    Parameters
    ----------
    filename : str
        filename of the log
    """
    logging.basicConfig(level=logging.INFO, filename=filename + ".log")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)


class EpisodesBufferEntry:
    """Entry for episode buffer"""

    def __init__(self):
        self.views = []
        self.features = []
        self.actions = []
        self.rewards = []
        self.terminal = False

    def append(self, view, feature, action, reward, alive):
        self.views.append(view.copy())
        self.features.append(feature.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        if not alive:
            self.terminal = True


class EpisodesBuffer:
    """Replay buffer to store a whole episode for all agents
    one entry for one agent
    """

    def __init__(self, capacity):
        self.buffer = {}
        self.capacity = capacity
        self.is_full = False

    def record_step(self, ids, obs, acts, rewards, alives):
        """record transitions (s, a, r, terminal) in a step"""
        buffer = self.buffer
        index = np.random.permutation(len(ids))

        if self.is_full:  # extract loop invariant in else part
            for i in range(len(ids)):
                entry = buffer.get(ids[i])
                if entry is None:
                    continue
                entry.append(obs[0][i], obs[1][i], acts[i], rewards[i], alives[i])
        else:
            for i in range(len(ids)):
                i = index[i]
                entry = buffer.get(ids[i])
                if entry is None:
                    if self.is_full:
                        continue
                    else:
                        entry = EpisodesBufferEntry()
                        buffer[ids[i]] = entry
                        if len(buffer) >= self.capacity:
                            self.is_full = True

                entry.append(obs[0][i], obs[1][i], acts[i], rewards[i], alives[i])

    def reset(self):
        """clear replay buffer"""
        self.buffer = {}
        self.is_full = False

    def episodes(self):
        """get episodes"""
        return self.buffer.values()


# decay schedulers
def exponential_decay(now_step, total_step, final_value, rate):
    """exponential decay scheduler"""
    decay = math.exp(math.log(final_value) / total_step**rate)
    return max(final_value, 1 * decay ** (now_step**rate))


def linear_decay(now_step, total_step, final_value):
    """linear decay scheduler"""
    decay = (1 - final_value) / total_step
    return max(final_value, 1 - decay * now_step)


def piecewise_decay(now_step, anchor, anchor_value):
    """piecewise linear decay scheduler

    Parameters
    ---------
    now_step : int
        current step
    anchor : list of integer
        step anchor
    anchor_value: list of float
        value at corresponding anchor
    """
    i = 0
    while i < len(anchor) and now_step >= anchor[i]:
        i += 1

    if i == len(anchor):
        return anchor_value[-1]
    else:
        return anchor_value[i - 1] + (now_step - anchor[i - 1]) * (
            (anchor_value[i] - anchor_value[i - 1]) / (anchor[i] - anchor[i - 1])
        )


# eval observation set generator
def sample_observation(env, handles, n_obs=-1, step=-1):
    """Sample observations by random actors.
    These samples can be used for evaluation

    Parameters
    ----------
    env : environment
    handles: list of handle
    n_obs : int
        number of observation
    step : int
        maximum step

    Returns
    -------
    ret : list of raw observation
        raw observation for every group
        the format of raw observation is tuple(view, feature)
    """
    models = [RandomActor(env, handle) for handle in handles]

    n = len(handles)
    views = [[] for _ in range(n)]
    features = [[] for _ in range(n)]

    done = False
    step_ct = 0
    while not done:
        obs = [env.get_observation(handle) for handle in handles]
        ids = [env.get_agent_id(handle) for handle in handles]

        for i in range(n):
            act = models[i].infer_action(obs[i], ids[i])
            env.set_action(handles[i], act)

        done = env.step()
        env.clear_dead()

        # record steps
        for i in range(n):
            views[i].append(obs[i][0])
            features[i].append(features[i][1])

        if step != -1 and step_ct > step:
            break

        if step_ct % 100 == 0:
            print("sample step %d" % step_ct)

        step_ct += 1

    for i in range(n):
        views[i] = np.array(views[i], dtype=np.float32).reshape(
            (-1,) + env.get_view_space(handles[i])
        )
        features[i] = np.array(features[i], dtype=np.float32).reshape(
            (-1,) + env.get_feature_space(handles[i])
        )

    if n_obs != -1:
        for i in range(n):
            views[i] = views[i][np.random.choice(np.arange(views[i].shape[0]), n_obs)]
            features[i] = features[i][
                np.random.choice(np.arange(features[i].shape[0]), n_obs)
            ]

    ret = [(v, f) for v, f in zip(views, features)]
    return ret


def has_gpu():
    """check where has a nvidia gpu"""
    ret = os.popen("nvidia-smi -L 2>/dev/null").read()
    return ret.find("GPU") != -1
