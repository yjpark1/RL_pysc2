from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random

import numpy as np

# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')
Trajectory = namedtuple('Trajectory', 'state, action, reward, terminal')
EpisodicTimestep = namedtuple('EpisodicTimestep', 'observation, action, reward, terminal')
EpisodicTimestepAcer = namedtuple('EpisodicTimestepAcer', 'observation, action, reward, policy, terminal')


def sample_batch_indexes(low, high, size):
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        try:
            r = xrange(low, high)
        except NameError:
            r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn(
            'Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs


class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.length + idx
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def zeroed_observation(observation):
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.


class Memory(object):
    def __init__(self, window_length=1, ignore_episode_boundaries=False):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

    def sample(self, batch_size, batch_idxs=None):
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal, training=True):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)

    def get_recent_state(self, current_observation):
        # This code is slightly complicated by the fact that subsequent observations might be
        # from different episodes. We ensure that an experience never spans multiple episodes.
        # This is probably not that important in practice but it seems cleaner.
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx - 1] if current_idx - 1 >= 0 else False
            if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            state.insert(0, self.recent_observations[current_idx])
        while len(state) < self.window_length:
            state.insert(0, zeroed_observation(state[0]))
        return state

    def get_config(self):
        config = {
            'window_length': self.window_length,
            'ignore_episode_boundaries': self.ignore_episode_boundaries,
        }
        return config


class SequentialMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(SequentialMemory, self).__init__(**kwargs)

        self.limit = limit

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        if batch_idxs is None:
            # Draw random indexes such that we have at least a single entry before each
            # index.
            batch_idxs = sample_batch_indexes(0, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = sample_batch_indexes(1, self.nb_entries, size=1)[0]
                terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            assert 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                current_terminal = self.terminals[current_idx - 1] if current_idx - 1 > 0 else False
                if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size
        return experiences

    def append(self, observation, action, reward, terminal, training=True):
        super(SequentialMemory, self).append(observation, action, reward, terminal, training=training)

        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    @property
    def nb_entries(self):
        return len(self.observations)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config

    @property
    def is_episodic(self):
        return False


class EpisodicMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(EpisodicMemory, self).__init__(**kwargs)

        self.limit = limit
        self.episodes = RingBuffer(limit)
        self.terminal = False

    def sample(self, batch_size, batch_idxs=None):
        if len(self.episodes) <= 1:
            # We don't have a complete episode yet ...
            return []

        if batch_idxs is None:
            # Draw random indexes such that we never use the last episode yet, which is
            # always incomplete by definition.
            batch_idxs = sample_batch_indexes(0, self.nb_entries - 1, size=batch_size)
        assert np.min(batch_idxs) >= 0
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create sequence of experiences.
        sequences = []
        for idx in batch_idxs:
            episode = self.episodes[idx]
            while len(episode) == 0:
                idx = sample_batch_indexes(0, self.nb_entries, size=1)[0]

            # Bootstrap state.
            running_state = deque(maxlen=self.window_length)
            for _ in range(self.window_length - 1):
                running_state.append(np.zeros(episode[0].observation.shape))
            assert len(running_state) == self.window_length - 1

            states, rewards, actions, terminals = [], [], [], []
            terminals.append(False)
            for idx, timestep in enumerate(episode):
                running_state.append(timestep.observation)
                states.append(np.array(running_state))
                rewards.append(timestep.reward)
                actions.append(timestep.action)
                terminals.append(timestep.terminal)  # offset by 1, see `terminals.append(False)` above
            assert len(states) == len(rewards)
            assert len(states) == len(actions)
            assert len(states) == len(terminals) - 1

            # Transform into experiences (to be consistent).
            sequence = []
            for idx in range(len(episode) - 1):
                state0 = states[idx]
                state1 = states[idx + 1]
                reward = rewards[idx]
                action = actions[idx]
                terminal1 = terminals[idx + 1]
                experience = Experience(state0=state0, state1=state1, reward=reward, action=action, terminal1=terminal1)
                sequence.append(experience)
            sequences.append(sequence)
            assert len(sequence) == len(episode) - 1
        assert len(sequences) == batch_size
        return sequences

    def append(self, observation, action, reward, terminal, training=True):
        super(EpisodicMemory, self).append(observation, action, reward, terminal, training=training)

        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            timestep = EpisodicTimestep(observation=observation, action=action, reward=reward, terminal=terminal)
            if len(self.episodes) == 0:
                self.episodes.append([])  # first episode
            self.episodes[-1].append(timestep)
            if self.terminal:
                self.episodes.append([])
            self.terminal = terminal

    @property
    def nb_entries(self):
        return len(self.episodes)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config

    @property
    def is_episodic(self):
        return True


class EpisodicMemoryAcer(Memory):
    def __init__(self, limit, **kwargs):
        super(EpisodicMemoryAcer, self).__init__(**kwargs)

        self.limit = limit
        self.episodes = RingBuffer(limit)
        self.terminal = False

    def sample(self, batch_size, batch_idxs=None):
        if len(self.episodes) <= 1:
            # We don't have a complete episode yet ...
            return []

        if batch_idxs is None:
            # Draw random indexes such that we never use the last episode yet, which is
            # always incomplete by definition.
            batch_idxs = sample_batch_indexes(0, self.nb_entries - 1, size=batch_size)
        assert np.min(batch_idxs) >= 0
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create sequence of experiences.
        sequences = []
        for idx in batch_idxs:
            episode = self.episodes[idx]
            sequences.append(episode)
        assert len(sequences) == batch_size
        return sequences

    def append(self, observation, action, reward, policy, terminal, training=True):
        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            timestep = EpisodicTimestepAcer(observation=observation, action=action,
                                            reward=reward, policy=policy, terminal=terminal)
            if len(self.episodes) == 0:
                self.episodes.append([])  # first episode
            self.episodes[-1].append(timestep)
            if terminal:
                self.episodes.append([])


    @property
    def nb_entries(self):
        return len(self.episodes)

    def get_config(self):
        config = super(EpisodicMemoryAcer, self).get_config()
        config['limit'] = self.limit
        return config

    @property
    def is_episodic(self):
        return True


class SingleEpisodeMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(SingleEpisodeMemory, self).__init__(**kwargs)

        self.limit = limit

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def clear(self):
        self.actions = RingBuffer(self.limit)
        self.rewards = RingBuffer(self.limit)
        self.terminals = RingBuffer(self.limit)
        self.observations = RingBuffer(self.limit)

    def sample(self):
        # Get indexes such that we have at least a single entry before each
        # index.
        batch_idxs = np.arange(self.nb_entries)
        # batch_idxs = np.array(batch_idxs) + 1

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            state = self.observations[idx]
            action = self.actions[idx]
            reward = self.rewards[idx]
            terminal = self.terminals[idx]
            experiences.append(Trajectory(state=state, action=action, reward=reward,
                                          terminal=terminal))

        return experiences

    def append(self, observation, action, reward, terminal, training=True):
        super(SingleEpisodeMemory, self).append(observation, action, reward, terminal, training=training)

        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    @property
    def nb_entries(self):
        return len(self.observations)

    def get_config(self):
        config = super(SingleEpisodeMemory, self).get_config()
        config['limit'] = self.limit
        return config

    @property
    def is_episodic(self):
        return True


if __name__ == '__main__':
    action_shape = {'categorical': (549,),
                    'screen1': (1, 64, 64),
                    'screen2': (1, 64, 64)}
    observation_shape = {'minimap': (7, 64, 64),
                         'screen': (17, 64, 64),
                         'nonspatial': (549,)}

    # 1) test SequentialMemory
    mem = SequentialMemory(limit=10, window_length=1)

    # insert
    for r in range(10):
        obs = {}
        for k, v in observation_shape.items():
            obs[k] = np.random.uniform(size=v)

        action = {}
        for k, v in action_shape.items():
            action[k] = np.random.uniform(size=v)

        mem.append(obs, action, r, terminal=False, training=True)

    # sample
    x = mem.sample(batch_size=3)

    # 2) test SingleEpisodeMemory
    mem = SingleEpisodeMemory(limit=10, window_length=1)

    # insert
    for r in range(10):
        obs = {}
        for k, v in observation_shape.items():
            obs[k] = np.random.uniform(size=v)

        action = {}
        for k, v in action_shape.items():
            action[k] = np.random.uniform(size=v)

        mem.append(obs, action, r, terminal=False, training=True)

    # sample
    x = mem.sample()

    # 3) test EpisodicMemory
    mem = EpisodicMemory(limit=10)

    # insert
    for e in range(5):
        for r in range(10):
            obs = {}
            for k, v in observation_shape.items():
                obs[k] = np.random.uniform(size=v)

            action = {}
            for k, v in action_shape.items():
                action[k] = np.random.uniform(size=v)

            d = 1. if r is 9 else 0.
            mem.append(obs, action, e, terminal=d, training=True)


    # sample
    x = mem.sample(2)
    x = mem.sample(batch_size=1, batch_idxs=[mem.nb_entries-1])

    # 4) test EpisodicMemoryAcer
    mem = EpisodicMemoryAcer(limit=10)

    # insert
    for e in range(5):
        d = False
        for r in range(10):
            obs = {}
            for k, v in observation_shape.items():
                obs[k] = np.random.uniform(size=v)

            action = {}
            for k, v in action_shape.items():
                action[k] = np.random.uniform(size=v)

            d = True if r is 9 else False
            mem.append(obs, action, e, action, terminal=d, training=True)
            print(r, end=',')
        print('')

    # sample
    for i in range(5):
        x = mem.sample(batch_size=1, batch_idxs=[i])
        print(len(x[0]))

    x = mem.sample(batch_size=1, batch_idxs=[0])[0]
    x[-2]
    e = mem.sample(2)
    np.save('replays.npy', e)

    import pickle
    def save_object(obj, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


    # sample usage
    save_object(mem, 'mem.pkl')

    with open('mem.pkl', 'rb') as input:
        mem_test = pickle.load(input)