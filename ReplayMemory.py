import numpy as np
from collections import deque
import random

class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.mem = deque(maxlen=self.max_size)

    def push(self, state, action, reward, next_state):
        experience = (state, action, np.array([reward]), next_state)
        self.mem.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []

        experiences = random.sample(self.mem, batch_size)

        for experience in experiences:
            state, action, reward, next_state = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)

        return state_batch, action_batch, reward_batch, next_state_batch

    def length(self):
        return len(self.mem)
