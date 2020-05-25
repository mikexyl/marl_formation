import numpy as np
from baselines.ddpg.memory import Memory as DMemory

class Memory(DMemory):
    def __init__(self, maxlen, shape, dtype='float32'):
        super(Memory, self).__init__(maxlen, shape, dtype)

    def sample(self, batch_size, id=None):
        if id is None:
            return super(Memory, self).sample(batch_size)
        # else