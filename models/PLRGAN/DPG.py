import numpy as np
from numpy_ringbuffer import RingBuffer
import torch

class DPG:
    memory_bank_size = 10 ** 6
    def __init__(self, z_size, z_prior, stddv, hot_start, n_samples, batch_size, test_passwords, device, static=False):
        self.DYNAMIC = False
        self.device = device
        self.hot_start = hot_start
        self.z_size = z_size
        self.stddv_p = z_prior
        self.stddv = stddv
        self.STATIC = static
        self.init_att_size = 0
        self.n_samples = n_samples

        self.batch_size = batch_size

        self.matched_i = 0

        self.test_passwords = test_passwords

        if not self.STATIC:
            self.guessed_z = RingBuffer(capacity=self.memory_bank_size, dtype=(np.float32, self.z_size))

    def __call__(self, z, x):
        if not self.init_att_size:
            self.init_att_size = len(self.test_passwords)

        # ASSUMPTION: test set composed of unique passwords
        if x in self.test_passwords:
            self.matched_i += 1
            self.test_passwords.remove(x)
            if not self.STATIC:
                self.guessed_z.append(z.cpu())

            if not self.STATIC and self.matched_i / self.init_att_size > self.hot_start and not self.DYNAMIC:
                print("DYNAMIC starts now ....")
                self.DYNAMIC = True

    def guess(self):
        if self.DYNAMIC and len(self.guessed_z):
            idxs = np.random.randint(0, len(self.guessed_z), self.batch_size, np.int32)
            gi = self.guessed_z[idxs]
            z = torch.normal(torch.FloatTensor(gi).to(self.device), self.stddv)
        else:
            z = torch.normal(0, self.stddv_p, size=(self.batch_size, self.z_size))
        return z
