import numpy as np
import torch

from env import postprocess_observation
from env import preprocess_observation


class ExperienceReplay:
    def __init__(self, size, symbolic_env, observation_size, action_size, device):
        """
        Stores experience in a sequential buffer, where each piece of experience is a single environment timestep.
        All episodes simply directly follow the previous in this 1-d buffer.
        TODO: is this how the original implementation did it?
        # NOPE

        TODO: document the norm of which time indexes are used for obs, action, reward.
        appears to be (observation, action, resulting reward, resulting done state)
        (o_t, a_t, r_t+1, terminal_t+1)

        - size: the number of pieces of experience
        - observation_size: the size of a single piece of experience (observation). hardcoded to one frame if not symbolic.
        """
        self.device = device
        self.symbolic_env = symbolic_env
        self.size = size
        # self.observations = np.empty(
        self.observations = torch.zeros(
            (size, observation_size) if symbolic_env else (size, 3, 64, 64),
            # dtype=np.float32 if symbolic_env else np.uint8,
            dtype=torch.uint8,
        )
        self.actions = np.empty((size, action_size), dtype=np.float32)
        self.rewards = np.empty((size,), dtype=np.float32)
        # Set to 1 if not the final observation in a sequence. 0 if final.
        # TODO: why is there the 1 in the shape here but not in rewards..
        self.nonterminals = np.empty((size, 1), dtype=np.float32)
        # Pointer to current memory slot to write to (circular buffer)
        self.idx = 0
        self.full = False
        self.steps = 0
        self.episodes = 0

    def append(self, observation, action, reward, done):
        if self.symbolic_env:
            self.observations[self.idx] = observation.numpy()
        else:
            # Decentre and discretise visual observations (to save memory)
            self.observations[self.idx] = torch.tensor(postprocess_observation(observation.numpy())).to(torch.uint8)
        self.actions[self.idx] = action.numpy()
        self.rewards[self.idx] = reward
        self.nonterminals[self.idx] = not done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps = self.steps + 1
        self.episodes = self.episodes + (1 if done else 0)

    def _sample_idx(self, L):
        """Returns a sequence of indexes for a valid single sequence chunk uniformly sampled from the memory.
        - L: sequence length

        TODO: this method of sampling will result in some sequences containing two episodes. Is that ok?
        could relatively easily make a check to reject such samples, although we might have to adjust weighting
        to get enough end samples like in dreamerv2
        """
        valid_idx = False
        while not valid_idx:
            # if full, we will wrap around.
            idx = np.random.randint(0, self.size if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.size
            # Make sure data does not cross the memory index
            valid_idx = not self.idx in idxs[1:]
        return idxs

    def _retrieve_batch(self, idxs):
        B, L = idxs.shape
        # Unroll indices
        vec_idxs = idxs.transpose().reshape(-1)
        # now has shape L * batch_size

        # observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
        # this one line takes .03 seconds for B*L = 50 * 50
        # observations = self.observations[vec_idxs]

        idx = torch.tensor(vec_idxs)
        observations = torch.index_select(self.observations, dim=0, index=idx)
        # Move to GPU first so subsequent operations will happen faster. This is a big tensor! 30MB?
        observations = observations.to(self.device)
        observations = observations.to(torch.float32)

        if not self.symbolic_env:
            # Undo discretisation for visual observations
            observations = preprocess_observation(observations)
        return (
            observations.reshape(L, B, *observations.shape[1:]),
            self.actions[vec_idxs].reshape(L, B, -1),
            self.rewards[vec_idxs].reshape(L, B),
            self.nonterminals[vec_idxs].reshape(L, B, 1),
        )

    # Returns a batch of sequence chunks uniformly sampled from the memory
    def sample(self, B, L):
        idx_batch = np.asarray([self._sample_idx(L) for _ in range(B)])
        batch = self._retrieve_batch(idx_batch)
        return [torch.as_tensor(item).to(device=self.device) for item in batch]
