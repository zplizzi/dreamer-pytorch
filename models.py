from typing import List, Tuple
from typing import Optional
from torch import Tensor

import numpy as np
import torch
import torch.distributions
from einops import rearrange
from torch import jit
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.nn import functional as F


def bottle(f, x_tuple):
    """Adapts inputs with shape (T, B, ...) to work with a function f expecting shape (B, ...).

    TODO: this might be clearer as a function wrapper."""
    T, B = x_tuple[0].shape[:2]
    x_tuple = (rearrange(x, "t b ... -> (t b) ...", t=T, b=B) for x in x_tuple)
    y = f(*x_tuple)
    return rearrange(y, "(t b) ... -> t b ...", t=T, b=B)


class TransitionModel(jit.ScriptModule):
    __constants__ = ["min_std_dev"]

    def __init__(
        self,
        belief_size,
        state_size,
        action_size,
        hidden_size,
        embedding_size,
        activation_function="relu",
        min_std_dev=0.1,
    ):
        super().__init__()
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
        self.modules = [
            self.fc_embed_state_action,
            self.fc_embed_belief_prior,
            self.fc_state_prior,
            self.fc_embed_belief_posterior,
            self.fc_state_posterior,
        ]

    # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # a : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    # @jit.script_method
    # def forward(
    #     self,
    #     prev_state: torch.Tensor,
    #     actions: torch.Tensor,
    #     prev_belief: torch.Tensor,
    #     enc_observations: torch.Tensor,
    #     nonterminals: torch.Tensor,
    # ):
    #     """
    #     Computes transitions for an entire sequence.
    #     If enc_observations is given, "training mode" is used where the state posterior is the base of the next
    #     timestep. Else, in "test mode", the prior is rolled out.
    #
    #     Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
    #     Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
    #             torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
    #     """
    #     # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
    #     T = actions.size(0) + 1
    #     # beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = (
    #     #     [torch.empty(0)] * T,
    #     #     [torch.empty(0)] * T,
    #     #     [torch.empty(0)] * T,
    #     #     [torch.empty(0)] * T,
    #     #     [torch.empty(0)] * T,
    #     #     [torch.empty(0)] * T,
    #     #     [torch.empty(0)] * T,
    #     # )
    #
    #     beliefs = []
    #     posterior_means = []
    #     posterior_std_devs = []
    #     posterior_states = []
    #
    #     beliefs.append(prev_belief)
    #     posterior_states.append(prev_state)
    #
    #     # beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
    #     # for t in range(T - 1):
    #     for t in range(10):
    #         # Select appropriate previous state
    #         _state = posterior_states[-1]
    #         # Set _state to 0 if previous transition was terminal
    #         _state = _state * nonterminals[t]
    #
    #         # Compute belief (deterministic hidden state)
    #         # This if f in the paper.
    #         # "beliefs" is h_t in the paper
    #         hidden = F.elu(self.fc_embed_state_action(torch.cat((_state, actions[t]), dim=1)))
    #         belief = self.rnn(hidden, beliefs[-1])
    #
    #         # Compute state prior by applying transition dynamics.
    #         # This is the "stochastic state model" or "prior" in the paper
    #         # The prior has no info on the actual observation.
    #         # TODO: pull this out of the loop and batch it all at the end
    #         # hidden = F.elu(self.fc_embed_belief_prior(beliefs[t + 1]))
    #         # prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
    #         # prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
    #         # prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
    #
    #         # Compute state posterior by applying transition dynamics and using current observation
    #         # This is the "encoder" or "posterior" in the paper
    #         # The posterior differs from the prior in that it also sees the observation.
    #         # Use t_ to deal with different time indexing for observations
    #         t_ = t - 1
    #         hidden = F.elu(
    #             self.fc_embed_belief_posterior(torch.cat([belief, enc_observations[t_ + 1]], dim=1))
    #         )
    #         posterior_mean, _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
    #         posterior_std_dev = F.softplus(_posterior_std_dev) + self.min_std_dev
    #         posterior_state = posterior_mean + posterior_std_dev #* torch.randn_like(posterior_mean)
    #
    #         posterior_states.append(posterior_state)
    #         posterior_means.append(posterior_mean)
    #         posterior_std_devs.append(posterior_std_dev)
    #         beliefs.append(belief)
    #
    #     # Return new hidden states
    #     return (
    #         torch.stack(beliefs[1:], dim=0),
    #         # torch.stack(prior_states[1:], dim=0),
    #         # torch.stack(prior_means[1:], dim=0),
    #         # torch.stack(prior_std_devs[1:], dim=0),
    #         torch.stack(posterior_states[1:], dim=0),
    #         torch.stack(posterior_means[1:], dim=0),
    #         torch.stack(posterior_std_devs[1:], dim=0),
    #     )

    def forward(
            self,
            prev_state: torch.Tensor,
            actions: torch.Tensor,
            prev_belief: torch.Tensor,
            enc_observations: Optional[torch.Tensor] = None,
            nonterminals: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Computes transitions for an entire sequence.
        If enc_observations is given, "training mode" is used where the state posterior is the base of the next
        timestep. Else, in "test mode", the prior is rolled out.

        Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
        Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
                torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        """
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = (
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
        )
        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
        for t in range(T - 1):
            # Select appropriate previous state
            _state = prior_states[t] if enc_observations is None else posterior_states[t]
            # Set _state to 0 if previous transition was terminal
            _state = _state if nonterminals is None else _state * nonterminals[t]

            # Compute belief (deterministic hidden state)
            # This if f in the paper.
            # "beliefs" is h_t in the paper
            hidden = F.elu(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])

            # Compute state prior by applying transition dynamics.
            # This is the "stochastic state model" or "prior" in the paper
            # The prior has no info on the actual observation.
            hidden = F.elu(self.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])

            if enc_observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                # This is the "encoder" or "posterior" in the paper
                # The posterior differs from the prior in that it also sees the observation.
                # Use t_ to deal with different time indexing for observations
                t_ = t - 1
                hidden = F.elu(
                    self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], enc_observations[t_ + 1]], dim=1))
                )
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(
                    posterior_means[t + 1]
                )
        # Return new hidden states
        hidden = [
            torch.stack(beliefs[1:], dim=0),
            torch.stack(prior_states[1:], dim=0),
            torch.stack(prior_means[1:], dim=0),
            torch.stack(prior_std_devs[1:], dim=0),
        ]
        if enc_observations is not None:
            hidden += [
                torch.stack(posterior_states[1:], dim=0),
                torch.stack(posterior_means[1:], dim=0),
                torch.stack(posterior_std_devs[1:], dim=0),
            ]
        return hidden

class SymbolicObservationModel(jit.ScriptModule):
    def __init__(self, observation_size, belief_size, state_size, embedding_size, activation_function="relu"):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, observation_size)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        observation = self.fc3(hidden)
        return observation


class VisualObservationModel(jit.ScriptModule):
    __constants__ = ["embedding_size"]

    def __init__(self, belief_size, state_size, embedding_size, activation_function="relu"):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
        self.modules = [self.fc1, self.conv1, self.conv2, self.conv3, self.conv4]

    @jit.script_method
    def forward(self, belief, state):
        # No nonlinearity here
        hidden = self.fc1(torch.cat([belief, state], dim=1))
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation


def ObservationModel(symbolic, observation_size, belief_size, state_size, embedding_size, activation_function="relu"):
    if symbolic:
        return SymbolicObservationModel(observation_size, belief_size, state_size, embedding_size, activation_function)
    else:
        return VisualObservationModel(belief_size, state_size, embedding_size, activation_function)


class RewardModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, activation_function="relu"):
        # [--belief-size: 200, --hidden-size: 200, --state-size: 30]
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        reward = self.fc3(hidden).squeeze(dim=1)
        return reward


class ValueModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, activation_function="relu"):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4]

    @jit.script_method
    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        reward = self.fc4(hidden).squeeze(dim=1)
        return reward


class ActorModel(jit.ScriptModule):
    def __init__(
        self,
        belief_size,
        state_size,
        hidden_size,
        action_size,
        dist="tanh_normal",
        activation_function="elu",
        min_std=1e-4,
        init_std=5,
        mean_scale=5,
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 2 * action_size)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    @jit.script_method
    def forward(self, belief, state):
        raw_init_std = torch.log(torch.exp(self._init_std) - 1)
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        hidden = self.act_fn(self.fc4(hidden))
        action = self.fc5(hidden).squeeze(dim=1)

        action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
        # TODO: why tanh here? are the valid actions in (-1, 1)? probably
        action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
        action_std = F.softplus(action_std_dev + raw_init_std) + self._min_std
        return action_mean, action_std

    def get_action(self, belief, state, det=False):
        action_mean, action_std = self.forward(belief, state)
        # TODO: understand+simplify this
        dist = Normal(action_mean, action_std)
        dist = TransformedDistribution(dist, TanhBijector())
        dist = torch.distributions.Independent(dist, 1)
        dist = SampleDist(dist)
        if det:
            return dist.mode()
        else:
            return dist.rsample()


class SymbolicEncoder(jit.ScriptModule):
    def __init__(self, observation_size, embedding_size, activation_function="relu"):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(observation_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, observation):
        hidden = self.act_fn(self.fc1(observation))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.fc3(hidden)
        return hidden


# TODO: isn't the encoder supposed to have a sampling step?
class VisualEncoder(jit.ScriptModule):
    __constants__ = ["embedding_size"]

    def __init__(self, embedding_size, activation_function="relu"):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        # TODO: huh
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
        self.modules = [self.conv1, self.conv2, self.conv3, self.conv4]

    @jit.script_method
    def forward(self, observation):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        return hidden


def Encoder(symbolic, observation_size, embedding_size, activation_function="relu"):
    if symbolic:
        return SymbolicEncoder(observation_size, embedding_size, activation_function)
    else:
        return VisualEncoder(embedding_size, activation_function)


# "atanh", "TanhBijector" and "SampleDist" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


class TanhBijector(torch.distributions.Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True
        self.domain = torch.distributions.constraints.real
        self.codomain = torch.distributions.constraints.interval(-1.0, 1.0)

    @property
    def sign(self):
        return 1.0

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where((torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y)
        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (np.log(2) - x - F.softplus(-2.0 * x))


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        sample = dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()
