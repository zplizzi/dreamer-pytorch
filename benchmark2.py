import argparse
import os

from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F


# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=50, metavar="B", help="Batch size")
parser.add_argument("--chunk-size", type=int, default=50, metavar="L", help="Chunk size")
parser.add_argument("--embedding-size", type=int, default=1024, metavar="E", help="Observation embedding size")
parser.add_argument("--hidden-size", type=int, default=200, metavar="H", help="Hidden size")
parser.add_argument("--belief-size", type=int, default=200, metavar="H", help="Belief/hidden size")
parser.add_argument("--state-size", type=int, default=30, metavar="Z", help="State/latent size")
args = parser.parse_args()


# Setup
import time
results_dir = os.path.join("results", "{}".format(time.time()))
os.makedirs(results_dir, exist_ok=True)
args.device = torch.device("cuda")

class TransitionModel(nn.Module):
    # class TransitionModel(jit.ScriptModule):
    __constants__ = ["min_std_dev"]

    def __init__(
            self,
            belief_size,
            state_size,
            action_size,
            hidden_size,
            embedding_size,
            min_std_dev=0.1,
    ):
        super().__init__()
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
        self.modules = [
            self.fc_embed_state_action,
            self.fc_embed_belief_posterior,
            self.fc_state_posterior,
        ]

    # @jit.script_method
    def forward(
            self,
            prev_state: torch.Tensor,
            actions: torch.Tensor,
            prev_belief: torch.Tensor,
            enc_observations: torch.Tensor,
            nonterminals: torch.Tensor,
    ):
        # for t in range(2):
        t = 0
        _state = prev_state * nonterminals[t]

        hidden = F.elu(self.fc_embed_state_action(torch.cat((_state, actions[t]), dim=1)))
        belief = self.rnn(hidden, prev_belief)

        hidden = F.elu(self.fc_embed_belief_posterior(torch.cat([belief, enc_observations[t]], dim=1)))

        posterior_mean, _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
        posterior_std_dev = F.softplus(_posterior_std_dev) + self.min_std_dev
        posterior_state = posterior_mean + posterior_std_dev #* torch.randn_like(posterior_mean)

        prev_belief = belief
        prev_state = posterior_state
        return posterior_state


action_size = 10
transition_model = TransitionModel(
    args.belief_size,
    args.state_size,
    action_size,
    args.hidden_size,
    args.embedding_size,
).to(args.device)


# import nvidia_dlprof_pytorch_nvtx
# nvidia_dlprof_pytorch_nvtx.init()

# Training (and testing)
# with torch.autograd.profiler.emit_nvtx():

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler(results_dir),
    # record_shapes=True,
    # profile_memory=True,
    # with_stack=True,
    # with_flops=True,
    # with_modules=True,
) as p:

    encoded_observations = torch.randn((args.chunk_size, args.batch_size, args.embedding_size)).to(args.device)
    actions = torch.randn((args.chunk_size, args.batch_size, action_size)).to(args.device)
    rewards = torch.randn((args.chunk_size, args.batch_size, 1)).to(args.device)
    nonterminals = torch.ones((args.chunk_size, args.batch_size, 1)).to(bool).to(args.device)

    init_belief = torch.zeros(args.batch_size, args.belief_size, device=args.device)
    init_state = torch.zeros(args.batch_size, args.state_size, device=args.device)

    transition_model = torch.jit.script(transition_model)
    # transition_model = torch.jit.trace_module(transition_model, {
    #     "forward": (init_state, actions[:-1], init_belief, encoded_observations, nonterminals[:-1])})
    # # print(transition_model.graph)
    # print(transition_model.code)
    # print(transition_model.graph_for(init_state, actions[:-1], init_belief, encoded_observations, nonterminals[:-1]))

    size = 5000
    a = torch.randn(size, size, device=args.device)
    b = torch.randn(size, size, device=args.device)

    for s in tqdm(range(500)):

        if s % 10 == 0:
            torch.matmul(a, b)

        out = transition_model(init_state, actions[:-1], init_belief, encoded_observations, nonterminals[:-1])

        # p.step()
