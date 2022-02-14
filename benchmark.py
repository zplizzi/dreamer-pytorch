import argparse
import os

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from tqdm import tqdm

from env import CONTROL_SUITE_ENVS
from env import GYM_ENVS
from env import Env
from memory import ExperienceReplay
from models import ActorModel
from models import Encoder
from models import ObservationModel
from models import RewardModel
from models import TransitionModel
from models import ValueModel
from models import bottle
from utils import FreezeParameters
from utils import imagine_ahead
from utils import lambda_return



# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, default="default", help="Experiment ID")
parser.add_argument("--seed", type=int, default=1, metavar="S", help="Random seed")
parser.add_argument(
    "--env",
    type=str,
    default="Pendulum-v0",
    choices=GYM_ENVS + CONTROL_SUITE_ENVS,
    help="Gym/Control Suite environment",
)
parser.add_argument("--symbolic-env", action="store_true", help="Symbolic features")
parser.add_argument("--max-episode-length", type=int, default=1000, metavar="T", help="Max episode length")
# Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument("--experience-size", type=int, default=1000000, metavar="D", help="Experience replay size")
parser.add_argument("--cnn-activation-function", type=str, default="relu", choices=dir(F))
parser.add_argument("--dense-activation-function", type=str, default="elu", choices=dir(F))
# Note that the default encoder for visual observations outputs a 1024D vector;
# for other embedding sizes an additional fully-connected layer is used
parser.add_argument("--embedding-size", type=int, default=1024, metavar="E", help="Observation embedding size")
parser.add_argument("--hidden-size", type=int, default=200, metavar="H", help="Hidden size")
parser.add_argument("--belief-size", type=int, default=200, metavar="H", help="Belief/hidden size")
parser.add_argument("--state-size", type=int, default=30, metavar="Z", help="State/latent size")
parser.add_argument("--action-repeat", type=int, default=2, metavar="R", help="Action repeat")
parser.add_argument("--action-noise", type=float, default=0.3, metavar="ε", help="Action noise")
parser.add_argument("--episodes", type=int, default=1000, metavar="E", help="Total number of episodes")
parser.add_argument("--seed-episodes", type=int, default=5, metavar="S", help="Seed episodes")
parser.add_argument("--collect-interval", type=int, default=100, metavar="C", help="Collect interval")
parser.add_argument("--batch-size", type=int, default=50, metavar="B", help="Batch size")
parser.add_argument("--chunk-size", type=int, default=50, metavar="L", help="Chunk size")
# TODO: shouldn't this be enabled by default?
parser.add_argument(
    "--worldmodel-LogProbLoss",
    action="store_true",
    help="use LogProb loss for observation_model and reward_model training",
)
parser.add_argument("--free-nats", type=float, default=3, metavar="F", help="Free nats")
parser.add_argument("--bit-depth", type=int, default=5, metavar="B", help="Image bit depth (quantisation)")
parser.add_argument("--model_learning-rate", type=float, default=1e-3, metavar="α", help="Learning rate")
parser.add_argument("--actor_learning-rate", type=float, default=8e-5, metavar="α", help="Learning rate")
parser.add_argument("--value_learning-rate", type=float, default=8e-5, metavar="α", help="Learning rate")
parser.add_argument("--adam-epsilon", type=float, default=1e-7, metavar="ε", help="Adam optimizer epsilon value")
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument("--grad-clip-norm", type=float, default=100.0, metavar="C", help="Gradient clipping norm")
parser.add_argument("--planning-horizon", type=int, default=15, metavar="H", help="Planning horizon distance")
parser.add_argument("--discount", type=float, default=0.99, metavar="H", help="Planning horizon distance")
parser.add_argument("--disclam", type=float, default=0.95, metavar="H", help="discount rate to compute return")
parser.add_argument("--optimisation-iters", type=int, default=10, metavar="I", help="Planning optimisation iterations")
parser.add_argument("--candidates", type=int, default=1000, metavar="J", help="Candidate samples per iteration")
parser.add_argument("--top-candidates", type=int, default=100, metavar="K", help="Number of top candidates to fit")
parser.add_argument("--test-interval", type=int, default=25, metavar="I", help="Test interval (episodes)")
parser.add_argument("--test-episodes", type=int, default=10, metavar="E", help="Number of test episodes")
parser.add_argument("--checkpoint-interval", type=int, default=50, metavar="I", help="Checkpoint interval (episodes)")
parser.add_argument("--models", type=str, default="", metavar="M", help="Load model checkpoint")
parser.add_argument("--render", action="store_true", help="Render environment")

args = parser.parse_args()

# Setup
results_dir = os.path.join("results", "{}_{}".format(args.env, args.id))
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.device = torch.device("cuda")
torch.cuda.manual_seed(args.seed)

# Initialise training environment and experience replay memory
env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat)

environment_steps = 0
training_steps = 0

# initialize experience dataset
D = ExperienceReplay(
    args.experience_size, args.symbolic_env, env.observation_size, env.action_size, args.device
)
# Initialise dataset D with S random seed episodes
print("building initial dataset")
observation, done, t = env.reset(), False, 0
for i in range(200):
    action = env.sample_random_action()
    next_observation, reward, done = env.step(action)
    D.append(observation, action, reward, done)
    observation = next_observation
    t += 1
    environment_steps += 1
print("done")


# Initialise models
print("initializing models")
transition_model = TransitionModel(
    args.belief_size,
    args.state_size,
    env.action_size,
    args.hidden_size,
    args.embedding_size,
    args.dense_activation_function,
)
# transition_model = torch.jit.script(transition_model)
observation_model = ObservationModel(
    args.symbolic_env,
    env.observation_size,
    args.belief_size,
    args.state_size,
    args.embedding_size,
    args.cnn_activation_function,
)
reward_model = RewardModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function)
encoder = Encoder(args.symbolic_env, env.observation_size, args.embedding_size, args.cnn_activation_function)
actor_model = ActorModel(
    args.belief_size, args.state_size, args.hidden_size, env.action_size, args.dense_activation_function
)
value_model = ValueModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function)
models = [transition_model, observation_model, reward_model, encoder, actor_model, value_model]
# models = [torch.jit.script(model) for model in models]
models = [model.to(device=args.device) for model in models]
# print(transition_model.graph)
# print(transition_model.code)


# Set up optimizers
param_list = (
    list(transition_model.parameters())
    + list(observation_model.parameters())
    + list(reward_model.parameters())
    + list(encoder.parameters())
)
value_actor_param_list = list(value_model.parameters()) + list(actor_model.parameters())
params_list = param_list + value_actor_param_list
model_optimizer = optim.Adam(param_list, lr=args.model_learning_rate, eps=args.adam_epsilon)
actor_optimizer = optim.Adam(actor_model.parameters(), lr=args.actor_learning_rate, eps=args.adam_epsilon)
value_optimizer = optim.Adam(value_model.parameters(), lr=args.value_learning_rate, eps=args.adam_epsilon)


global_prior = Normal(
    torch.zeros(args.batch_size, args.state_size, device=args.device),
    torch.ones(args.batch_size, args.state_size, device=args.device),
)
# Allowed deviation in KL divergence
free_nats = torch.full((1,), args.free_nats, device=args.device)

traced_transition = None



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
    for episode in range(1):
        # Model fitting
        losses = []
        # model_modules = transition_model.modules + encoder.modules + observation_model.modules + reward_model.modules

        print("training loop")
        observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size)
        for s in range(50):
        # for s in tqdm(range(args.collect_interval)):

            # if s > 50 and episode == 0:
            #     torch.cuda.profiler.start()
            #
            # if s > 80 and episode == 0:
            #     torch.cuda.profiler.stop()


            # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
            # Transitions start at time t = 0
            # Create initial belief and state for time t = 0
            init_belief = torch.zeros(args.batch_size, args.belief_size, device=args.device)
            init_state = torch.zeros(args.batch_size, args.state_size, device=args.device)
            # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
            encoded_observations = bottle(encoder, (observations[1:],))
            if traced_transition is None:
                traced_transition = torch.jit.trace_module(transition_model, {"forward": (init_state, actions[:-1], init_belief, encoded_observations, nonterminals[:-1])})
                # print(traced_transition.graph)
                print(traced_transition.code)
                print(traced_transition.graph_for(init_state, actions[:-1], init_belief, encoded_observations, nonterminals[:-1]))

            (
                beliefs,
                # prior_states,
                # prior_means,
                # prior_std_devs,
                posterior_states,
                posterior_means,
                posterior_std_devs,
            # ) = transition_model(init_state, actions[:-1], init_belief, encoded_observations, nonterminals[:-1])
            ) = traced_transition(init_state, actions[:-1], init_belief, encoded_observations, nonterminals[:-1])

        # TODO: what does this comment mean?
            # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting);

            ## RSSM losses + training step

            # # Reconstruction loss - based on posterior
            # obs_prediction = bottle(observation_model, (beliefs, posterior_states))
            # if args.worldmodel_LogProbLoss:
            #     observation_dist = Normal(obs_prediction, 1)
            #     log_prob = observation_dist.log_prob(observations[1:])
            #     observation_loss = -log_prob.sum(dim=2 if args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))
            # else:
            # observation_loss = (
            #     F.mse_loss(obs_prediction, observations[1:], reduction="none")
            #     .sum(dim=2 if args.symbolic_env else (2, 3, 4))
            #     .mean(dim=(0, 1))
            # )
            #
            # Reward loss - based on posterior
            # reward_preds = bottle(reward_model, (beliefs, posterior_states))
            # reward_loss = F.mse_loss(reward_preds, rewards[:-1], reduction="none").mean(dim=(0, 1))
            #
            # # transition loss
            # div = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(
            #     dim=2
            # )
            # kl_loss = torch.max(div, free_nats).mean(dim=(0, 1))
            #
            # model_loss = kl_loss #+ reward_loss + kl_loss
            # # Update model parameters
            # model_optimizer.zero_grad(set_to_none=True)
            # model_loss.backward()
            # nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
            # model_optimizer.step()

            # # Dreamer implementation: actor loss calculation and optimization
            # # TODO: remove this no_grad, lol
            # with torch.no_grad():
            #     actor_states = posterior_states.detach()
            #     actor_beliefs = beliefs.detach()
            # # TODO: replace this with a properly placed no_grad?
            # # Do an imagination rollout
            # with FreezeParameters(model_modules):
            #     # TODO: https://github.com/yusukeurakami/dreamer-pytorch/issues/10
            #     # I agree with the issue - it's weird that this works.....
            #     imagination_traj = imagine_ahead(
            #         actor_states, actor_beliefs, actor_model, transition_model, args.planning_horizon
            #     )
            # imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = imagination_traj
            # with FreezeParameters(model_modules + value_model.modules):
            #     imged_reward = bottle(reward_model, (imged_beliefs, imged_prior_states))
            #     value_pred = bottle(value_model, (imged_beliefs, imged_prior_states))
            # returns = lambda_return(
            #     imged_reward, value_pred, bootstrap=value_pred[-1], discount=args.discount, lambda_=args.disclam
            # )
            # actor_loss = -torch.mean(returns)
            # # Update model parameters
            # actor_optimizer.zero_grad(set_to_none=True)
            # actor_loss.backward()
            # nn.utils.clip_grad_norm_(actor_model.parameters(), args.grad_clip_norm, norm_type=2)
            # actor_optimizer.step()
            #
            # # Dreamer implementation: value loss calculation and optimization
            # with torch.no_grad():
            #     value_beliefs = imged_beliefs.detach()
            #     value_prior_states = imged_prior_states.detach()
            #     target_return = returns.detach()
            # # detach the input tensor from the transition network.
            # # TODO: we're running the value model a second time on the same inputs. can we simplify this?
            # # TODO: shouldn't the value loss use MSE loss? and not predict a distribution?
            # value_dist = Normal(bottle(value_model, (value_beliefs, value_prior_states)), 1)
            # value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1))
            # # Update model parameters
            # value_optimizer.zero_grad(set_to_none=True)
            # value_loss.backward()
            # nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip_norm, norm_type=2)
            # value_optimizer.step()
            #
            # losses.append(
            #     [observation_loss.item(), reward_loss.item(), kl_loss.item(), actor_loss.item(), value_loss.item()]
            # )
            #
            # training_steps += 1

            p.step()

    # Close training environment
    env.close()
