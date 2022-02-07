import argparse
import os
from test import test

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
from env import EnvBatcher
from memory import ExperienceReplay
from models import ActorModel
from models import Encoder
from models import ObservationModel
from models import RewardModel
from models import TransitionModel
from models import ValueModel
from models import bottle
from tensorboardX import SummaryWriter
from utils import FreezeParameters
from utils import imagine_ahead
from utils import lambda_return
from utils import lineplot

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
metrics = {
    "steps": [],
    "episodes": [],
    "train_rewards": [],
    "test_episodes": [],
    "test_rewards": [],
    "observation_loss": [],
    "reward_loss": [],
    "kl_loss": [],
    "actor_loss": [],
    "value_loss": [],
}

summary_name = results_dir + "/{}_{}_log"
writer = SummaryWriter(summary_name.format(args.env, args.id))

# Initialise training environment and experience replay memory
env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)

# initialize experience dataset
D = ExperienceReplay(
    args.experience_size, args.symbolic_env, env.observation_size, env.action_size, args.bit_depth, args.device
)
# Initialise dataset D with S random seed episodes
# TODO: clean this up
for s in range(1, args.seed_episodes + 1):
    observation, done, t = env.reset(), False, 0
    while not done:
        action = env.sample_random_action()
        next_observation, reward, done = env.step(action)
        D.append(observation, action, reward, done)
        observation = next_observation
        t += 1
    metrics["steps"].append(t * args.action_repeat + (0 if len(metrics["steps"]) == 0 else metrics["steps"][-1]))
    metrics["episodes"].append(s)


# Initialise models
transition_model = TransitionModel(
    args.belief_size,
    args.state_size,
    env.action_size,
    args.hidden_size,
    args.embedding_size,
    args.dense_activation_function,
)
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
models = [model.to(device=args.device) for model in models]

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

# load models from checkpoint
if args.models != "" and os.path.exists(args.models):
    model_dicts = torch.load(args.models)
    transition_model.load_state_dict(model_dicts["transition_model"])
    observation_model.load_state_dict(model_dicts["observation_model"])
    reward_model.load_state_dict(model_dicts["reward_model"])
    encoder.load_state_dict(model_dicts["encoder"])
    actor_model.load_state_dict(model_dicts["actor_model"])
    value_model.load_state_dict(model_dicts["value_model"])
    model_optimizer.load_state_dict(model_dicts["model_optimizer"])

global_prior = Normal(
    torch.zeros(args.batch_size, args.state_size, device=args.device),
    torch.ones(args.batch_size, args.state_size, device=args.device),
)
# Allowed deviation in KL divergence
free_nats = torch.full((1,), args.free_nats, device=args.device)


def update_belief_and_act(
    args, env, actor_model, transition_model, encoder, belief, posterior_state, action, observation, explore=False
):
    # Infer belief over current state q(s_t|o≤t,a<t) from the history
    # print("action size: ",action.size()) torch.Size([1, 6])
    # Action and observation need extra time dimension
    belief, _, _, _, posterior_state, _, _ = transition_model(
        posterior_state, action.unsqueeze(dim=0), belief, encoder(observation).unsqueeze(dim=0)
    )
    # Remove time dimension from belief/state
    belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)
    action = actor_model.get_action(belief, posterior_state, det=not (explore))
    if explore:
        # Add gaussian exploration noise on top of the sampled action
        action = torch.clamp(Normal(action, args.action_noise).rsample(), -1, 1)
        # action = action + args.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
    # Perform environment step (action repeats handled internally)
    next_observation, reward, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu())
    return belief, posterior_state, action, next_observation, reward, done


# Training (and testing)
for episode in tqdm(
    range(metrics["episodes"][-1] + 1, args.episodes + 1), total=args.episodes, initial=metrics["episodes"][-1] + 1
):
    # Model fitting
    losses = []
    model_modules = transition_model.modules + encoder.modules + observation_model.modules + reward_model.modules

    print("training loop")
    for s in tqdm(range(args.collect_interval)):
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        # Transitions start at time t = 0
        observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size)
        # Create initial belief and state for time t = 0
        init_belief, init_state = torch.zeros(args.batch_size, args.belief_size, device=args.device), torch.zeros(
            args.batch_size, args.state_size, device=args.device
        )
        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        (
            beliefs,
            prior_states,
            prior_means,
            prior_std_devs,
            posterior_states,
            posterior_means,
            posterior_std_devs,
        ) = transition_model(
            init_state, actions[:-1], init_belief, bottle(encoder, (observations[1:],)), nonterminals[:-1]
        )
        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting);
        # sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
        if args.worldmodel_LogProbLoss:
            observation_dist = Normal(bottle(observation_model, (beliefs, posterior_states)), 1)
            observation_loss = (
                -observation_dist.log_prob(observations[1:])
                .sum(dim=2 if args.symbolic_env else (2, 3, 4))
                .mean(dim=(0, 1))
            )
        else:
            observation_loss = (
                F.mse_loss(bottle(observation_model, (beliefs, posterior_states)), observations[1:], reduction="none")
                .sum(dim=2 if args.symbolic_env else (2, 3, 4))
                .mean(dim=(0, 1))
            )
        if args.worldmodel_LogProbLoss:
            reward_dist = Normal(bottle(reward_model, (beliefs, posterior_states)), 1)
            reward_loss = -reward_dist.log_prob(rewards[:-1]).mean(dim=(0, 1))
        else:
            reward_loss = F.mse_loss(
                bottle(reward_model, (beliefs, posterior_states)), rewards[:-1], reduction="none"
            ).mean(dim=(0, 1))
        # transition loss
        div = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(
            dim=2
        )
        # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
        kl_loss = torch.max(div, free_nats).mean(dim=(0, 1))
        # Apply linearly ramping learning rate schedule
        if args.learning_rate_schedule != 0:
            for group in model_optimizer.param_groups:
                group["lr"] = min(
                    group["lr"] + args.model_learning_rate / args.model_learning_rate_schedule,
                    args.model_learning_rate,
                )
        model_loss = observation_loss + reward_loss + kl_loss
        # Update model parameters
        model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
        model_optimizer.step()

        # Dreamer implementation: actor loss calculation and optimization
        with torch.no_grad():
            actor_states = posterior_states.detach()
            actor_beliefs = beliefs.detach()
        with FreezeParameters(model_modules):
            imagination_traj = imagine_ahead(
                actor_states, actor_beliefs, actor_model, transition_model, args.planning_horizon
            )
        imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = imagination_traj
        with FreezeParameters(model_modules + value_model.modules):
            imged_reward = bottle(reward_model, (imged_beliefs, imged_prior_states))
            value_pred = bottle(value_model, (imged_beliefs, imged_prior_states))
        returns = lambda_return(
            imged_reward, value_pred, bootstrap=value_pred[-1], discount=args.discount, lambda_=args.disclam
        )
        actor_loss = -torch.mean(returns)
        # Update model parameters
        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor_model.parameters(), args.grad_clip_norm, norm_type=2)
        actor_optimizer.step()

        # Dreamer implementation: value loss calculation and optimization
        with torch.no_grad():
            value_beliefs = imged_beliefs.detach()
            value_prior_states = imged_prior_states.detach()
            target_return = returns.detach()
        # detach the input tensor from the transition network.
        value_dist = Normal(bottle(value_model, (value_beliefs, value_prior_states)), 1)
        value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1))
        # Update model parameters
        value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip_norm, norm_type=2)
        value_optimizer.step()

        # Store (0) observation loss (1) reward loss (2) KL loss (3) actor loss (4) value loss
        losses.append(
            [observation_loss.item(), reward_loss.item(), kl_loss.item(), actor_loss.item(), value_loss.item()]
        )

    # Update and plot loss metrics
    losses = tuple(zip(*losses))
    metrics["observation_loss"].append(losses[0])
    metrics["reward_loss"].append(losses[1])
    metrics["kl_loss"].append(losses[2])
    metrics["actor_loss"].append(losses[3])
    metrics["value_loss"].append(losses[4])
    # lineplot(
    #     metrics["episodes"][-len(metrics["observation_loss"]) :],
    #     metrics["observation_loss"],
    #     "observation_loss",
    #     results_dir,
    # )
    # lineplot(metrics["episodes"][-len(metrics["reward_loss"]) :], metrics["reward_loss"], "reward_loss", results_dir)
    # lineplot(metrics["episodes"][-len(metrics["kl_loss"]) :], metrics["kl_loss"], "kl_loss", results_dir)
    # lineplot(metrics["episodes"][-len(metrics["actor_loss"]) :], metrics["actor_loss"], "actor_loss", results_dir)
    # lineplot(metrics["episodes"][-len(metrics["value_loss"]) :], metrics["value_loss"], "value_loss", results_dir)

    # Data collection
    print("Data collection")
    with torch.no_grad():
        observation, total_reward = env.reset(), 0
        belief, posterior_state, action = (
            torch.zeros(1, args.belief_size, device=args.device),
            torch.zeros(1, args.state_size, device=args.device),
            torch.zeros(1, env.action_size, device=args.device),
        )
        pbar = tqdm(range(args.max_episode_length // args.action_repeat))
        for t in pbar:
            # print("step",t)
            belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(
                args,
                env,
                actor_model,
                transition_model,
                encoder,
                belief,
                posterior_state,
                action,
                observation.to(device=args.device),
                explore=True,
            )
            D.append(observation, action.cpu(), reward, done)
            total_reward += reward
            observation = next_observation
            if args.render:
                env.render()
            if done:
                pbar.close()
                break

        # Update and plot train reward metrics
        metrics["steps"].append(t + metrics["steps"][-1])
        metrics["episodes"].append(episode)
        metrics["train_rewards"].append(total_reward)
        lineplot(
            metrics["episodes"][-len(metrics["train_rewards"]) :],
            metrics["train_rewards"],
            "train_rewards",
            results_dir,
        )

    # Test model
    print("Test model")
    if episode % args.test_interval == 0:
        test(args, models, metrics)

    writer.add_scalar("train_reward", metrics["train_rewards"][-1], metrics["steps"][-1])
    writer.add_scalar("train/episode_reward", metrics["train_rewards"][-1], metrics["steps"][-1] * args.action_repeat)
    writer.add_scalar("observation_loss", metrics["observation_loss"][0][-1], metrics["steps"][-1])
    writer.add_scalar("reward_loss", metrics["reward_loss"][0][-1], metrics["steps"][-1])
    writer.add_scalar("kl_loss", metrics["kl_loss"][0][-1], metrics["steps"][-1])
    writer.add_scalar("actor_loss", metrics["actor_loss"][0][-1], metrics["steps"][-1])
    writer.add_scalar("value_loss", metrics["value_loss"][0][-1], metrics["steps"][-1])
    print(
        "episodes: {}, total_steps: {}, train_reward: {} ".format(
            metrics["episodes"][-1], metrics["steps"][-1], metrics["train_rewards"][-1]
        )
    )

    # Checkpoint models
    if episode % args.checkpoint_interval == 0:
        torch.save(
            {
                "transition_model": transition_model.state_dict(),
                "observation_model": observation_model.state_dict(),
                "reward_model": reward_model.state_dict(),
                "encoder": encoder.state_dict(),
                "actor_model": actor_model.state_dict(),
                "value_model": value_model.state_dict(),
                "model_optimizer": model_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "value_optimizer": value_optimizer.state_dict(),
            },
            os.path.join(results_dir, "models_%d.pth" % episode),
        )


# Close training environment
env.close()
