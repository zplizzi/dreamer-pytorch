import numpy as np
import torch
from torchvision.utils import make_grid
from tqdm import tqdm

from env import Env
from env import EnvBatcher
from stuff import update_belief_and_act


def test(args, models):
    models = [model.eval() for model in models]
    transition_model, observation_model, reward_model, encoder, actor_model, value_model = models

    # Initialise parallelised test environments
    test_envs = EnvBatcher(
        Env,
        (args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat),
        {},
        args.test_episodes,
    )

    with torch.no_grad():
        observation, total_rewards, video_frames = test_envs.reset(), np.zeros((args.test_episodes,)), []
        belief, posterior_state, action = (
            torch.zeros(args.test_episodes, args.belief_size, device=args.device),
            torch.zeros(args.test_episodes, args.state_size, device=args.device),
            torch.zeros(args.test_episodes, test_envs.envs[0].action_size, device=args.device),
        )
        pbar = tqdm(range(args.max_episode_length // args.action_repeat))
        for t in pbar:
            belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(
                args,
                test_envs,
                actor_model,
                transition_model,
                encoder,
                belief,
                posterior_state,
                action,
                observation.to(device=args.device),
            )
            total_rewards += reward.numpy()
            if not args.symbolic_env:  # Collect real vs. predicted frames for video
                # Decentre
                video_frames.append(
                    make_grid(
                        torch.cat([observation, observation_model(belief, posterior_state).cpu()], dim=3) + 0.5,
                        nrow=5,
                    ).numpy()
                )
            observation = next_observation
            if done.sum().item() == args.test_episodes:
                pbar.close()
                break

    # Update and plot reward metrics (and write video if applicable) and save metrics
    # metrics["test_episodes"].append(episode)
    # metrics["test_rewards"].append(total_rewards.tolist())
    # lineplot(metrics["test_episodes"], metrics["test_rewards"], "test_rewards", results_dir)
    # lineplot(
    #     np.asarray(metrics["steps"])[np.asarray(metrics["test_episodes"]) - 1],
    #     metrics["test_rewards"],
    #     "test_rewards_steps",
    #     results_dir,
    #     xaxis="step",
    # )
    # if not args.symbolic_env:
    #     episode_str = str(episode).zfill(len(str(args.episodes)))
    #     write_video(video_frames, "test_episode_%s" % episode_str, results_dir)  # Lossy compression
    #     save_image(torch.as_tensor(video_frames[-1]), os.path.join(results_dir, "test_episode_%s.png" % episode_str))
    # torch.save(metrics, os.path.join(results_dir, "metrics.pth"))

    # Set models to train mode
    [model.train() for model in models]
    # Close test environments
    test_envs.close()

    # total_rewards should be a list of shape (batch_size, )?
    return total_rewards
