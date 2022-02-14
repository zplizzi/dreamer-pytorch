import torch
from torch.distributions import Normal
from env import EnvBatcher


def update_belief_and_act(
        args, env, actor_model, transition_model, encoder, belief, posterior_state, action, observation, explore=False
):
    # Infer belief over current state q(s_t|o≤t,a<t) from the history
    # print("action size: ",action.size()) torch.Size([1, 6])
    # Action and observation need extra time dimension
    # TODO: surely we're not computing this from scratch each time, right??
    # no, looks like the number of steps computed is based on the action size, which here should be just 1 (?).
    # TODO: how do we compute the posterior when we don't have the next observation?
    # make better sense of the time indexes on the posterior model. does z_t represent the state before or after o_t?
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
