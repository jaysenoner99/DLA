import torch
from torch.distributions import Categorical
import numpy as np

MAX_LEN = 1000


# Given an environment, observation, and policy, sample from pi(a | obs). Returns the
# selected action and the log probability of that action (needed for policy gradient).
def select_action(env, obs, policy, temperature=1.0, deterministic=False):
    probs = policy(obs, temperature=temperature)
    if deterministic:
        action = torch.argmax(probs)
        log_prob = torch.log(probs[action])
    else:
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
    return (action.item(), log_prob.reshape(1))


# Utility to compute the discounted total reward. Torch doesn't like flipped arrays, so we need to
# .copy() the final numpy array. There's probably a better way to do this.
# def compute_returns(rewards, gamma):
#     return np.flip(
#         np.cumsum([gamma ** (i + 1) * r for (i, r) in enumerate(rewards)][::-1]), 0
#     ).copy()
def compute_returns(rewards, gamma):
    """
    Computes discounted returns for a list of rewards.
    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
    """
    returns = np.zeros(len(rewards), dtype=np.float32)
    G = 0.0
    # Iterate backwards through the rewards
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


# Given an environment and a policy, run it up to the maximum number of steps.
def run_episode(env, policy, maxlen=MAX_LEN, temperature=1.0, deterministic=False):
    # Collect just about everything.
    observations = []
    actions = []
    log_probs = []
    rewards = []

    # Reset the environment and start the episode.
    (obs, info) = env.reset()
    for i in range(maxlen):
        # Get the current observation, run the policy and select an action.
        obs = torch.tensor(obs)
        (action, log_prob) = select_action(env, obs, policy, temperature, deterministic)
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)

        # Advance the episode by executing the selected action.
        (obs, reward, term, trunc, info) = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break
    return (observations, actions, torch.cat(log_probs), rewards)


def evaluate_policy(
    env, policy, episodes=5, maxlen=MAX_LEN, temperature=1.0, deterministic=False
):
    policy.eval()
    total_rewards = []
    lengths = []

    for _ in range(episodes):
        obs, _ = env.reset()
        rewards = 0
        for t in range(maxlen):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, _ = select_action(
                    env,
                    obs_tensor,
                    policy,
                    temperature=temperature,
                    deterministic=deterministic,
                )
            obs, reward, term, trunc, _ = env.step(action)
            rewards += reward
            if term or trunc:
                break
        total_rewards.append(rewards)
        lengths.append(t + 1)

    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_length = sum(lengths) / len(lengths)
    policy.train()
    return avg_reward, avg_length
