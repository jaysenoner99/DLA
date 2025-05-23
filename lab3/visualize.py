import torch
import gymnasium as gym
import os
import argparse
from networks import PolicyNet  # Make sure your policy class is accessible
import imageio
from utils import run_episode


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model


def render_and_record_episode(env, policy, temperature=1.0, deterministic=False):
    obs, _ = env.reset()
    done = False
    frames = []

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_probs = policy(obs_tensor)
            if deterministic:
                action = torch.argmax(action_probs, dim=-1).item()
            else:
                action = (
                    torch.distributions.Categorical(action_probs / temperature)
                    .sample()
                    .item()
                )
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        frames.append(frame)
        done = terminated or truncated

    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="wandb/latest-run/files/checkpoint-BEST_EVAL_POLICY.pt",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Whether to apply a deterministic sampler for the policy",
    )
    parser.add_argument(
        "--env",
        choices=["cartpole", "lunarlander"],
        help="Choose between the Cartpole and the LunarLander environment",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Witdh of the layers of the pretrained policy network",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="If set, record a gif of the first episode.",
    )
    parser.add_argument(
        "--gif-path",
        type=str,
        default="lander.gif",
        help="Path to save the gif if --gif is set.",
    )
    parser.add_argument(
        "--depth", type=int, default=2, help="Depth of the policy network"
    )
    args = parser.parse_args()

    # Define your policy architecture (must match what was trained)

    render_mode = "rgb_array" if args.gif else "human"
    if args.env == "cartpole":
        env = gym.make("CartPole-v1", render_mode=render_mode)
    elif args.env == "lunarlander":
        env = gym.make("LunarLander-v3", render_mode=render_mode)
    policy = PolicyNet(env, width=args.width, n_hidden=args.depth)
    policy.eval()

    # Load checkpoint
    policy = load_checkpoint(policy, args.checkpoint)

    # Create rendering environment
    if args.gif:
        print(f"Recording GIF to {args.gif_path}...")
        frames = render_and_record_episode(
            env, policy, temperature=args.temperature, deterministic=args.deterministic
        )
        imageio.mimsave(args.gif_path, frames, fps=30)
        print(f"GIF saved at {args.gif_path}")
    else:
        for i in range(args.episodes):
            run_episode(
                env,
                policy,
                temperature=args.temperature,
                deterministic=args.deterministic,
            )

    env.close()


if __name__ == "__main__":
    main()
