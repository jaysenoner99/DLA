import torch
import gymnasium as gym
import os
import argparse
from networks import PolicyNet  # Make sure your policy class is accessible
from utils import run_episode


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model


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
    args = parser.parse_args()

    # Define your policy architecture (must match what was trained)

    env = gym.make("LunarLander-v3", render_mode="human")
    policy = PolicyNet(env, width=256, n_hidden=2)
    policy.eval()

    # Load checkpoint
    policy = load_checkpoint(policy, args.checkpoint)

    # Create rendering environment

    for i in range(args.episodes):
        run_episode(
            env, policy, temperature=args.temperature, deterministic=args.deterministic
        )

    env.close()


if __name__ == "__main__":
    main()
