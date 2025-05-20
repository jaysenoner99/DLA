import argparse
import torch
import wandb
import gymnasium
from networks import PolicyNet, ValueNet
from reinforce import reinforce
from utils import run_episode


def parse_args():
    """The argument parser for the main training script."""
    parser = argparse.ArgumentParser(
        description="A script implementing REINFORCE on the Cartpole and LunarLander environments."
    )
    parser.add_argument(
        "--project",
        type=str,
        default="DLA2025-LunarLander",
        help="Wandb project to log to.",
    )
    parser.add_argument(
        "--baseline", type=str, default="none", help="Baseline to use (none, std)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor for future rewards"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize final agent"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Number of hidden layers in the policy and value networks",
    )
    parser.add_argument("--seed", type=int, default=69, help="Seed")
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Width of the layers in the policy and value networks",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50,
        help="Evaluate the policy every --eval-interval iterations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Evaluate the policy for --eval-episodes episodes",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="If true, normalize G_t - b_t to zero mean and unit variance",
    )
    parser.add_argument(
        "--clip-grad",
        action="store_true",
        help="If true, clip gradients to unit norm for both the policy and the value networks",
    )
    parser.add_argument(
        "--det",
        action="store_true",
        help="Enable deterministic policy evaluation every --eval-interval iterations",
    )
    parser.add_argument(
        "--T",
        type=float,
        default=1.0,
        help="Softmax temperature for the policy. If a temperature scheduler is used, this will be the starting temperature",
    )
    parser.add_argument(
        "--t-schedule",
        choices=["lin", "exp"],
        help="Choose between a linear or exponential temperature scheduler",
    )
    parser.add_argument(
        "--env",
        default="cartpole",
        choices=["cartpole", "lunarlander"],
        help="Choose between the Cartpole and the LunarLander environment",
    )
    parser.set_defaults(visualize=False)
    args = parser.parse_args()
    return args


# Main entry point.
if __name__ == "__main__":
    # Get command line arguments.
    args = parse_args()
    torch.manual_seed(args.seed)
    # Initialize wandb with our configuration parameters.
    run = wandb.init(
        project=args.project,
        config={
            "learning_rate": args.lr,
            "baseline": args.baseline,
            "gamma": args.gamma,
            "num_episodes": args.episodes,
            "depth": args.depth,
            "width": args.width,
            "seed": args.seed,
            "eval_interval": args.eval_interval,
            "eval_episodes": args.eval_episodes,
            "normalize": args.normalize,
            "clip_grad": args.clip_grad,
            "deterministic_eval": args.det,
            "temperature": args.T,
            "t_schedule": args.t_schedule,
        },
    )

    # Instantiate the Cartpole environment (no visualization).
    if args.env == "cartpole":
        env = gymnasium.make("CartPole-v1")
    elif args.env == "lunarlander":
        env = gymnasium.make("LunarLander-v3")

    # Make a policy network.
    policy = PolicyNet(env, n_hidden=args.depth, width=args.width)
    if args.baseline == "learned":
        value = ValueNet(env, n_hidden=args.depth, width=args.width)
    else:
        value = None
    # Train the agent.
    reinforce(
        policy,
        env,
        run,
        value,
        lr=args.lr,
        baseline=args.baseline,
        num_episodes=args.episodes,
        gamma=args.gamma,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        norm_advantages=args.normalize,
        clip_gradients=args.clip_grad,
        deterministic=args.det,
        temperature=args.T,
        t_schedule=args.t_schedule,
    )

    # And optionally run the final agent for a few episodes.
    if args.visualize:
        if args.env == "cartpole":
            env_render = gymnasium.make("CartPole-v1", render_mode="human")
        elif args.env == "lunarlander":
            env_render = gymnasium.make("LunarLander-v3", render_mode="human")
        for _ in range(10):
            run_episode(env_render, policy)

        # Close the visualization environment.
        env_render.close()

    # Close the Cartpole environment and finish the wandb run.
    env.close()
    run.finish()
