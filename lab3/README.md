# Lab 3 — Deep Reinforcement Learning

This lab implements the REINFORCE algorithm to train an agent in the **CartPole-v1** and **LunarLander-v3** environments using PyTorch. 
The project supports various enhancements including value baselines, entropy regularization, gradient clipping, and temperature scheduling.

## 📁 Folder Structure

```
├── main.py           # Entry point for training
├── reinforce.py      # Core REINFORCE training loop
├── visualize.py      # Visualize a trained agent
├── networks.py       # Policy and value network definitions
├── utils.py          # Utility functions for training and evaluation
└── README.md         # This file
```


## 🧠 REINFORCE Overview

REINFORCE is a Monte Carlo policy gradient method with the update rule:

\[
\nabla_\theta J(\theta) = \mathbb{E}_\pi [\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b_t)]
\]

Where:
- \( \pi_\theta(a_t|s_t) \) is the policy probability
- \( G_t \) is the return (discounted sum of future rewards)
- \( b_t \) is a baseline to reduce variance (e.g., average return or value network)

---

## 🚀  Usage

Use `main.py` to train an agent on a given environment.

### 🔧 Example: Train an agent on LunarLander-v3 

```bash
python main.py --env lunarlander --baseline std --episodes 1000 --depth 2 --width 256 --normalize --clip-grad --det --T 1.0 --t-schedule exp
```

### 🎮 Example: Visualize a pretrained agent trying to do its best

'''bash
python visualize.py --checkpoint path/to/policy_weights
'''



## 📥 Supported Arguments

###  `main.py`

This script is the main training entry point for running the REINFORCE algorithm using PyTorch on CartPole-v1 or LunarLander-v3 environments. 

-Manages experiment configuration and logging

-Sets up environments and models

-Trains a REINFORCE agent using the specified configuration

-Optionally visualizes the trained agent

| Argument              | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `--env`               | Environment to train on: `cartpole` or `lunarlander`                        |
| `--baseline`          | Baseline to use: `none`, `std`, or `learned`                                |
| `--episodes`          | Number of episodes to train for                                             |
| `--depth`             | Number of hidden layers in policy/value networks                            |
| `--width`             | Width of each hidden layer                                                  |
| `--lr`                | Learning rate (default: `1e-3`)                                              |
| `--gamma`             | Discount factor (default: `0.99`)                                           |
| `--normalize`         | Normalize the advantage term                                                |
| `--clip-grad`         | Clip gradients to unit norm                                                 |
| `--det`               | Perform deterministic evaluations                                           |
| `--T`                 | Initial temperature for action sampling                                     |
| `--t-schedule`        | Temperature schedule: `lin`, `exp`, or leave unset for static temperature   |
| `--eval-interval`     | How often to evaluate the policy                                            |
| `--eval-episodes`     | Number of episodes to run during each evaluation                            |
| `--visualize`         | Visualize the final trained policy after training                           |


### `visualize.py`


This script is used to **load and visualize a trained policy network** in either the CartPole or LunarLander environments. 
The policy is loaded from a specified checkpoint and used to run a number of episodes, rendered in a window for visual inspection.


| Argument           | Type     | Default Value                                                   | Description                                                                 |
|--------------------|----------|------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `--checkpoint`     | `str`    | `"wandb/latest-run/files/checkpoint-BEST_EVAL_POLICY.pt"`       | Path to the saved model checkpoint.                                        |
| `--episodes`       | `int`    | `10`                                                             | Number of episodes to run and visualize.                                   |
| `--temperature`    | `float`  | `1.0`                                                            | Softmax temperature for action sampling.                                   |
| `--deterministic`  | `flag`   | `False`                                                          | Use deterministic action selection instead of sampling.                    |
| `--env`            | `str`    | *Required*                                                       | Environment to visualize: `"cartpole"` or `"lunarlander"`.                 |
| `--width`          | `int`    | `256`                                                            | Width of hidden layers in the policy network (must match training).        |
| `--depth`          | `int`    | `2`                                                              | Number of hidden layers in the policy network (must match training).       |

---

## 📊 Evaluation & Logging

Evaluation is done every `--eval-interval` steps with `--eval-episodes` episodes.

Logged metrics include:
- Episode return and length
- Evaluation reward (stochastic and deterministic)
- Losses: policy and value loss (if using learned baseline)
- Temperature schedule

Model checkpoints are saved to:
```
wandb/latest-run/files/checkpoint-BEST_EVAL_POLICY.pt
```
---

## 📊 Results Summary


### 🖼️ Qualitative Results: 
