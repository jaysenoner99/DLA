import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt
from dataloaders import CIFAR10DataLoader
from model import CNN, Autoencoder
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
import argparse
import torch


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def denorm(batch, args, mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(args.device)
    if isinstance(std, list):
        std = torch.tensor(std).to(args.device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def test_autoencoder(model, args, test_loader, epsilon):
    total_loss = 0
    adv_examples = []

    for data, _ in tqdm(test_loader):
        data = data.to(args.device)
        data.requires_grad = True

        # Forward pass
        _, x_rec = model(data)
        loss = F.mse_loss(x_rec, data)
        model.zero_grad()
        loss.backward()

        # Gradient on input
        data_grad = data.grad.data

        # Denormalize for attack
        data_denorm = denorm(data, args)

        # Attack
        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)

        # Re-normalize for model input
        perturbed_data_normalized = transforms.Normalize(
            [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        )(perturbed_data)

        # Forward pass with perturbed data
        _, rec_adv = model(perturbed_data_normalized)

        # Calculate reconstruction loss
        loss_adv = F.mse_loss(rec_adv, perturbed_data_normalized)
        total_loss += loss_adv.item()

        # Save some examples
        if len(adv_examples) < 5:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((loss.item(), loss_adv.item(), adv_ex))

    avg_loss = total_loss / len(test_loader)
    print(f"Epsilon: {epsilon}\tAvg Reconstruction Loss = {avg_loss}")

    return avg_loss, adv_examples


def make_plots_autoencoder(epsilons, losses, examples):
    # Plot 1: MSE vs Epsilon
    fig_loss = plt.figure(figsize=(5, 5))
    plt.plot(epsilons, losses, "*-")
    # plt.xticks(np.arange(0, 0.35, step=0.05))
    plt.title("Adversarial MSE Loss vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("MSE Loss")
    if args.use_wandb:
        wandb.log({"MSE Loss vs Epsilon": wandb.Image(fig_loss)})
    plt.close(fig_loss)

    # Plot 2: Examples
    fig_adv = plt.figure(figsize=(10, 12))
    cnt = 0
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"Eps: {epsilons[i]}", fontsize=12)

            orig_loss, adv_loss, adv_ex = examples[i][j]
            plt.title(f"MSE: {orig_loss:.3f} → {adv_loss:.3f}")
            plt.imshow(np.transpose(adv_ex, (1, 2, 0)))
    plt.tight_layout()
    if args.use_wandb:
        wandb.log({"Autoencoder Adversarial Examples": wandb.Image(fig_adv)})
    plt.close(fig_adv)


def test(model, args, test_loader, epsilon):
    correct = 0
    adv_examples = []
    criterion = nn.CrossEntropyLoss()
    for data, target in tqdm(test_loader):
        data, target = data.to(args.device), target.to(args.device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue

        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        data_nonorm = denorm(data, args)
        perturbed_data = fgsm_attack(data_nonorm, epsilon, data_grad)
        perturbed_data_norm = transforms.Normalize(
            (0.4914, 0.4822, 0.4465),  # Mean for each channel
            (0.2470, 0.2435, 0.2616),  # Std for each channel
        )(perturbed_data)

        output = model(perturbed_data_norm)
        final_pred = output.max(1, keepdim=True)[
            1
        ]  # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print(
        f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}"
    )

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform a targeted/untargeted fgsm attack"
    )
    device = "cuda" if torch.cuda.is_available else "cpu"
    parser.add_argument(
        "--device",
        default=device,
        type=str,
        metavar="D",
        help="The device where the training process will be executed. Autoset to cuda if cuda is available on the local machine",
    )

    parser.add_argument(
        "--batch-size", default=1, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--val-split",
        default=0.1,
        type=float,
        metavar="v",
        help="Proportion of the training data to be reserved to the validation split.",
    )
    parser.add_argument(
        "--seed",
        default=69,
        type=int,
        metavar="seed",
        help="Seed of the experiments, to be set for reproducibility",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable wandb logging. If not provided, wandb will be disabled.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["ae", "cnn"],
        default="cnn",
        help="Choose between ae or cnn",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.025,
        help="Value of the step to build the epsilon vector",
    )
    parser.add_argument(
        "--max",
        type=float,
        default=0.2,
        help="Maximum value of the epsilon perturbation coefficient",
    )
    args = parser.parse_args()
    return args


def make_plots(epsilons, accuracies, examples):
    # Plot 1: Accuracy vs Epsilon
    fig_acc = plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, "*-")
    # plt.yticks(np.arange(0, 1.1, step=0.1))
    # plt.xticks(np.arange(0, 0.35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    if args.use_wandb:
        wandb.log({"Accuracy vs Epsilon": wandb.Image(fig_acc)})
    plt.close(fig_acc)

    # Plot 2: Adversarial Examples
    fig_adv = plt.figure(figsize=(8, 10))
    cnt = 0
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title(f"{orig} -> {adv}")
            plt.imshow(np.transpose(ex, (1, 2, 0)))  # Convert from C×H×W to H×W×C
    plt.tight_layout()
    if args.use_wandb:
        wandb.log({"Adversarial Examples": wandb.Image(fig_adv)})
    plt.close(fig_adv)


args = parse_args()
torch.manual_seed(args.seed)


run_name = f"{args.model}_fgsm"
if args.use_wandb:
    wandb.init(project="lab_2_DLA", name=run_name, config=args)
if args.model == "cnn":
    model = CNN()
    model.load_state_dict(torch.load("cifar10_CNN.pth"))
elif args.model == "ae":
    model = Autoencoder()
    model.load_state_dict(torch.load("cifar10_ae.pth"))


model.to(args.device)
accuracies = []
examples = []
max = args.max
step = args.step
epsilons = np.arange(0, max, step).tolist()
loader = CIFAR10DataLoader(args)
_, _, test_loader = loader.get_dataloaders()

if args.model == "cnn":
    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(model, args, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)
    make_plots(epsilons, accuracies, examples)

elif args.model == "ae":
    for eps in epsilons:
        loss, ex = test_autoencoder(model, args, test_loader, eps)
        accuracies.append(loss)
        examples.append(ex)
    make_plots_autoencoder(epsilons, accuracies, examples)
