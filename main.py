import argparse
import wandb
from models import MLP, ResidualMLP, CustomCNN
import torch
from train_and_test import train, test, check_gradient_signal
from dataloaders import MNISTDataLoader, CIFAR100DataLoader, CIFAR10DataLoader
import torch.nn as nn
from torchinfo import summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a MLP/ResMLP/CNN on MNIST/CIFAR10/CIFAR100"
    )

    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.001,
        type=float,
        metavar="lr",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--schedule",
        default=[35],
        nargs="*",
        type=int,
        help="learning rate schedule (when to drop lr by 10x)",
    )
    parser.add_argument(
        "--bn",
        action="store_true",
        help="Enable Batch Normalization between the models layers. Only for MLP and ResMLP",
    )
    parser.add_argument(
        "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")

    device = "cuda" if torch.cuda.is_available else "cpu"
    parser.add_argument(
        "--device",
        default=device,
        type=str,
        metavar="D",
        help="The device where the training process will be executed. Autoset to cuda if cuda is available on the local machine",
    )
    parser.add_argument(
        "--dataset",
        default="mnist",
        type=str,
        metavar="dataset",
        help="Training/testing dataset",
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
        "--model",
        type=str,
        choices=["mlp", "resmlp", "cnn"],
        default="mlp",
        help="Choose model architecture: mlp, resmlp, or cnn",
    )
    parser.add_argument(
        "--width",
        default=16,
        type=int,
        metavar="w",
        help="Width of the neural network(hidden dimension)",
    )
    parser.add_argument(
        "--depth",
        default=2,
        type=int,
        metavar="depth",
        help="Depth of the neural network(number of layers)",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="Whether to use skip connnections inside the BasicBlock of a CNN",
    )
    parser.add_argument(
        "--layers",
        default=[2, 2, 2, 2],
        nargs="*",
        type=int,
        help="Layer configuration for the custom CNN",
    )

    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable wandb logging. If not provided, wandb will be disabled.",
    )
    args = parser.parse_args()
    return args


def get_dataloaders(args):
    print("Preparing dataloaders for the" + args.dataset + "dataset")
    if args.dataset == "mnist":
        loader = MNISTDataLoader(args)
    elif args.dataset == "cifar100":
        loader = CIFAR100DataLoader(args)

    elif args.dataset == "cifar10":
        loader = CIFAR10DataLoader(args)
    train_loader, val_loader, test_loader = loader.get_dataloaders()
    return train_loader, val_loader, test_loader


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    train_loader, val_loader, test_loader = get_dataloaders(args)
    num_classes = len(train_loader.dataset.dataset.classes)
    print("Dataset " + args.dataset + "  has " + str(num_classes) + " classes")
    if args.model == "mlp":
        model = MLP([28 * 28] + [args.width] * args.depth + [num_classes], args.bn)
        run_name = f"mlp_w{args.width}_d{args.depth}"
    elif args.model == "resmlp":
        model = ResidualMLP(28 * 28, args.width, args.depth, num_classes, args.bn)
        run_name = f"resmlp_w{args.width}_d{args.depth}"
    elif args.model == "cnn":
        layers = args.layers
        model = CustomCNN(block_type="basic", layers=layers, use_skip=args.skip)
        run_name = f"cnn_skip{args.skip}_layers{layers}"
    if args.use_wandb:
        wandb.init(project="lab_1_DLA", name=run_name, config=args)
    model = model.to(args.device)
    data, _ = next(iter(train_loader))
    data = data.to(args.device)
    summary(model, input_data=data)
    opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    train(model, opt, train_loader, val_loader, criterion, args)
    data, target = next(iter(train_loader))
    data = data.to(args.device)
    target = target.to(args.device)
    check_gradient_signal(model, criterion, data, target, args)
    top1, top5, _ = test(model, test_loader, criterion, 0, args)
    if args.use_wandb:
        wandb.log(
            {
                "top1_test": top1,
                "top5_test": top5,
            }
        )
        wandb.finish()
    else:
        print(f"Top1 accuracy on test set: {top1}")
        print(f"Top5 accuracy on test set: {top5}")


if __name__ == "__main__":
    main()
