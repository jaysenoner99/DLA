import argparse
from model import CNN, Autoencoder
import torch
from train_and_test import train, test, train_ae, test_ae
from dataloaders import CIFAR10DataLoader, FakeLoader
import torch.nn as nn
import wandb


def parse_args_main():
    parser = argparse.ArgumentParser(
        description="Train a MLP/ResMLP/CNN on MNIST/CIFAR10/CIFAR100"
    )

    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.0001,
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
        default=[125, 175],
        nargs="*",
        type=int,
        help="learning rate schedule (when to drop lr by 10x)",
    )
    parser.add_argument(
        "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument(
        "--device",
        default=device,
        type=str,
        metavar="D",
        help="The device where the training process will be executed. Autoset to cuda if cuda is available on the local machine",
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
        "--fgsm",
        action="store_true",
        help="Enable training with FGSM perturbed samples",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.07,
        help="Epsilon value for the FGSM perturbation. Defaults to 0.07",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Name of the .pth file that contains the model weights",
    )
    args = parser.parse_args()
    return args


def main():
    print("Running main.py")
    args = parse_args_main()
    torch.manual_seed(args.seed)
    loader = CIFAR10DataLoader(args)

    train_loader, val_loader, test_loader = loader.get_dataloaders()

    loader = FakeLoader(args)

    run_name = f"simple_{args.model}_fgsm_{args.fgsm}"
    num_classes = len(train_loader.dataset.dataset.classes)
    print("Dataset has " + str(num_classes) + " classes")
    if args.use_wandb:
        wandb.init(project="lab_2_DLA", name=run_name, config=args)
    if args.model == "cnn":
        model = CNN()
        criterion = nn.CrossEntropyLoss()
        model = model.to(args.device)
        opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)

        train(model, opt, train_loader, val_loader, criterion, args)
        model_name = f"cifar10_{args.model}_fgsm_{args.fgsm}"
        torch.save(model.state_dict(), "./" + model_name + ".pth")
        top1, top5, _ = test(model, test_loader, criterion, 0, args)
        if args.use_wandb:
            wandb.log(
                {
                    "top1_test": top1,
                    "top5_test": top5,
                }
            )
        else:
            print(f"Top1 accuracy on test set: {top1}")
            print(f"Top5 accuracy on test set: {top5}")
    elif args.model == "ae":
        model = Autoencoder()
        criterion = nn.MSELoss()
        model = model.to(args.device)
        opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        train_ae(model, opt, train_loader, val_loader, criterion, args)
        model_name = f"cifar10_{args.model}_fgsm_{args.fgsm}"
        torch.save(model.state_dict(), "./" + model_name + ".pth")
        scores_test_ae, val_loss = test_ae(model, test_loader, criterion, 0, args)

    wandb.finish()


if __name__ == "__main__":
    main()
