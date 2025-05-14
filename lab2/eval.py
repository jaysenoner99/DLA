import argparse
import torch
from train_and_test import test_autoencoder_ood, test_cnn_ood, max_logit, max_softmax
from dataloaders import CIFAR10DataLoader, FakeLoader
import wandb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ood detection of a pretrained cnn/ae model"
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
        "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--seed",
        default=69,
        type=int,
        metavar="seed",
        help="Seed of the experiments, to be set for reproducibility",
    )

    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--val-split",
        default=0.1,
        type=float,
        metavar="v",
        help="Proportion of the training data to be reserved to the validation split.",
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
        "--score",
        type=str,
        choices=["logit", "softmax"],
        help="Choose between a max logit or max softmax scoring function",
    )
    args = parser.parse_args()
    return args


args = parse_args()
torch.manual_seed(args.seed)
loader = CIFAR10DataLoader(args)

train_loader, val_loader, test_loader = loader.get_dataloaders()

loader = FakeLoader(args)
fake_loader = loader.get_dataloaders()

run_name = f"evaluate_{args.model}_ood"
num_classes = len(train_loader.dataset.dataset.classes)
print("Dataset has " + str(num_classes) + " classes")
if args.use_wandb:
    wandb.init(project="lab_2_DLA", name=run_name, config=args)

if args.model == "ae":
    test_autoencoder_ood(args, test_loader, fake_loader)

elif args.model == "cnn":
    if args.score == "logit":
        score_fun = max_logit
    elif args.score == "softmax":
        temp = 1000
        score_fun = lambda l: max_softmax(l, 1000)
    test_cnn_ood(args, test_loader, fake_loader, score_fun)
