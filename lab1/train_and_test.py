from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import math
import wandb


# Train a net for a single epoch, monitoring the loss(where each loss.item() is weighted by how much
# elements are present in the current minibatch)
def train_single_epoch(net, dataloader, train_optimizer, criterion, epoch, args):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloader)
    adjust_learning_rate(train_optimizer, epoch, args)
    for img, label in train_bar:
        img = img.to(args.device)
        label = label.to(args.device)
        logits = net(img)
        loss = criterion(logits, label)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        total_num += img.size(0)
        total_loss += loss.item() * img.size(0)

        train_bar.set_description(
            "Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}".format(
                epoch,
                args.epochs,
                train_optimizer.param_groups[0]["lr"],
                total_loss / total_num,
            )
        )
    return total_loss / total_num


def train(model, opt, train_loader, val_loader, criterion, args):
    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss = train_single_epoch(
            model, train_loader, opt, criterion, epoch, args
        )
        test_acc_1, test_acc_5, val_loss = test(
            model, val_loader, criterion, epoch, args
        )
        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc_1": test_acc_1,
                "val_acc_5": test_acc_5,
                "epoch": epoch,
            }
        )


# Adjust the learning rate to follow a cosine decay schedule or a multi step decay schedule
def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # multi step lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# Function that tests a net given a test dataloader(both a valloader and a testloader can be given as input)
# and computes total top1,top5 and validation loss.
def test(net, test_data_loader, criterion, epoch, args):
    net.eval()
    total_top1, total_top5, total_num = 0.0, 0.0, 0
    total_loss = 0.0
    with torch.no_grad():
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.to(args.device), target.to(args.device)
            logits = net(data)
            # Compute pred_labels and validation loss
            _, pred_labels = torch.topk(logits, k=5, dim=1)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            total_top5 += sum(
                [
                    target[i].item() in pred_labels[i, :5].tolist()
                    for i in range(target.size(0))
                ]
            )
            val_loss = criterion(logits, target)
            total_num += data.size(0)
            total_loss += val_loss.item() * data.size(0)
            test_bar.set_description(
                "Test Epoch: [{}/{}] Acc@1:{:.2f}%, Acc@5:{:.2f}%, Validation Loss: {:.4f}".format(
                    epoch,
                    args.epochs,
                    total_top1 / total_num * 100,
                    total_top5 / total_num * 100,
                    total_loss / total_num,
                )
            )
    return (
        total_top1 / total_num * 100,
        total_top5 / total_num * 100,
        total_loss / total_num,
    )


def check_gradient_signal(model, criterion, data, target, args):
    """
    Computes and plots the norm of gradient updates at each layer of the model and logs the plot to wandb.

    Args:
        model (torch.nn.Module): The model to analyze.
        criterion (torch.nn.Module): The loss function.
        data (torch.Tensor): Input data.
        target (torch.Tensor): Ground truth labels.
    """
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    grad_weights = {}
    grad_biases = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            if "weight" in name:
                grad_weights[name] = param.grad.norm().item()
            elif "bias" in name:
                grad_biases[name] = param.grad.norm().item()

    sorted_weight_layers = sorted(grad_weights.keys())
    sorted_bias_layers = sorted(grad_biases.keys())

    weight_norms = [grad_weights[layer] for layer in sorted_weight_layers]
    bias_norms = [grad_biases[layer] for layer in sorted_bias_layers]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(sorted_weight_layers)), weight_norms, color="b", label="Weights")
    plt.bar(
        range(len(sorted_bias_layers)), bias_norms, color="r", label="Biases", alpha=0.7
    )
    plt.xlabel("Layer Index")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm per Layer")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Log plot directly to wandb
    if args.use_wandb:
        wandb.log({"Gradient Norm Plot": wandb.Image(plt)})
