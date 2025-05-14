from tqdm import tqdm
import torch.nn.functional as F
from sklearn import metrics
from model import CNN, Autoencoder
import torch.nn as nn
import torch
import math
import wandb
import matplotlib.pyplot as plt


def train_single_epoch_ae(net, dataloader, train_optimizer, criterion, epoch, args):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloader)
    adjust_learning_rate(train_optimizer, epoch, args)
    for data in train_bar:
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        z, x_rec = net(x)
        loss = criterion(x, x_rec)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += x.size(0)
        total_loss += loss.item() * x.size(0)

        train_bar.set_description(
            "Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}".format(
                epoch,
                args.epochs,
                train_optimizer.param_groups[0]["lr"],
                total_loss / total_num,
            )
        )
    return total_loss / total_num


def train_ae(model, opt, train_loader, val_loader, criterion, args):
    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss = train_single_epoch_ae(
            model, train_loader, opt, criterion, epoch, args
        )
        test_scores_ae, val_loss = test_ae(model, val_loader, criterion, epoch, args)
        if args.use_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch": epoch,
                }
            )


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
        if args.use_wandb:
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


def test_ae(net, test_loader, criterion, epoch, args):
    net.eval()
    scores_test_ae = []
    total_loss = []
    criterion = nn.MSELoss(reduction="none")
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            x, y = data
            x = x.to(args.device)
            z, x_rec = net(x)
            loss = criterion(x, x_rec)
            score = loss.mean([1, 2, 3])
            scores_test_ae.append(-score)
            total_loss.append(score)
            test_bar.set_description(
                "Test Epoch: [{}/{}]:  Validation Loss: {:.4f}".format(
                    epoch, args.epochs, torch.mean(torch.cat(total_loss))
                )
            )
    scores_test_ae = torch.cat(scores_test_ae)

    return scores_test_ae, torch.mean(torch.cat(total_loss))


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


def test_autoencoder_ood(args, test_loader, fake_loader):
    model = Autoencoder()
    model.to(args.device)
    model.load_state_dict(torch.load("cifar10_ae.pth"))
    criterion = nn.MSELoss(reduction="none")
    scores_test, _ = test_ae(model, test_loader, criterion, 0, args)
    scores_fake, _ = test_ae(model, fake_loader, criterion, 0, args)

    real_fake, ax = plt.subplots()
    ax.hist(scores_test.cpu(), density=True, alpha=0.5, bins=25, label="test")
    ax.hist(scores_fake.cpu(), density=True, alpha=0.5, bins=25, label="fake")
    ax.legend()

    fig_sorted_scores, ax = plt.subplots()
    ax.plot(sorted(scores_test.cpu()), label="test")
    ax.plot(sorted(scores_fake.cpu()), label="fake")
    ax.legend()

    ypred = torch.cat((scores_test, scores_fake))
    y_test = torch.ones_like(scores_test)
    y_fake = torch.zeros_like(scores_fake)
    y = torch.cat((y_test, y_fake))
    roc_curve = metrics.RocCurveDisplay.from_predictions(y.cpu(), ypred.cpu())
    fig_roc = roc_curve.figure_
    pr_display = metrics.PrecisionRecallDisplay.from_predictions(y.cpu(), ypred.cpu())
    fig_pr = pr_display.figure_
    if args.use_wandb:
        wandb.log(
            {
                "Real vs. Fake scores": wandb.Image(real_fake),
                "Sorted Real vs. Fake": wandb.Image(fig_sorted_scores),
                "ROC curve": wandb.Image(fig_roc),
                "PR curve": wandb.Image(fig_pr),
            }
        )
        wandb.finish()


def test_cnn_ood(args, test_loader, fake_loader, score_fun):
    model = CNN()
    model.to(args.device)
    model.load_state_dict(torch.load("cifar10_CNN.pth"))
    scores_test = compute_scores(test_loader, score_fun, model, args)
    scores_fake = compute_scores(fake_loader, score_fun, model, args)

    real_fake, ax = plt.subplots()
    ax.hist(scores_test.cpu(), density=True, alpha=0.5, bins=25, label="test")
    ax.hist(scores_fake.cpu(), density=True, alpha=0.5, bins=25, label="fake")
    ax.legend()
    ypred = torch.cat((scores_test, scores_fake))
    y_test = torch.ones_like(scores_test)
    y_fake = torch.zeros_like(scores_fake)
    y = torch.cat((y_test, y_fake))

    fig_sorted_scores, ax = plt.subplots()
    ax.plot(sorted(scores_test.cpu()), label="test")
    ax.plot(sorted(scores_fake.cpu()), label="fake")
    ax.legend()

    roc_curve = metrics.RocCurveDisplay.from_predictions(y.cpu(), ypred.cpu())
    fig_roc = roc_curve.figure_
    pr_display = metrics.PrecisionRecallDisplay.from_predictions(y.cpu(), ypred.cpu())
    fig_pr = pr_display.figure_
    if args.use_wandb:
        wandb.log(
            {
                "Real vs. Fake scores": wandb.Image(real_fake),
                "Sorted Real vs. Fake": wandb.Image(fig_sorted_scores),
                "ROC curve": wandb.Image(fig_roc),
                "PR curve": wandb.Image(fig_pr),
            }
        )
        wandb.finish()


def max_logit(logit):
    s = logit.max(dim=1)[0]  # get the max for each element of the batch
    return s


def max_softmax(logit, T=1.0):
    s = F.softmax(logit / T, 1)
    s = s.max(dim=1)[0]  # get the max for each element of the batch
    return s


def compute_scores(data_loader, score_fun, model, args):
    scores = []
    with torch.no_grad():
        for data in data_loader:
            x, y = data
            output = model(x.to(args.device))
            s = score_fun(output)
            scores.append(s)
        scores_t = torch.cat(scores)
        return scores_t
