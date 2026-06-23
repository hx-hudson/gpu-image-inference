import torch
import torch.nn as nn
from model import build_model
from data import build_dataloaders
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train_one_epoch(
        model: nn.Module,
        train_loader,
        optimizer: torch.optim.Optimizer,
        loss_func: nn.Module
):

    model.train()
    total_loss = 0
    correct = 0

    for (x, labels) in train_loader:

        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_func(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.shape[0]
        prediction = logits.argmax(dim=1)
        correct += (prediction == labels).sum().item()

    total_samples = len(train_loader.dataset)
    avg_loss = total_loss / total_samples
    acc = correct / total_samples

    return avg_loss, acc


def evaluate(
        model: nn.Module,
        test_loader,
        loss_func
):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for (x, label) in test_loader:

            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            logits = model(x)
            loss = loss_func(logits, label)

            batch_size = x.shape[0]
            total_loss += loss.item() * batch_size
            prediction = logits.argmax(dim=1)
            correct += (prediction == label).sum().item()

    total_samples = len(test_loader.dataset)
    avg_loss = total_loss / total_samples
    acc = correct / total_samples

    return avg_loss, acc

if __name__ == "__main__":

    set_seed(42)
    epochs = 30

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    save_dir = os.path.join(root_dir, "models")
    save_path = os.path.join(save_dir, "resnet_cifar10.pth")

    model = build_model().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    loss_func = nn.CrossEntropyLoss()
    train_loader, test_loader = build_dataloaders()

    best_acc = 0

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_func=loss_func
        )
        test_loss, test_acc = evaluate(
            model=model, test_loader=test_loader, loss_func=loss_func
        )

        print(
            f"Epoch [{epoch:03d}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc * 100:.2f}% "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc * 100:.2f}%"
        )

        if best_acc < test_acc:
            best_acc = test_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc
                },
                save_path
            )

            print(f"Saved best model with test_acc={best_acc * 100:.2f}%")
