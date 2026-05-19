# For confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch

def plot_confusion_matrix(model, X_test, y_test):
    """
    Plots a confusion matrix for a binary classification model.
    Threshold of 0.5 is used to convert predicted probabilities to class labels.
    """
    preds = model.forward(X_test)

    # Convert tensors to numpy arrays for sklearn compatibility
    # following two lines from chatgpt
    y_true = y_test.detach().cpu().numpy()
    y_pred = (preds >= 0.5).int().detach().cpu().numpy()

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def plot_confusion_matrix_cnn(model, X_test, y_test):
    """
    Plots a confusion matrix for a multi-class CNN model.
    Uses argmax to select the highest-confidence predicted class.
    """
    preds = model.forward(X_test)
    y_true = y_test.cpu().numpy()
    y_pred = preds.argmax(dim=1).detach().cpu().numpy()
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def plot_loss(train_loss, val_loss):
    """Plots training and validation loss curves over epochs."""
    plt.plot(train_loss, label = "Loss")
    plt.plot(val_loss, label="Test Loss")
    plt.legend()
    plt.savefig("cnn_loss.png")
    plt.show()

def accuracy(model, X, y):
    """
    Computes classification accuracy for a multi-class model.
    Returns the fraction of correctly predicted samples.
    """
    with torch.no_grad():
        preds = model.forward(X).argmax(dim=1)
        return (preds == y).float().mean().item()