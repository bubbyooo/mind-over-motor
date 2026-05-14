# For confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(model, X_test, y_test):
    preds = model.forward(X_test)

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
    preds = model.forward(X_test)
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def plot_loss(train_loss, val_loss):
    plt.plot(train_loss, label = "Loss")
    plt.plot(val_loss, label="Test Loss")
    plt.legend()
    plt.show()