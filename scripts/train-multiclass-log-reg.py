# Multiclass log-reg model training
# Run from project root with `python train-multiclass-log-reg.py`
import os
import pickle
import sys
from pathlib import Path
from xml.parsers.expat import model

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from matplotlib.pylab import rint
import torch

from dataset import EEGDataset, random_split
from epocher import Data_Epoch
from evaluation import plot_confusion_matrix_cnn, plot_loss
from features import build_feature_matrix, normalize
from multiclass_log_reg import MulticlassLogReg, GradientDescentOptimizer, cross_entropy, accuracy



DATA_DIR     = "data"
SEED         = 42
N_SUBJECTS   = 49
N_TRAIN      = 40
LR           = 0.01
EPOCHS       = 1000
LOG_EVERY    = 100
TRAIN_FRAC     = 0.8
USE_BEST_MODEL = True


def main():
    # Build dataset
    print("Loading and epoching EDF files...")
    dataset = Data_Epoch()
    dataset = dataset.build_dataset(DATA_DIR)
    print(f"Total trials: {dataset.__len__()}")

    # Split
    train_data, test_data = random_split(dataset, frac=TRAIN_FRAC)

    X_train, y_train, subj_train = build_feature_matrix(train_data)
    X_test, y_test, subj_test = build_feature_matrix(test_data)

    # Normalize
    X_train, X_test = normalize(X_train, X_test, subj_train, subj_test)


    # Cast X to float, y to long for loss and accuracy
    X_train = X_train.float()
    y_train = y_train.long()
    X_test = X_test.float()
    y_test = y_test.long()

    # Model and optimizer
    n_features = X_train.shape[1]
    n_classes = len(torch.unique(y_train))
    if USE_BEST_MODEL and os.path.exists("best_model.pth"):
        with open("best_model.pth", "rb") as f:
            model = pickle.load(f)
        print("Loaded best model from disk.")
    else:
        model = MulticlassLogReg(d_features=n_features, k_classes=n_classes)
        print("Initialized new model.")

    counts = torch.bincount(y_train).float()
    class_weights = 1.0 / (counts + 1e-8)
    class_weights = class_weights / class_weights.sum()

    opt   = GradientDescentOptimizer(model, lr=LR, class_weights=class_weights)

    train_losses = []
    test_losses = []


    # Training loop
    print(f"\nTraining for {EPOCHS} epochs (lr={LR})...")
    for ep in range(EPOCHS):
        q    = model.forward(X_train)
        loss = cross_entropy(q, y_train, weights=class_weights)
        opt.step(X_train, y_train)

        train_losses.append(loss.item())
        test_q = model.forward(X_test)
        test_loss = cross_entropy(test_q, y_test, weights=class_weights)
        test_losses.append(test_loss.item())
 
        if (ep + 1) % LOG_EVERY == 0:
            train_acc = accuracy(model, X_train, y_train)
            print(
                f"  Epoch {ep+1:3d} | "
                f"loss: {loss.item():.4f} | "
                f"train acc: {train_acc:.3f}")

    # Evaluation
    test_acc = accuracy(model, X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.3f}")

    with open("best_model.pth", "wb") as f:
        pickle.dump(model, f)

    plot_loss(train_losses, test_losses)


    if os.path.exists("best_model.pth"):  
        model = torch.load("best_model.pth")
    plot_confusion_matrix_cnn(model, X_test, y_test)
    print("train accuracy: ", accuracy(model, X_train, y_train))
    print("test accuracy: ", accuracy(model, X_test, y_test))

    # saves model via pickle for version control
    with open("model_in_prog.pkl", "wb") as f:
        pickle.dump(model, f)
 
 
if __name__ == "__main__":
    main()