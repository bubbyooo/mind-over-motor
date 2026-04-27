from data_epoching_pipeline import Data_Epoch
from features import build_feature_set
import random

# basis of logistic regression model

# train-test split: 

epoch = Data_Epoch()
dataset = epoch.build_dataset("edffile")
print("length of dataset: ", len(dataset))
ids = list(range(0,49))
random.seed(42)
random.shuffle(ids)
train_ids = ids[:40]
test_ids = ids[41:]
train = [x for x in dataset if x['subject'] in train_ids] # from chatgpt
test = [x for x in dataset if x['subject'] in test_ids] #from chatgpt
X_train, y_train = build_feature_set(train)
X_test, y_test = build_feature_set(test)


import torch

# copied from lecture mar2
def binary_cross_entropy(q, y):
    return -(y * torch.log(q) + (1-y)*torch.log(1-q)).mean()
