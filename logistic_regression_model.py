from data_epoching_pipeline import Data_Epoch
import random

# basis of logistic regression model

# train-test split: 

epoch = Data_Epoch()
dataset = epoch.build_dataset("edffile")
ids = list(range(0,49))
random.seed(42)
random.shuffle(ids)
train_ids = ids[:40]
test_ids = ids[41:]
train = [x for x in dataset.data if x[2] in train_ids] # from chatgpt
test = [x for x in dataset.data if x[2] in test_ids] #from chatgpt
# X_train, y_train = pipeline.transform(train)
# X_test, y_test = pipeline.transform(test)
