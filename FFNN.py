import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pytorchtools import EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, auc, recall_score
import torchmetrics
import wandb

# load train data and train labels
X_train = pd.read_csv('data/EstrogenReceptorStatus_Train.csv',index_col=0)
y_train = pd.read_csv('data/EstrogenReceptorStatus_Train_labels.txt',header=None)

# convert them to numpy arrays
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

# load test data and test labels
X_test = pd.read_csv('data/EstrogenReceptorStatus_Test.csv',index_col=0)
y_test = pd.read_csv('data/EstrogenReceptorStatus_Test_labels.txt',header=None)

# convert them to numpy arrays
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

# split training data and labels into training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=14)

# convert features (data) to tensors
X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)

# convert labels to tensors
y_train = torch.LongTensor(y_train)[:,0]
y_val = torch.FloatTensor(y_val)[:,0]
y_test = torch.LongTensor(y_test)[:,0]

# length of the datasets
n_examples_train = len(y_train)
n_examples_test = len(y_test)

# hyperparameters
epochs = 1000
learning_rate = 0.0001
batch_size = 10
n_features = 162
patience = 5
weight_decay_opt = 0.025
delta_es = 0.001

# weight and biases run
# start a new wandb run to track the script
wandb.init(
    project="EstrogenReceptor_FeedForwardNN", 
    config={
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay_opt,        
        'patience': patience, 
        'delta_earlyStopping': delta_es,
        'number_features': n_features,
    })
    
# create the Neural Network, it is a FeedForward
class Network(nn.Module):
    # create the layers of the Network: input layer, two hidden layers, output layer.
    def __init__(self, in_features=n_features, h1=30, h2=30, out_features=1):
        super().__init__() # instantiate the model
        torch.manual_seed(14)
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    # set the activations functions that will be used in every layer
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.out(x))
        return x
    
# set the network    
network = Network()
# set the criterion model to measure the loss/error
loss_criterion = nn.BCELoss()
# set the optimizer and learning rate
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay_opt)
# define early stopping function
early_stopping = EarlyStopping(patience=patience, verbose=True, delta=delta_es)

# train the model
losses_train = []
stop = (n_examples_train//batch_size)*batch_size

# convert the output of the softmax to 0 or 1, taking 0.5 as the thresholds
def toBinary(prediction):
    output = []
    for pred in prediction:
        if pred < 0.5: output.append(0)
        else: output.append(1)
    return output

def training(a, b):
    # go for a prediction
    y_pred_train = network.forward(X_train[a:b,:])
    # measure the loss/error
    loss = loss_criterion(y_pred_train, torch.unsqueeze(y_train[a:b], 1).float())
    # keep track of the losses
    losses_train.append(loss.detach().numpy()) # we dont want it to save it as a tensor

    # back propagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# function for the test/evaluation in which we dont want back propagation, it takes into acount mini batches
# variable that counts how many correct predictions we've got in the evaluation
correct = 0
y_pred_test = []
def evaluation(a,b):
    with torch.no_grad():
        for i, data in enumerate(X_test[a:b,]):
            y_eval = network.forward(data)
            y_eval = toBinary(y_eval)
            # print(y_eval, y_test[i].item())
            # get the number of correct predicitions
            if y_eval[0] == y_test[i].item():
                global correct
                correct += 1
            y_pred_test.append(y_eval[0])

# in this function we get the indexes for the batches                
def get_batches(n_examples, train=1): # train is used to decide if the we are on training or in testing
    for begin in range(0, n_examples, batch_size):
        # indexes for all the batches but the last one
        if begin != stop:
            # define the index for the last example that will be taken into acount in the current batch
            final = begin+batch_size
            # decide if it is training or testing
            if train: training(begin, final)
            else: evaluation(begin, final)
        # indexes fot the last batch
        else: 
            # decide if it is training or testing
            if train: training(begin, X_train.shape[0])
            else: evaluation(begin, X_test.shape[0])

# train the model
losses_train = []
losses_val = []

# TRAINING THE MODEL AND VALIDATION
for epoch in range(epochs):
    get_batches(n_examples_train)

    # VALIDATE THE MODEL
    y_pred_val = network.forward(X_val)
    # measure the loss/error
    loss_val = loss_criterion(y_pred_val, torch.unsqueeze(y_val, 1).float())
    #keep track of the losses
    losses_val.append(loss_val.detach().numpy())
    early_stopping(np.average(losses_val), network)

    # Log the metrics to wandb
    wandb.log({'epoch':epoch, 'loss_train': losses_train[-1], 'loss_validation': loss_val})

    # check if it is needed to early stop
    if early_stopping.early_stop:
        print("Early stopping")
        break

print(losses_train[0].item(), losses_train[-1].item())
print(losses_val[0].item(), losses_val[-1].item())

# test the network
get_batches(n_examples_test, 0)

# get the accuracy and print it
print(f'Accuracy: {correct/len(y_test)}'), losses_val[-1].item()

# calculate and print the F1 score
f1 = f1_score(y_test, y_pred_test)
print('F1 Score:', f1)

# Calculate and print the Recall
recall = recall_score(y_test, y_pred_test)
print("Recall:", recall)

# get the confusion matrix and print it 
conf_matrix = confusion_matrix(y_test, y_pred_test, labels=[1,0])
conf_matrix = pd.DataFrame(conf_matrix)
print(f'Confusion Matrix:\n{conf_matrix}')

# calculate and print the AUROC
auroc = roc_auc_score(y_test, y_pred_test)
print('AUROC:', auroc)
