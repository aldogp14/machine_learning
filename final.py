import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import wandb
from pytorchtools import EarlyStopping

# Load train data and train labels
X_train = pd.read_csv('data/EstrogenReceptorStatus_Train.csv', index_col=0)
y_train = pd.read_csv('data/EstrogenReceptorStatus_Train_labels.txt', header=None)

# Convert them to numpy arrays
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

# Load test data and test labels
X_test = pd.read_csv('data/EstrogenReceptorStatus_Test.csv', index_col=0)
y_test = pd.read_csv('data/EstrogenReceptorStatus_Test_labels.txt', header=None)

# Convert them to numpy arrays
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

# Split training data and labels into training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=14)

# Convert features (data) to tensors
X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)

# Convert labels to tensors
y_train = torch.LongTensor(y_train)[:, 0]
y_val = torch.FloatTensor(y_val)[:, 0]
y_test = torch.LongTensor(y_test)[:, 0]

# Length of the datasets
n_examples_train = len(y_train)

n_features = 162

# Hyperparameters
epochs = 1000
learning_rate = 0.0001
batch_size = 10
patience = 5
weight_decay_opt = 0.025
delta_es = 0.001

# Define the sweep configuration
sweep_configuration = {
    'method': 'random',
    'name': 'EstrogenReceptor_FeedForwardNN',
    'metric': {
        'goal': 'minimize',
        'name': 'loss_train',  # Use the validation loss as the metric
    },
    'parameters': {
        'epochs': {'value': 2000},
        'batch_size': {'min': 3, 'max':162},
        'learning_rate': {'min': 0.00005, 'max': 0.1},
        'patience': {'min': 3, 'max': 25},
        'weight_decay_opt': {'min': 0.01, 'max': 0.5},
        'delta_es': {'min': 0.0001, 'max': 0.1},
    },
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project='EstrogenReceptor_FeedForwardNN')

# Create the Neural Network, it is a FeedForward
class Network(nn.Module):
    def __init__(self, in_features=n_features, h1=30, h2=30, out_features=1):
        super().__init__()
        torch.manual_seed(14)
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x

# Set the network
network = Network()

# set the lists
losses_train = []
losses_val = []

# Convert the output of the softmax to 0 or 1, taking 0.5 as the threshold
def toBinary(prediction):
    output = []
    for pred in prediction:
        if pred < 0.5:
            output.append(0)
        else:
            output.append(1)
    return output

def training(a, b, optimizer, loss_criterion,):
    y_pred_train = network.forward(X_train[a:b, :])
    loss = loss_criterion(y_pred_train, torch.unsqueeze(y_train[a:b], 1).float())
    losses_train.append(loss.detach().numpy())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

correct = 0
y_pred_test = []

def evaluation(a, b):
    with torch.no_grad():
        for i, data in enumerate(X_test[a:b, ]):
            y_eval = network.forward(data)
            y_eval = toBinary(y_eval)

            if y_eval[0] == y_test[i].item():
                global correct
                correct += 1
            y_pred_test.append(y_eval[0])

def get_batches(n_examples, batch_size, optimizer, loss_criterion, stop, train=1):
    for begin in range(0, n_examples, batch_size):
        if begin != stop:
            final = begin + batch_size
            if train:
                training(begin, final, optimizer, loss_criterion,)
            else:
                evaluation(begin, final)
        else:
            if train:
                training(begin, X_train.shape[0], optimizer, loss_criterion,)
            else:
                evaluation(begin, X_test.shape[0])

# TRAINING THE MODEL AND VALIDATION
def train():

    wandb.init(project='EstrogenReceptor_FeedForwardNN')

    epochs = wandb.config['epochs']
    batch_size = wandb.config['batch_size']
    learning_rate = wandb.config['learning_rate']
    patience = wandb.config['patience']
    weight_decay_opt = wandb.config['weight_decay_opt']
    delta_es = wandb.config['delta_es']

    torch.manual_seed(14)

    # Set the criterion model to measure the loss/error
    loss_criterion = nn.BCELoss()

    # Set the optimizer and learning rate
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay_opt)

    # Define early stopping function
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=delta_es)

    stop = (n_examples_train // batch_size) * batch_size

    for epoch in range(epochs):
        # training
        get_batches(n_examples_train, batch_size, optimizer, loss_criterion, stop)

        # validation
        y_pred_val = network.forward(X_val)
        loss_val = loss_criterion(y_pred_val, torch.unsqueeze(y_val, 1).float())
        losses_val.append(loss_val.detach().numpy())
        early_stopping(np.average(losses_val), network)

        wandb.log({'loss_train': losses_train[-1], 'loss_validation': loss_val})

        if early_stopping.early_stop:
            print('Early stopping')
            break

# Run the sweep agent command
wandb.agent(sweep_id, function=train, count=100)
