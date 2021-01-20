import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Baseline Models (NN Models)

# ---- Baseline Model 1: Conversion NN

class ConversionNN(nn.Module):

    def __init__(self, input_dim, hidden_layers, criterion):

        super().__init__()

        # define layers of the neural network
        self.hidden_layer_sizes = [input_dim] + hidden_layers
        # define the optimizer for a flexible assignment later
        self.optimizer = None
        # loss criterion assignment
        self.criterion = criterion
        
        # define hidden layers with ReLU activation funcs
        self.hidden =\
          nn.Sequential(*[nn.Sequential(nn.Linear(input_, output_), nn.ReLU())
          for input_, output_ in
          zip(self.hidden_layer_sizes, self.hidden_layer_sizes[1:])])

        # define output activation as sigmoid
        self.output = nn.Sequential(nn.Linear(self.hidden_layer_sizes[-1], 1), nn.Sigmoid())

    def forward(self, x):
        # forward iteration for neural network
        x = self.hidden(x)
        x = self.output(x)
        return x

    def train_iteration(self, data_loader):
        # one iteration of training on train data
        running_loss = 0.0
        c_pred = []
        c_true = []
        for i, (X,y,c) in enumerate(data_loader):

            c_hat = self(X)
            c_pred.extend(c_hat.squeeze())
            # since the dataset is reshuffled after each epoch, we need an array with true values of c
            # for positional consistency with the predicted ones
            c_true.extend(c)

            # calculate loss between actual and predicted values, based on chosen criterion
            loss = self.criterion(c_hat.squeeze(), c)
            self.optimizer.zero_grad()   
            # backpropagation     
            loss.backward()
            self.optimizer.step()

            # sum the losses to find running loss
            running_loss += loss.item()

        n_batches = int(np.ceil(len(data_loader.dataset)/data_loader.batch_size))
        # avg loss per batch
        running_loss = running_loss/n_batches
        # calculate performance metric ROC-AUC
        train_auc = roc_auc_score(torch.Tensor(c_true), torch.Tensor(c_pred))

        return running_loss, train_auc

    def val_iteration(self, data_loader):
        # one iteration for validation on validation data
        val_loss = 0.0
        c_pred_val = []
        c_true_val = []
        with torch.no_grad():
            for i, (X_val,y_val,c_val) in enumerate(data_loader):

                c_hat_val = self(X_val)
                c_pred_val.extend(c_hat_val.squeeze())
                c_true_val.extend(c_val)

                # calculate loss between actual and predicted values, based on chosen criterion
                loss = self.criterion(c_hat_val.squeeze(), c_val)
                val_loss += loss.item()

            # calculate performance metric AUC for calculations
            val_auc = roc_auc_score(torch.Tensor(c_true_val), torch.Tensor(c_pred_val))

        return val_loss, val_auc

    def fit(self, train_loader, epochs, val_loader, path, threshold):
        # training and evaluation
        # keep track of the training history
        val_auc_list = []
        cols = ['epoch', 'train_loss', 'train_auc', 'val_loss', 'val_auc']
        history = pd.DataFrame(columns = cols, index = range(epochs))

        for epoch in range(epochs):
            self.train()
            running_loss, train_auc = self.train_iteration(train_loader)
            self.eval()
            val_loss, val_auc = self.val_iteration(val_loader)

            # print results of each train iteration
            print('[%d] loss: %.4f | validation loss: %.4f | auc: %.4f | val_auc: %.4f' % 
                (epoch+1, running_loss, val_loss, train_auc, val_auc))

            # if validation AUC reached the defined threshold, then stop training and save the model
            if val_auc>threshold:
                if len(val_auc_list)>1 and val_auc>max(val_auc_list):
                    torch.save({'model_state_dict': self.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()},
                             dir_path + "/tuning_results/simple_nn_tune/%s/model_%s_%s_%.4f_%d_%.4f.tar" % 
                             (path, self.__class__.__name__, self.hidden_layer_sizes[1:], 
                              self.optimizer.param_groups[0]['lr'], epoch+1, val_auc))
            val_auc_list.append(val_auc)
            values = [epoch+1, running_loss, train_auc, val_loss, val_auc]
            history.iloc[epoch] = values

        history[cols] = history[cols].apply(pd.to_numeric, errors = 'ignore')

        # select iteration number that delivers the maximum validation AUC
        max_val_auc = max(history['val_auc'])
        max_val_auc_epoch = history['epoch'][history['val_auc'].argmax()]

        return max_val_auc, max_val_auc_epoch, history

# ---- Baseline Model 2: Checkout NN

class CheckoutNN(nn.Module):

    def __init__(self, input_dim, hidden_layers, criterion):

        super().__init__()
        
        # define layers of the neural network
        self.hidden_layer_sizes = [input_dim] + hidden_layers
        self.optimizer = None
        self.criterion = criterion

          # define hidden layer activation function as ReLU
        self.hidden =\
            nn.Sequential(*[nn.Sequential(nn.Linear(input_, output_), nn.ReLU())
            for input_, output_ in
            zip(self.hidden_layer_sizes, self.hidden_layer_sizes[1:])])

        # define output activation as linear
        self.output = nn.Linear(self.hidden_layer_sizes[-1], 1)

    def forward(self, x):
        # forward iteration for neural network
        x = self.hidden(x)
        x = self.output(x)
        return x

    def train_iteration(self, data_loader):
        # one iteration of training on train data
        running_loss = 0.0
        for i, (X,y,c) in enumerate(data_loader):

            y_hat = self(X)
          
            # calculate loss using selected loss criterion
            loss = self.criterion(y_hat.squeeze(), y)

            self.optimizer.zero_grad()     
            # back propagation   
            loss.backward()
            self.optimizer.step()

            # sum the losses to find running loss
            running_loss += loss.item()

        n_batches = int(np.ceil(len(data_loader.dataset)/data_loader.batch_size))
        # loss per batch
        running_loss = running_loss/n_batches

        return running_loss

    def val_iteration(self, data_loader):
        # one iteration of validation on validation dataset
        val_loss = 0.0
        with torch.no_grad():
            for i, (X_val,y_val,c_val) in enumerate(data_loader):

                y_hat_val = self(X_val)
                # calculate loss using selected loss criterion
                loss = self.criterion(y_hat_val.squeeze(), y_val)
                val_loss += loss.item()

        return val_loss

    def fit(self, train_loader, epochs, val_loader, path, threshold):
        # training and evaluation
        # keep track of the training history
        val_loss_list = []
        cols = ['epoch', 'train_loss', 'val_loss']
        history = pd.DataFrame(columns = cols, index = range(epochs))

        for epoch in range(epochs):
            # calculate train and validation loss at each epoch
            self.train()
            running_loss = self.train_iteration(train_loader)
            self.eval()
            val_loss = self.val_iteration(val_loader)

            print('[%d] loss: %.4f | validation loss: %.4f' % (epoch+1, running_loss, val_loss))

            # if validation loss reached the defined threshold, then stop training and save the model
            if val_loss<threshold:
                if len(val_loss_list)>1 and val_loss<min(val_loss_list):
                    torch.save({'model_state_dict': self.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()},
                        dir_path + "/tuning_results/simple_nn_tune/%s/model_%s_%s_%.4f_%d_%.4f.tar" % 
                        (path, self.__class__.__name__, self.hidden_layer_sizes[1:], 
                        self.optimizer.param_groups[0]['lr'], epoch+1, val_loss))
            val_loss_list.append(val_loss)
            values = [epoch+1, running_loss, val_loss]
            history.iloc[epoch] = values

        history[cols] = history[cols].apply(pd.to_numeric, errors = 'ignore')

        # select iteration number that delivers the minimum validation MSE
        min_val_loss = min(history['val_loss'])
        min_val_loss_epoch = history['epoch'][history['val_loss'].argmin()]

        return min_val_loss, min_val_loss_epoch, history
