import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from helper_funcs import set_seed
from sklearn.metrics import roc_auc_score
import math


class MTNet(nn.Module):
    def __init__(self, input_dim, joint_layer_size, disjoint_layer_size, criterion_y, criterion_c, drop, weighted):
        super().__init__()
        self.joint_layer_size = joint_layer_size
        self.joint_layer_sizes = [input_dim] + [joint_layer_size, joint_layer_size, joint_layer_size]
        self.disjoint_layer_size = disjoint_layer_size
        self.disjoint_layer_sizes = [joint_layer_size] + [disjoint_layer_size, disjoint_layer_size, disjoint_layer_size]
        self.drop = drop
        self.optimizer = None
        self.criterion_y = criterion_y
        self.criterion_c = criterion_c
        self.weighted = weighted

        self.joint =\
            nn.Sequential(*[nn.Sequential(nn.Linear(input_, output_), nn.ReLU(), nn.Dropout(drop))
            for input_, output_ in 
            zip(self.joint_layer_sizes, self.joint_layer_sizes[1:])])

        self.disjoint_c =\
            nn.Sequential(*[nn.Sequential(nn.Linear(input_, output_), nn.ReLU(), nn.Dropout(drop))
            for input_, output_ in
            zip(self.disjoint_layer_sizes, self.disjoint_layer_sizes[1:])])

        self.disjoint_r =\
            nn.Sequential(*[nn.Sequential(nn.Linear(input_, output_), nn.ReLU(), nn.Dropout(drop))
            for input_, output_ in
            zip(self.disjoint_layer_sizes, self.disjoint_layer_sizes[1:])])

        self.output_c = nn.Sequential(nn.Linear(self.disjoint_layer_sizes[-1], 1), nn.Sigmoid())
        self.output_r = nn.Linear(self.disjoint_layer_sizes[-1], 1)

    def forward(self, x):
        x = self.joint(x)
        c = self.disjoint_c(x)
        y = self.disjoint_r(x)
        c = self.output_c(c)
        y = self.output_r(y)
        return c,y

    def train_iteration(self, data_loader):
        running_loss = 0.0
        train_loss_mse = 0.0
        train_loss_bce = 0.0
        basket_value_train_mse = 0.0
        c_pred = []
        c_true = []
        for i, (X,y,c,w) in enumerate(data_loader):
            pred = self(X)
            c_hat, y_hat = pred[0], pred[1]
            c_pred.extend(c_hat.squeeze())
            #since the dataset is reshuffled after each epoch, need an array with true values of c
            #for positional consistency with the predicted ones, in order to correctly calculate AUROC
            c_true.extend(c)

            loss1 = self.criterion_y(y_hat.squeeze(), y)
            if self.weighted == True:
                weighted_criterion = nn.BCELoss(weight = w)
                loss2 = weighted_criterion(c_hat.squeeze(), c)
            else:
                loss2 = self.criterion_c(c_hat.squeeze(), c)
            loss3 = self.criterion_y((y_hat*c_hat).squeeze(), y)
            loss = loss1 + loss2

            self.optimizer.zero_grad()        
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            train_loss_mse += loss1.item()
            train_loss_bce += loss2.item()
            basket_value_train_mse += loss3.item()

        n_batches = int(np.ceil(len(data_loader.dataset)/data_loader.batch_size))
        running_loss = running_loss/n_batches
        basket_value_train_mse = basket_value_train_mse/n_batches
        train_loss_mse = train_loss_mse/n_batches
        train_loss_bce = train_loss_bce/n_batches

        train_auc = roc_auc_score(torch.Tensor(c_true), torch.Tensor(c_pred))

        # clunky workaround to avoid rewriting the whole train_iteration for MTXNet
        if 'MTXNet' in self.__class__.__name__:
            basket_value_train_mse = train_loss_mse

        return running_loss, train_auc, train_loss_mse, train_loss_bce, basket_value_train_mse

    def val_iteration(self, data_loader):
        val_loss = 0.0
        val_loss_mse = 0.0
        val_loss_bce = 0.0
        basket_value_val_mse = 0.0
        c_pred_val = []
        c_true_val = []
        with torch.no_grad():
            for i, (X_val,y_val,c_val) in enumerate(data_loader):

                pred_val = self(X_val)
                c_hat_val, y_hat_val = pred_val[0], pred_val[1]
                c_pred_val.extend(c_hat_val.squeeze())
                #here not really needed, since we use the full bacth for validation, but is kept for consistency
                #batching can be required during validation when we are dealing with huge datasets 
                c_true_val.extend(c_val)
                loss1 = self.criterion_y(y_hat_val.squeeze(), y_val)
                loss2 = self.criterion_c(c_hat_val.squeeze(), c_val)
                loss3 = self.criterion_y((y_hat_val*c_hat_val).squeeze(), y_val)
                loss = loss1 + loss2
                val_loss += loss.item()
                val_loss_mse += loss1.item()
                val_loss_bce += loss2.item()
                basket_value_val_mse += loss3.item()

            val_auc = roc_auc_score(torch.Tensor(c_true_val), torch.Tensor(c_pred_val))

            if 'MTXNet' in self.__class__.__name__:
                basket_value_val_mse = val_loss_mse

        return val_loss, val_auc, val_loss_mse, val_loss_bce, basket_value_val_mse


class MTCat(MTNet):

    def __init__(self, input_dim, joint_layer_size, disjoint_layer_size, criterion_y, criterion_c, drop, weighted):
        super().__init__(input_dim, joint_layer_size, disjoint_layer_size, criterion_y, criterion_c, drop, weighted)
        self.output_c = nn.Sequential(nn.Linear(self.disjoint_layer_sizes[-1]*2, 1), nn.Sigmoid())
        self.output_r = nn.Linear(self.disjoint_layer_sizes[-1]*2, 1)

    def forward(self, x): 
        x = self.joint(x)
        c = self.disjoint_c(x)
        y = self.disjoint_r(x)
        cat = torch.cat((c, y), 1)
        c = self.output_c(cat)
        y = self.output_r(cat)
        return c,y


class MTXNet(MTNet):

    def __init__(self, input_dim, joint_layer_size, disjoint_layer_size, criterion_y, criterion_c, drop, weighted):
        super().__init__(input_dim, joint_layer_size, disjoint_layer_size, criterion_y, criterion_c, drop, weighted)

    def forward(self, x):    
        x = self.joint(x)
        c = self.disjoint_c(x)
        y = self.disjoint_r(x)
        c = self.output_c(c)
        y = self.output_r(y)
        cy = c*y
        return c,cy,y


class DAW_MTXNet(MTXNet):

  def __init__(self, input_dim, joint_layer_size, disjoint_layer_size, criterion_y, criterion_c, drop, weighted):
    super().__init__(input_dim, joint_layer_size, disjoint_layer_size, criterion_y, criterion_c, drop, weighted)

    def train_iteration(self, lambda1, lambda2, data_loader):
        running_loss = 0.0
        train_loss_mse = 0.0
        train_loss_bce = 0.0
        basket_value_train_mse = 0.0
        c_pred = []
        c_true = []
        for i, (X,y,c,w) in enumerate(data_loader):
            pred = self(X)
            c_hat, y_hat = pred[0], pred[1]
            c_pred.extend(c_hat.squeeze())
            #since the dataset is reshuffled after each epoch, need an array with true values of c
            #for positional consistency with the predicted ones
            c_true.extend(c)

            loss1 = self.criterion_y(y_hat.squeeze(), y)
            if self.weighted == True:
                weighted_criterion = nn.BCELoss(weight = w)
                loss2 = weighted_criterion(c_hat.squeeze(), c)
            else:
                loss2 = self.criterion_c(c_hat.squeeze(), c)

            loss = lambda1*loss1 + lambda2*loss2

            self.optimizer.zero_grad()        
            loss.backward()
            self.optimizer.step()

            loss = loss1 + loss2
            running_loss += loss.item()
            train_loss_mse += loss1.item()
            train_loss_bce += loss2.item()

        n_batches = int(np.ceil(len(data_loader.dataset)/data_loader.batch_size))
        running_loss = running_loss/n_batches
        basket_value_train_mse = basket_value_train_mse/n_batches
        train_loss_mse = train_loss_mse/n_batches
        train_loss_bce = train_loss_bce/n_batches
        basket_value_train_mse = train_loss_mse

        train_auc = roc_auc_score(torch.Tensor(c_true), torch.Tensor(c_pred))

        return running_loss, train_auc, train_loss_mse, train_loss_bce, basket_value_train_mse


def fit_mt(model, path, train_loader, epochs, val_loader):
    # auc and loss lists necessary for tracking performance
    val_auc_list = []
    val_bv_loss_list = []
    # instantiating a data frame that saves history
    cols = ['epoch', 'train_loss', 'basket_value_train_loss', 'train_auc', 'train_mse', 'train_bce',
        'val_loss', 'basket_value_val_loss', 'val_auc', 'val_mse', 'val_bce']
    history = pd.DataFrame(columns = cols, index = range(epochs))
    for epoch in range(epochs):
        # Set the model into training mode - dropout ON
        model.train()
        running_loss, train_auc, train_mse, train_bce, train_bv_loss = model.train_iteration(train_loader)
        # Set the model into evaluation mode - dropout OFF
        model.eval()
        val_loss, val_auc, val_mse, val_bce, val_bv_loss = model.val_iteration(val_loader)
        # print out training information
        print('[%d] loss: %.4f | bv train loss: %.4f | validation loss: %.4f | bv val loss: %.4f | auc: %.4f | val auc: %.4f' % 
              (epoch+1, running_loss, train_bv_loss, val_loss, val_bv_loss, train_auc, val_auc))
        # save good performing models
        if val_auc>0.64:
            if len(val_auc_list)>1 and val_auc>max(val_auc_list):
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': model.optimizer.state_dict()},
                    "/tuning_results/%s/models/model_%s_%s_%d_%d_%.1f_%d_%.4f_%.4f_%.4f.tar" % 
                    (path, model.__class__.__name__, model.weighted, model.joint_layer_size, model.disjoint_layer_size,
                        model.drop, epoch+1, val_auc, val_loss, val_bv_loss))
        # also add saving based on bv loss
        if val_bv_loss<771:
            if len(val_bv_loss_list)>1 and val_bv_loss<min(val_bv_loss_list):
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': model.optimizer.state_dict()},
                    dir_path + "/tuning_results/%s/models/model_%s_%s_%d_%d_%.1f_%d_%.4f_%.4f_%.4f.tar" %
                    (path, model.__class__.__name__, model.weighted, model.joint_layer_size, model.disjoint_layer_size,
                        model.drop, epoch+1, val_auc, val_loss, val_bv_loss))
        val_auc_list.append(val_auc)
        val_bv_loss_list.append(val_bv_loss)
        values = [epoch+1, running_loss, train_bv_loss, train_auc, train_mse, train_bce, val_loss, 
                  val_bv_loss, val_auc, val_mse, val_bce]
        history.iloc[epoch] = values
        # 'early stopping'
        if len(val_auc_list)>9 and all(i < 0.52 for i in val_auc_list[-9:]):
            break

    history[cols] = history[cols].apply(pd.to_numeric, errors = 'ignore')
    return history


def evaluate(model, data_loader):
    test_loss = 0.0
    basket_value_test_mse = 0.0
    test_loss_mse = 0.0
    c_pred_test = []
    c_true_test = []
    y_pred_test = []
    with torch.no_grad():
        for i, (X_test,y_test,c_test) in enumerate(data_loader):
            pred = model(X_test)
            c_hat_test, y_hat_test = pred[0], pred[1]
            c_pred_test.extend(c_hat_test.squeeze())
            c_true_test.extend(c_test)
            loss1 = model.criterion_y(y_hat_test.squeeze(), y_test)
            loss2 = model.criterion_c(c_hat_test.squeeze(), c_test)
            loss3 = model.criterion_y((y_hat_test*c_hat_test).squeeze(), y_test)
            loss = loss1 + loss2
            test_loss += loss.item()
            test_loss_mse += loss1.item()
            basket_value_test_mse += loss3.item()
    test_auc = roc_auc_score(torch.Tensor(c_true_test), torch.Tensor(c_pred_test))
    if model.__class__.__name__ == 'MTXNet':
      basket_value_test_mse = test_loss_mse

    return test_loss, test_auc, basket_value_test_mse, c_hat_test, y_hat_test


def daw_fit(model, path, train_loader, epochs, val_loader, temp, th_auc, th_loss, th_stop):
    val_auc_list = []
    val_bv_loss_list = []
    cols = ['epoch', 'train_loss', 'basket_value_train_loss', 'train_auc', 'train_mse', 'train_bce', 
          'val_loss', 'basket_value_val_loss', 'val_auc', 'val_mse', 'val_bce', 'lambda1', 'lambda2']
    history = pd.DataFrame(columns = cols, index = range(epochs))
    lambda1 = 1
    lambda2 = 1
    T = torch.tensor([temp], dtype=float, device=device, requires_grad=False)
    train_mse_prev = 0
    train_bce_prev = 0
    for epoch in range(epochs):
        model.train()
        running_loss, train_auc, train_mse, train_bce, train_bv_loss = model.train_iteration(lambda1, lambda2, train_loader)
        model.eval()
        val_loss, val_auc, val_mse, val_bce, val_bv_loss = model.val_iteration(val_loader)

        if epoch>1:
            print("Loss Y: ", train_mse, " Loss Y_prev: ", train_mse_prev)
            print("Loss C: ", train_bce, " Loss C_prev: ", train_bce_prev)
            w1 = train_mse/train_mse_prev
            w2 = train_bce/train_bce_prev
            try:
                lambda1 = 2*math.exp(w1/T)/(math.exp(w1/T)+math.exp(w2/T))
                lambda2 = 2*math.exp(w2/T)/(math.exp(w1/T)+math.exp(w2/T))
            except OverflowError:
                lambda1 = 1
                lambda2 = 1

        print("Lambda1: ", lambda1, " - Lambda2: ", lambda2)

        train_mse_prev = train_mse
        train_bce_prev = train_bce

        #print out training information
        print('[%d] loss: %.4f | bv train loss: %.4f | validation loss: %.4f | bv val loss: %.4f | auc: %.4f | val_auc: %.4f' % 
              (epoch+1, running_loss, train_bv_loss, val_loss, val_bv_loss, train_auc, val_auc))
        #save good performing models
        if val_auc>th_auc:
            if len(val_auc_list)>1 and val_auc>max(val_auc_list):
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': model.optimizer.state_dict()},
                    dir_path + "/tuning_results/%s/models/model_%s_%s_%d_%d_%.1f_%.2f_%d_%.4f_%.4f_%.4f.tar" % 
                    (path, model.__class__.__name__, model.weighted, model.joint_layer_size, model.disjoint_layer_size, 
                    model.drop, temp, epoch+1, val_auc, val_loss, val_bv_loss))
        #also add saving based on bv loss
        if val_bv_loss<th_loss: 
            if len(val_bv_loss_list)>1 and val_bv_loss<min(val_bv_loss_list):
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': model.optimizer.state_dict()},
                    dir_path + "/tuning_results/%s/models/model_%s_%s_%d_%d_%.1f_%.2f_%d_%.4f_%.4f_%.4f.tar" % 
                    (path, model.__class__.__name__, model.weighted, model.joint_layer_size, model.disjoint_layer_size, 
                    model.drop, temp, epoch+1, val_auc, val_loss, val_bv_loss))
        val_auc_list.append(val_auc)
        val_bv_loss_list.append(val_bv_loss)
        values = [epoch+1, running_loss, train_bv_loss, train_auc, train_mse, train_bce, val_loss, val_bv_loss,
                  val_auc, val_mse, val_bce, lambda1, lambda2]
        history.iloc[epoch] = values
        #'early stopping'
        if len(val_auc_list)>th_stop and all(i < 0.52 for i in val_auc_list[-th_stop:]):
            break

    history[cols] = history[cols].apply(pd.to_numeric, errors = 'ignore')

    max_auc = max(history['val_auc'])
    max_auc_loss = history['val_loss'][history['val_auc'].argmax()]
    max_auc_epoch = history['epoch'][history['val_auc'].argmax()]
    max_auc_bv_loss = history['basket_value_val_loss'][history['val_auc'].argmax()]

    min_val_bv_loss = min(history['basket_value_val_loss'])
    min_val_bv_epoch = history['epoch'][history['basket_value_val_loss'].argmin()]
    min_val_bv_auc = history['val_auc'][history['basket_value_val_loss'].argmin()]
    min_val_bv_yloss = history['val_loss'][history['basket_value_val_loss'].argmin()]

    return (max_auc, max_auc_loss, max_auc_epoch, max_auc_bv_loss, min_val_bv_loss, min_val_bv_epoch, 
            min_val_bv_auc, min_val_bv_yloss, history)

  