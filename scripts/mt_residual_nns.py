import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from helper_funcs import set_seed
from multitask_nns import MTNet
from baseline_nns import CheckoutNN

# Auxiliary multitask network predcting treatment effects on conversion and spending simultaneously
class MTIte(MTNet):
    def __init__(self, input_dim, joint_layer_size, disjoint_layer_size, criterion, drop):

        super().__init__(input_dim, joint_layer_size, disjoint_layer_size, criterion, criterion, drop, weighted=False)

        self.criterion = criterion
        self.output_c = nn.Linear(self.disjoint_layer_sizes[-1], 1)

# Corresponding residual model that considers heterogeneity in treatment effects
class MTResidualModel(nn.Module):
    
    def __init__(self, mt_model, ite_model, criterion_y, criterion_c, sigm):
        super().__init__()
        self.mt_model = mt_model
        self.ite_model = ite_model
        self.optimizer = None
        self.criterion_y = criterion_y
        self.criterion_c = criterion_c
        self.sigm = sigm
        #for compatibility with daw_fit function
        self.joint_layer_size = self.ite_model.joint_layer_size
        self.disjoint_layer_size = self.ite_model.disjoint_layer_size
        self.drop = self.ite_model.drop
        self.weighted=sigm

    def forward(self, x, T):
        mt_output = self.mt_model(x)
        ite_pred = self.ite_model(x)
        # Either put conversion probabilities through sigmoid or simply set out of range values
        # to boundary values
        if self.sigm == True:
            conv = torch.sigmoid(mt_output[0] + (T * ite_pred[0]))
        else:
            conv = mt_output[0] + (T * ite_pred[0])
            conv[conv>1] = 1
            conv[conv<0] = 0
        checkout = mt_output[2] + (T * ite_pred[1])
        exp_basket = conv*checkout
        return mt_output, ite_pred, conv, checkout, exp_basket

    def train_iteration(self, lambda1, lambda2, data_loader):
        running_loss = 0.0
        train_loss_mse = 0.0
        train_loss_bce = 0.0
        c_true=[]
        c_pred=[]
        for i, (X,y,c,w,g) in enumerate(data_loader):

            _,_,conv,_,exp_basket = self(X, g.unsqueeze(1))

            c_pred.extend(conv.squeeze())
            c_true.extend(c)

            loss1 = self.criterion_y(exp_basket.squeeze(), y)

            if self.weighted == True:
                weighted_criterion = nn.BCELoss(weight = w)
                loss2 = weighted_criterion(conv.squeeze(), c)
            else:
                loss2 = self.criterion_c(conv.squeeze(), c)

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
        train_loss_mse = train_loss_mse/n_batches
        train_loss_bce = train_loss_bce/n_batches
        #for compatibility with the daw_fit function
        bv_train_mse = train_loss_mse

        train_auc = roc_auc_score(torch.Tensor(c_true), torch.Tensor(c_pred))

        return running_loss, train_auc, train_loss_mse, train_loss_bce, bv_train_mse

    def val_iteration(self, data_loader):
        val_loss = 0.0
        val_loss_mse = 0.0
        val_loss_bce = 0.0
        c_true_val=[]
        c_pred_val=[]
        with torch.no_grad():
            for i, (X_val,y_val,c_val,g_val) in enumerate(data_loader):

                _,_,conv_val,_,exp_basket_val = self(X_val, g_val.unsqueeze(1))
                c_pred_val.extend(conv_val.squeeze())
                c_true_val.extend(c_val)
                loss1 = self.criterion_y(exp_basket_val.squeeze(), y_val)
                loss2 = self.criterion_c(conv_val.squeeze(), c_val)
                loss = loss1 + loss2
                val_loss += loss.item()
                val_loss_mse += loss1.item()
                val_loss_bce += loss2.item()
                bv_val_mse = val_loss_mse

            val_auc = roc_auc_score(torch.Tensor(c_true_val), torch.Tensor(c_pred_val))

        return val_loss, val_auc, val_loss_mse, val_loss_bce, bv_val_mse


# Auxiliary NN predicting the treatment effect on basket value
class IteNN(CheckoutNN):
    def __init__(self, input_dim, hidden_layers, criterion):
        super().__init__(input_dim, hidden_layers, criterion)


# Residual model that considers only the homogeneous total treatment effect on the expected basket value
class ResponseResidualModel(nn.Module):
    def __init__(self, mt_model, ite_model, criterion_y, criterion_c):
        super().__init__()
        self.mt_model = mt_model
        self.ite_model = ite_model
        self.optimizer = None
        self.criterion_y = criterion_y
        self.criterion_c = criterion_c
        self.weighted = mt_model.weighted

    def forward(self, x, T):
        mt_output = self.mt_model(x)
        ite_pred = self.ite_model(x)
        basket = mt_output[1] + (T * ite_pred)
        return basket, mt_output, ite_pred

    def train_iteration(self, lambda1, lambda2, data_loader):
        running_loss = 0.0
        train_loss_mse = 0.0
        train_loss_bce = 0.0
        c_true=[]
        c_pred=[]
        for i, (X,y,c,w,g) in enumerate(data_loader):

            cy_hat,mt_outp,_ = self(X, g.unsqueeze(1))

            c_pred.extend(mt_outp[0].squeeze())
            c_true.extend(c)

            loss1 = self.criterion_y(cy_hat.squeeze(), y)

            if self.weighted == True:
                weighted_criterion = nn.BCELoss(weight = w)
                loss2 = weighted_criterion(mt_outp[0].squeeze(), c)
            else:
                loss2 = self.criterion_c(mt_outp[0].squeeze(), c)
          
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
        train_loss_mse = train_loss_mse/n_batches
        train_loss_bce = train_loss_bce/n_batches
        #for compatibility with the daw_fit function
        bv_train_mse = train_loss_mse

        train_auc = roc_auc_score(torch.Tensor(c_true), torch.Tensor(c_pred))

        return running_loss, train_auc, train_loss_mse, train_loss_bce, bv_train_mse

    def val_iteration(self, data_loader):
        val_loss = 0.0
        val_loss_mse = 0.0
        val_loss_bce = 0.0
        c_true_val=[]
        c_pred_val=[]
        with torch.no_grad():
            for i, (X_val,y_val,c_val,g_val) in enumerate(data_loader):

                cy_hat_val,mt_outp_val,_ = self(X_val, g_val.unsqueeze(1))
                c_pred_val.extend(mt_outp_val[0].squeeze())
                c_true_val.extend(c_val)
                loss1 = self.criterion_y(cy_hat_val.squeeze(), y_val)
                loss2 = self.criterion_c(mt_outp_val[0].squeeze(), c_val)
                loss = loss1 + loss2
                val_loss += loss.item()
                val_loss_mse += loss1.item()
                val_loss_bce += loss2.item()
                bv_val_mse = val_loss_mse

        val_auc = roc_auc_score(torch.Tensor(c_true_val), torch.Tensor(c_pred_val))

        return val_loss, val_auc, val_loss_mse, val_loss_bce, bv_val_mse

    def fit(self, path, train_loader, epochs, val_loader, temp):
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
            self.train()
            running_loss, train_auc, train_mse, train_bce, train_bv_loss = self.train_iteration(lambda1, lambda2, train_loader)
            self.eval()
            val_loss, val_auc, val_mse, val_bce, val_bv_loss = self.val_iteration(val_loader)

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
            if val_auc>0.64:
                if len(val_auc_list)>1 and val_auc>max(val_auc_list):
                  torch.save({'model_state_dict': self.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()},
                    "/content/drive/My Drive/apa/%s/models/model_%s_%s_%s_%d_%.4f_%.4f_%.4f.tar" % 
                    (path, self.__class__.__name__, self.weighted, self.ite_model.hidden_layer_sizes[1:],
                    epoch+1, val_auc, val_loss, val_bv_loss))
            #also add saving based on bv loss
            if val_bv_loss<761: 
                if len(val_bv_loss_list)>1 and val_bv_loss<min(val_bv_loss_list):
                    torch.save({'model_state_dict': self.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()},
                        "/content/drive/My Drive/apa/%s/models/model_%s_%s_%s_%d_%.4f_%.4f_%.4f.tar" % 
                        (path, self.__class__.__name__, self.weighted, self.ite_model.hidden_layer_sizes[1:],
                        epoch+1, val_auc, val_loss, val_bv_loss))
            val_auc_list.append(val_auc)
            val_bv_loss_list.append(val_bv_loss)
            values = [epoch+1, running_loss, train_bv_loss, train_auc, train_mse, train_bce, val_loss, val_bv_loss,
                    val_auc, val_mse, val_bce, lambda1, lambda2]
            history.iloc[epoch] = values
            #'early stopping'
            if len(val_auc_list)>19 and all(i < 0.52 for i in val_auc_list[-19:]):
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