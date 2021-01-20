import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from helper_funcs import set_seed
from mt_residual_nns import *
from multitask_nns import fit_mt, DAW_MTXNet
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()

# Specify the gpu device used for training
# WARNING: the GPU device affects training reproducibility! (presumably because of floating point precision? fp16 vs fp32)
# we used Tesla T4 all the neural networks (except for the one Residual Model, where it is specified otherwise) and CPU for their evaluation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dir_path = os.getcwd()


def hyperparam_tune(model_type, joint_dim, disjoint_dim, lr, batch, drop, train_data, val_loader, path):
    set_seed()
    train_loader = DataLoader(train_data, batch_size = batch, shuffle = True)
    epochs = 200
    res = []

    set_seed()
    net = model_type(61, joint_dim, disjoint_dim, mse_loss, bce_loss, drop, weighted = False).to(device)
    net.optimizer = optim.Adam(net.parameters(), lr=lr)

    net_history = fit_mt(path = path, model = net, train_loader = train_loader, epochs = epochs, val_loader = val_loader)

    net_history.to_csv(dir_path + '/tuning_results/hyperparam_tune/training_history/%s_%s/%d_%d_%.4f_%d_%.1f.csv' % 
                     (net.__class__.__name__, net.weighted, joint_dim, disjoint_dim, lr, batch, drop))

    net_max_auc = max(history['val_auc'])
    net_max_auc_loss = history['val_loss'][history['val_auc'].argmax()]
    net_max_auc_epoch = history['epoch'][history['val_auc'].argmax()]
    net_max_auc_bv_loss = history['basket_value_val_loss'][history['val_auc'].argmax()]
    net_min_bv_loss = min(history['basket_value_val_loss'])
    net_min_bv_epoch = history['epoch'][history['basket_value_val_loss'].argmin()]
    net_min_bv_auc = history['val_auc'][history['basket_value_val_loss'].argmin()]
    net_min_val_bv_yloss = history['val_loss'][history['basket_value_val_loss'].argmin()]

    res.append([net.__class__.__name__, net.weighted, joint_dim, disjoint_dim, lr, batch, drop, 
              net_max_auc_epoch, net_max_auc, net_max_auc_loss, net_max_auc_bv_loss, net_min_bv_loss,
              net_min_bv_epoch, net_min_bv_auc, net_min_val_bv_yloss])

    set_seed()
    wnet = model_type(61, joint_dim, disjoint_dim, mse_loss, bce_loss, drop, weighted = True).to(device)
    wnet.optimizer = optim.Adam(wnet.parameters(), lr=lr)
    wnet_history = fit_mt(path = path, model = wnet, train_loader = train_loader, epochs = epochs, val_loader = val_loader)

    wnet_history.to_csv(dir_path + '/tuning_results/hyperparam_tune/training_history/%s_%s/%d_%d_%.4f_%d_%.1f.csv' % 
                      (wnet.__class__.__name__, wnet.weighted, joint_dim, disjoint_dim, lr, batch, drop))

    wnet_max_auc = max(history['val_auc'])
    wnet_max_auc_loss = history['val_loss'][history['val_auc'].argmax()]
    wnet_max_auc_epoch = history['epoch'][history['val_auc'].argmax()]
    wnet_max_auc_bv_loss = history['basket_value_val_loss'][history['val_auc'].argmax()]
    wnet_min_bv_loss = min(history['basket_value_val_loss'])
    wnet_min_bv_epoch = history['epoch'][history['basket_value_val_loss'].argmin()]
    wnet_min_bv_auc = history['val_auc'][history['basket_value_val_loss'].argmin()]
    wnet_min_val_bv_yloss = history['val_loss'][history['basket_value_val_loss'].argmin()]

    res.append([wnet.__class__.__name__, wnet.weighted, joint_dim, disjoint_dim, lr, batch, drop, 
              wnet_max_auc_epoch, wnet_max_auc, wnet_max_auc_loss, wnet_max_auc_bv_loss, wnet_min_bv_loss, 
              wnet_min_bv_epoch, wnet_min_bv_auc, wnet_min_bv_yloss])

    return res


def simple_nn_tune(net_type, train_data, val_data, criterion, hidden, lr, batch, epochs, path, th):
    # hyperparameter tuning for baseline neural network models

    set_seed()

    # define train and validation data
    train_loader = DataLoader(train_data, batch_size = batch, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = len(val_data))
    res = []

    set_seed()
    # create neural network model using the given hyperparameters
    net = net_type(61, hidden_layers = hidden, criterion = criterion).to(device)
    net.optimizer = optim.Adam(net.parameters(), lr=lr)

    # model fit results
    best_metric_value, best_metric_value_epoch,_ = net.fit(train_loader=train_loader, epochs=epochs, 
                                                         val_loader=val_loader, path=path, threshold=th)
    res.append([net.__class__.__name__, hidden, lr, batch, best_metric_value, best_metric_value_epoch])

    return res


# Training neural nets with multiple learning rates
def separate_lr_tune(model, train_loader, val_loader, joint_lr, reg_lr, class_lr, epochs, path):

    net = model
    # Sepcify the optimization algorithm and learning rates for each part of the NN
    net.optimizer = optim.Adam([{'params': net.joint.parameters(), 'lr': joint_lr},
                              {'params': net.disjoint_c.parameters(), 'lr': class_lr},
                              {'params': net.output_c.parameters(), 'lr': class_lr},
                              {'params': net.disjoint_r.parameters(), 'lr': reg_lr},
                              {'params': net.output_r.parameters(), 'lr': reg_lr}])

    # Train the network
    (net_max_auc, net_max_auc_loss, net_max_auc_epoch, net_max_auc_bv_loss, net_min_bv_loss, 
    net_min_bv_epoch, net_min_bv_auc, net_min_bv_yloss, net_history) = fit_mt(path = path, model = net,
                                                                         train_loader = train_loader,
                                                                         epochs = epochs,
                                                                         val_loader = val_loader)
    # Save training history
    net_history.to_csv(dir_path + '/tuning_results/%s/training_history/%s_%s/%.4f_%.4f_%.4f.csv' % 
                      (path, net.__class__.__name__, net.weighted, joint_lr, class_lr, reg_lr))

    res = [net.__class__.__name__, net.weighted, net.joint_layer_size, net.disjoint_layer_size, net.drop,
         joint_lr, class_lr, reg_lr, net_max_auc_epoch, net_max_auc, net_max_auc_loss,
         net_max_auc_bv_loss, net_min_bv_loss, net_min_bv_epoch, net_min_bv_auc, net_min_bv_yloss]

    return res

def mt_residual_tune(ite_joint, ite_disjoint, drop, epochs, temp, path):

    results = []
    set_seed()
    mt_ite_model = MTIte(61, ite_joint, ite_disjoint, mse_loss, drop).to(device)

    set_seed()
    mt_residual_model_cutoff = MTResidualModel(mt_daw_model, mt_ite_model, mse_loss, bce_loss, sigm = False).to(device)
    mt_residual_model_cutoff.optimizer = optim.Adam(mt_residual_model_cutoff.parameters(), lr=0.001)

    (net_max_auc, net_max_auc_loss, net_max_auc_epoch, net_max_auc_bv_loss, net_min_bv_loss, net_min_bv_epoch,
    net_min_bv_auc, net_min_bv_total_loss, net_history) = daw_fit(path = path, model = mt_residual_model_cutoff, 
                                                                 train_loader = res_train_loader, epochs = epochs, 
                                                                 val_loader = res_val_loader, temp = temp,
                                                                 th_auc=0.64, th_loss=771, th_stop=19)
    net_history.to_csv(dir_path + '/tuning_results//%s/training_history/%s_%s_%d_%d_%.1f.csv' % 
                     (path, mt_residual_model_cutoff.__class__.__name__, mt_residual_model_cutoff.sigm,
                      ite_joint, ite_disjoint, drop))
    results.append([mt_residual_model_cutoff.__class__.__name__, mt_residual_model_cutoff.sigm, ite_joint,
                  ite_disjoint, net_max_auc_epoch, net_max_auc, net_max_auc_loss, net_max_auc_bv_loss, 
                  net_min_bv_loss, net_min_bv_epoch, net_min_bv_auc, net_min_bv_total_loss])

    set_seed()
    mt_residual_model_sigm = MTResidualModel(mt_daw_model, mt_ite_model, mse_loss, bce_loss, sigm = True).to(device)
    mt_residual_model_sigm.optimizer = optim.Adam(mt_residual_model_sigm.parameters(), lr=0.001)

    (net_max_auc, net_max_auc_loss, net_max_auc_epoch, net_max_auc_bv_loss, net_min_bv_loss, net_min_bv_epoch,
    net_min_bv_auc, net_min_bv_total_loss, net_history) = daw_fit(path = path, model = mt_residual_model_sigm,
                                                                 train_loader = res_train_loader, epochs = epochs,
                                                                 val_loader = res_val_loader, temp = temp,
                                                                 th_auc=0.64, th_loss=771, th_stop=19)
    net_history.to_csv(dir_path + '/tuning_results//%s/training_history/%s_%s_%d_%d_%.1f.csv' % 
                     (path, mt_residual_model_sigm.__class__.__name__, mt_residual_model_sigm.sigm,
                      ite_joint, ite_disjoint, drop))
    results.append([mt_residual_model_sigm.__class__.__name__, mt_residual_model_sigm.sigm, ite_joint, 
                  ite_disjoint, net_max_auc_epoch, net_max_auc, net_max_auc_loss, net_max_auc_bv_loss, 
                  net_min_bv_loss, net_min_bv_epoch, net_min_bv_auc, net_min_bv_total_loss])

    return results


def daw_tune(joint_dim, disjoint_dim, lr, batch, drop, temp, path, th_auc, th_loss, th_stop, train_data, val_loader):

    set_seed()
    train_loader = DataLoader(train_data, batch_size = batch, shuffle = True)
    epochs = 200
    res = []

    set_seed()
    net = DAW_MTXNet(61, joint_dim, disjoint_dim, mse_loss, bce_loss, drop, weighted = False).to(device)
    net.optimizer = optim.Adam(net.parameters(), lr=lr)

    (net_max_auc, net_max_auc_loss, net_max_auc_epoch, net_max_auc_bv_loss, net_min_bv_loss, 
    net_min_bv_epoch, net_min_bv_auc, net_min_bv_total_loss, net_history) = daw_fit(path = path, model = net, 
                                                                                train_loader = train_loader, 
                                                                                epochs = epochs, 
                                                                                val_loader = val_loader, 
                                                                                temp = temp,
                                                                                th_auc = th_auc,
                                                                                th_loss = th_loss,
                                                                                th_stop = th_stop)
    net_history.to_csv(dir_path + '/tuning_results/%s/training_history/%s_%s/%d_%d_%.4f_%d_%.1f_%.2f.csv' % 
                     (path, net.__class__.__name__, net.weighted, joint_dim, disjoint_dim, lr, batch, drop, temp))
    res.append([net.__class__.__name__, net.weighted, joint_dim, disjoint_dim, lr, batch, drop, temp, 
              net_max_auc_epoch, net_max_auc, net_max_auc_loss, net_max_auc_bv_loss, net_min_bv_loss, 
              net_min_bv_epoch, net_min_bv_auc, net_min_bv_total_loss])

    set_seed()
    net = DAW_MTXNet(61, joint_dim, disjoint_dim, mse_loss, bce_loss, drop, weighted = True).to(device)
    net.optimizer = optim.Adam(net.parameters(), lr=lr)

    (net_max_auc, net_max_auc_loss, net_max_auc_epoch, net_max_auc_bv_loss, net_min_bv_loss,
    net_min_bv_epoch, net_min_bv_auc, net_min_bv_total_loss, net_history) = daw_fit(path = path, model = net,
                                                                                train_loader = train_loader,
                                                                                epochs = epochs,
                                                                                val_loader = val_loader,
                                                                                temp = temp, th_auc=th_auc,
                                                                                th_loss=th_loss, th_stop=th_stop)

    net_history.to_csv(dir_path + '/tuning_results/%s/training_history/%s_%s/%d_%d_%.4f_%d_%.1f_%.2f.csv' % 
                     (path, net.__class__.__name__, net.weighted, joint_dim, disjoint_dim, lr, batch, drop, temp))
    res.append([net.__class__.__name__, net.weighted, joint_dim, disjoint_dim, lr, batch, drop, temp,
              net_max_auc_epoch,net_max_auc, net_max_auc_loss, net_max_auc_bv_loss, net_min_bv_loss,
              net_min_bv_epoch, net_min_bv_auc, net_min_bv_total_loss])

    return res


def response_residual_tune(mt_model, hidden_layers, path):

    set_seed()
    ite_model = IteNN(61, hidden_layers, mse_loss)
    set_seed()
    net = ResponseResidualModel(mt_model, ite_model, mse_loss, bce_loss).to(device)
    net.optimizer = optim.Adam(net.parameters(), lr=0.001)

    (net_max_auc, net_max_auc_loss, net_max_auc_epoch, net_max_auc_bv_loss, net_min_bv_loss,
    net_min_bv_epoch, net_min_bv_auc, net_min_bv_total_loss, net_history) = net.fit(path = path,
                                                                                   train_loader = res_train_loader,
                                                                                   epochs = 150, 
                                                                                   val_loader = res_val_loader, 
                                                                                   temp = 0.05)
    net_history.to_csv(dir_path + '/tuning_results/%s/training_history/%s_%s.csv' % 
                     (path, net.__class__.__name__, hidden_layers))
    res = [net.__class__.__name__, net.weighted, hidden_layers, net_max_auc_epoch, net_max_auc, net_max_auc_loss,
         net_max_auc_bv_loss, net_min_bv_loss, net_min_bv_epoch, net_min_bv_auc, net_min_bv_total_loss]

    return res