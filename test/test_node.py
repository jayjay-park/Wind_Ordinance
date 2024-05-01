import torch
import torch.nn as nn
import torch.autograd.functional as F
import torch.optim as optim
import torchdiffeq
import datetime
import numpy as np
import argparse
import json
import logging
import os
import csv
import math
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split


device = 'cuda' if torch.cuda.is_available() else 'cpu'



class ODE_MLP(nn.Module):
    '''Define Neural Network that approximates differential equation system of Chaotic Lorenz'''

    def __init__(self, y_dim=3, n_hidden=512, n_layers=2):
        super(ODE_MLP, self).__init__()
        layers = [nn.Linear(y_dim, n_hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(n_hidden, n_hidden), nn.GELU()])
        layers.append(nn.Linear(n_hidden, y_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, y):
        res = self.net(y)
        return res

class ODEBlock(nn.Module):
  
    '''
    Code credit: https://www.kaggle.com/code/shivanshuman/learning-physics-with-pytorch
    '''

    def __init__(self, f):
        super(ODEBlock, self).__init__()
        self.f = f
        self.integration_time = torch.Tensor([0,1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = torchdiffeq.odeint(self.f, x, self.integration_time)
        return out[1]


class ODENet(nn.Module):
  
    '''
    Code credit: https://www.kaggle.com/code/shivanshuman/learning-physics-with-pytorch
    '''

    def __init__(self, in_dim, mid_dim, n_layers, out_dim):
        super(ODENet, self).__init__()
        fx = ODE_MLP(y_dim=in_dim, n_hidden=mid_dim, n_layers=n_layers)
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm1d(mid_dim)
        self.ode_block = ODEBlock(fx)
        self.dropout = nn.Dropout(0.4)
        self.norm2 = nn.BatchNorm1d(mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        # out = self.fc1(x)
        # out = self.relu1(out)
        # out = self.norm1(out)
        out = self.ode_block(x)
        out = self.fc1(out)
        # out = self.norm2(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


##############
## Training ##
##############


def update_lr(optimizer, epoch, total_e, origin_lr):
    """ A decay factor of 0.1 raised to the power of epoch / total_epochs. Learning rate decreases gradually as the epoch number increases towards the total number of epochs. """
    new_lr = origin_lr * (0.1 ** (epoch / float(total_e)))
    for params in optimizer.param_groups:
        params['lr'] = new_lr
    return

def train_node(dyn_sys_info, model, device, data_loader, criterion, epochs, lr, weight_decay):

    # Initialize
    n_store, k  = 100, 0
    ep_num, loss_hist, test_loss_hist = torch.empty(n_store+1,dtype=int), torch.empty(n_store+1), torch.empty(n_store+1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader, val_loader, test_loader = data_loader
    dim, time_step = dyn_sys_info
    t_eval_point = torch.linspace(0, time_step, 2).to(device)
    torch.cuda.empty_cache()
    all_treatment = []
    
    # Training Loop
    min_relative_error = 1000000
    for i in range(epochs):
        model.train()
        full_loss = 0.0
        for X_train, Y_train in train_loader:
            X_train, Y_train = X_train.cuda(), Y_train.cuda()
            y_pred = model(X_train)
            y_pred = y_pred.to(device)
    
            optimizer.zero_grad()
            train_loss = criterion(y_pred.squeeze(), Y_train)
            train_loss.backward()
            optimizer.step()
            full_loss += train_loss.item()

        print(i, full_loss)
        avg_train_loss = full_loss / len(train_loader)
        update_lr(optimizer, i, epochs, args.lr)

        # Save Training and Test History
        if i % (epochs//n_store) == 0 or (i == epochs-1):
            with torch.no_grad():
                model.eval()
                full_test_loss = 0.0
                full_val_loss = 0.0

                for X_val, Y_val in val_loader:
                    X_val, Y_val = X_val.cuda(), Y_val.cuda()
                    y_pred_val = model(X_val)
                    y_pred_val = y_pred_val.to(device)         
                    val_loss = criterion(y_pred_val.squeeze(), Y_val)
                    full_val_loss += val_loss.item()
                avg_val_loss = full_val_loss / len(val_loader)

                if full_val_loss < min_relative_error:
                    min_relative_error = full_val_loss
                    # Save the model
                    torch.save(model.state_dict(), f"../test_result/best_model.pth")
                    logger.info(f"Epoch {i}: New minimal validation loss: {min_relative_error:.2f}%, model saved.")

                    for X_test, Y_test in test_loader:
                        X_test, Y_test = X_test.cuda(), Y_test.cuda()
                        y_pred_test = model(X_test)
                        y_pred_test = y_pred_test.to(device)         
                        test_loss = criterion(y_pred_test.squeeze(), Y_test)
                        treatment = y_pred_test - Y_test
                        csv_file_path = 'treatment.csv'
                        with open(csv_file_path, mode='w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(treatment)

                        print("std of treatment", torch.std(treatment))
                        print("mean of treatment", torch.mean(treatment))
                        print("mean of pred_test", torch.mean(y_pred_test))
                        print("mean of true_test", torch.mean(Y_test))
                        logger.info("%s: %s", "std of treatment", str(torch.std(treatment)))
                        logger.info("%s: %s", "mean of treatment", str(torch.mean(treatment)))
                        logger.info("%s: %s", "mean of pred_test", str(y_pred_test))
                        logger.info("%s: %s", "mean of true_test", str(torch.mean(Y_test)))

                        full_test_loss += test_loss.item()
                    avg_test_loss = full_test_loss / len(test_loader)

                logger.info("Epoch: %d Train: %.5f Val: %.5f Test: %.5f", i, avg_train_loss, avg_val_loss, avg_test_loss)
                print("Epoch: ", i, " Train: {:.5f}".format(avg_train_loss), " Val: {:.5f}".format(avg_val_loss))
                ep_num[k], loss_hist[k], test_loss_hist[k] = i, avg_train_loss, avg_val_loss

                k = k + 1

    return ep_num, loss_hist, test_loss_hist, Y_test



##############
#### Plot ####
##############

def plot_loss(epochs, train, test, path):
    fig, ax = subplots()
    ax.plot(epochs[0:].numpy(), train[0:].detach().cpu().numpy(), "P-", lw=2.0, ms=5.0, label="Train")
    ax.plot(epochs[0:].numpy(), test[0:].detach().cpu().numpy(), "P-", lw=2.0, ms=5.0, label="Val")
    ax.set_xlabel("Epochs",fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.legend(fontsize=24)
    ax.grid(True)
    tight_layout()
    savefig(path, bbox_inches ='tight', pad_inches = 0.1)


def rk4(x, f, dt):
    k1 = f(0, x)
    k2 = f(0, x + dt*k1/2)
    k3 = f(0, x + dt*k2/2)
    k4 = f(0, x + dt*k3)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)



if __name__ == '__main__':

    # Set device
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=1.)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--model_type", default="MLP", choices=["MLP"])
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--optim_name", default="AdamW", choices=["AdamW", "Adam", "RMSprop", "SGD"])

    # Initialize Settings
    args = parser.parse_args()
    dim = args.dim
    dyn_sys_info = [dim, args.time_step]
    criterion = torch.nn.MSELoss()

    # Save initial settings
    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"{start_time}.txt")
    logging.basicConfig(filename=out_file, level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)

    # Create Dataset
    # Label Encoding
    le = preprocessing.LabelEncoder()
    df = pd.read_csv('../data/merged_clevel_data.csv')
    df[['t_state', 't_county']] = df[['t_state', 't_county']].astype(str).apply(le.fit_transform)

    # Select Covariates among: t_state, t_county, p_tnum, t_hh, t_rd, t_rsa, t_ttlh, xlong, ylat
    df = df[['t_county', 't_state', 'p_tnum', 'ordinance', 'p_cap', 't_rsa']]
    logger.info("%s", str(df.columns))

    # Divide the dataframe to train and test dataset
    train = df[df['ordinance'] == 0]
    test = df[df['ordinance'] == 1]

    # Divide X and y
    train_y = train.pop('p_cap')
    test_y = test.pop('p_cap')
    print(train_y.shape, test_y.shape)

    train = train.to_numpy(dtype='float32')
    test = test.to_numpy(dtype='float32')
    train_y = train_y.to_numpy(dtype='float32')
    test_y = test_y.to_numpy(dtype='float32')
    train = torch.tensor(train, dtype=torch.float32)
    test = torch.tensor(test, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32) 
    print("X dim: ", train.shape)
    print("y dim: ", train_y.shape)

    # Train-Val split
    dataset = TensorDataset(train, train_y)
    test_dataset = TensorDataset(test, test_y)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    data_loader = [train_loader, val_loader, test_loader]

    # Create model
    if args.model_type == "MLP":
        m = ODENet(in_dim=dim, mid_dim=args.n_hidden, n_layers=args.n_layers, out_dim=1).to(device)

    print("Training...") # Train the model, return node
    epochs, loss_hist, test_loss_hist, Y_test = train_node(dyn_sys_info, m, device, data_loader, criterion, args.num_epoch, args.lr, args.weight_decay)

    # Plot Loss
    loss_path = f"../plot/Loss/{start_time}.png"

    plot_loss(epochs, loss_hist, test_loss_hist, loss_path)
    logger.info("%s: %s", "Training Loss", str(loss_hist[-1]))
    logger.info("%s: %s", "Test Loss", str(test_loss_hist[-1]))
