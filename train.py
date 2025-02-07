import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

import argparse 
import pathlib
import os
import random
import pickle
import time
import numpy as np
import sys

from model import HenonNet
from utils import generate_split, to_standard, from_standard

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hid_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--filename', type=str, default='data_1D_dim=2.pkl')
    args = parser.parse_args()

    #load data, generate split, and construct data loaders
    data_dict = pickle.load(open(args.filename, "rb"))
    X = data_dict['X']
    V = data_dict['V']
    XV, XV_mean, XV_std = to_standard(np.stack((X,V), axis=-1))
    XV = torch.from_numpy(XV).float() #(num_samples, 2)
    X_o = data_dict['X_o']
    V_o = data_dict['V_o']
    XV_out, XVout_mean, XVout_std = to_standard(np.stack((X_o,V_o), axis=-1))
    XV_out = torch.from_numpy(XV_out).float() #(num_samples, 2)
    num_samples = XV.shape[0]
    train_indices, val_indices, test_indices = generate_split(num_samples)

    train_dataset = TensorDataset(XV[train_indices, :], XV_out[train_indices, :])
    val_dataset = TensorDataset(XV[val_indices, :], XV_out[val_indices, :])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10*args.batch_size, shuffle=False)

    #initialize model and optim
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HenonNet(input_dim=1, hid_dim=args.hid_dim, num_layers=args.num_layers)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    #training
    print("Training model...")
    loss_history = []
    for epoch in range(args.num_epochs):  
        for step, (xv, xv_out) in enumerate(train_loader):
            xv = xv.to(device)
            xv_out = xv_out.to(device)
            x_pred, v_pred = model(xv[:,0:1], xv[:,1:2])
            loss = F.mse_loss(torch.concatenate([x_pred,v_pred],axis=-1), xv_out)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            loss_history.append(loss.item())
        print(f'epoch = {epoch}, step={step}, loss = {loss.item()}')
    pickle.dump(loss_history, open(f'train_loss_layer={args.num_layers}_hid={args.hid_dim}_lr={args.lr}.pkl',"wb"))
    







