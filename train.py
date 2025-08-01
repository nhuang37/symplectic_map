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
from utils import generate_split, to_standard, from_standard, to_volume_preserving_standard, load_variables

def relative_MSE(yhat, y, weight=None):
    '''
    yhat/y: (n, d) shape
    '''
    ynorm = torch.linalg.norm(y, dim=1) #(n)
    if weight is None:
        SE = torch.sum((yhat - y)**2, dim=1) #(n)
    else:
        assert weight.shape[0] == y.shape[0]
        SE = torch.sum(((yhat - y)**2)**weight, dim=1) 
    RSE = SE / ynorm #(n,2)
    return torch.mean(RSE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--tie_weights', action='store_true', help='if used, then tie weights in the HenonLayer across 4 Henon Maps')
    parser.add_argument('--epsilon', type=float, default=0.01)

    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--filename', type=str, default='data_1D_dim=2.pkl')
    parser.add_argument('--save_path', type=str, default='results/')
    parser.add_argument('--swap', action='store_false', 
                        help='default true: x_o=\sqrt{2J} cos\theta; v_o = \sqrt{2J} sin\theta; \
                            if false, swap to x_o=\sqrt{2J} sin\theta; v_o = \sqrt{2J} cos\theta')
    parser.add_argument('--relativeMSE', action='store_true', help='relative MSE') 

    args = parser.parse_args()
    #save path
    outdir = f'{args.save_path}_layer={args.num_layers}_hid={args.hid_dim}_lr={args.lr}_tie={args.tie_weights}_ep={args.num_epochs}_epsilon={args.epsilon}_relMSE={args.relativeMSE}'
    os.makedirs(outdir, exist_ok=True)

    #load data, generate split, and construct data loaders
    data_dict = pickle.load(open(args.filename, "rb"))
    XV, XV_mean, XV_std = load_variables(data_dict['X'], data_dict['V'])
    if args.swap: #swap the output coordinates for easier fit
        XV_out, XVout_mean, XVout_std = load_variables(data_dict['V_o'], data_dict['X_o'], standard=False)
    else:
        XV_out, XVout_mean, XVout_std = load_variables(data_dict['X_o'], data_dict['V_o'], standard=False)
    input_max, _ = torch.max(XV, dim=0) 
    output_max, _ = torch.max(XV_out, dim=0)
    print(f'Input range: {input_max}, Output range:{output_max}')
    print(f'standardizing both input and output...')
    XV /= input_max #(2)
    XV_out /= input_max
    print(f'input range: {XV.max()}, output range: {XV_out.max()}')
    num_samples = XV.shape[0]
    train_indices, val_indices, test_indices = data_dict['splits']

    train_dataset = TensorDataset(XV[train_indices, :], XV_out[train_indices, :])
    val_dataset = TensorDataset(XV[val_indices, :], XV_out[val_indices, :])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    #initialize model and optim
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HenonNet(input_dim=1, hid_dim=args.hid_dim, num_layers=args.num_layers, 
                     tie_weights=args.tie_weights, epsilon=args.epsilon)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.relativeMSE:
        loss_fn = relative_MSE
    else:
        loss_fn = F.mse_loss


    #training
    print("Training model...")
    loss_history = []
    for epoch in range(args.num_epochs):  
        model.train()
        for step, (xv, xv_out) in enumerate(train_loader):
            xv = xv.to(device)
            xv_out = xv_out.to(device)
            x_pred, v_pred = model(xv[:,0:1], xv[:,1:2])
            weight = 1/0.5*(xv[:,0:1]**2 + xv[:,1:2]**2)  ##effectively Jhat
            weight = weight.repeat(1,2) #match input shape
            loss = loss_fn(torch.concatenate([x_pred,v_pred],axis=-1), xv_out, weight=weight)
            optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_history.append(loss.item())
        print(f'epoch = {epoch+1}, step={step}, loss = {loss.item():.4f}')
        if (epoch+1) % args.eval_every == 0:
            model.eval()
            loss_all = 0
            for (xv, xv_out) in val_loader:
                xv = xv.to(device)
                xv_out = xv_out.to(device)
                x_pred, v_pred = model(xv[:,0:1], xv[:,1:2])
                weight = 1/0.5*(xv[:,0:1]**2 + xv[:,1:2]**2)  ##effectively Jhat
                weight = weight.repeat(1,2) #match input shape
                loss_all += loss_fn(torch.concatenate([x_pred,v_pred],axis=-1), xv_out, weight=weight).item()
            print(f'epoch = {epoch+1}, validation_loss = {loss_all:.4f}')
    
    #save training loss, model weights
    save_file = {'train_loss': loss_history, 'input_mean_std': (XV_mean, XV_std),
                 'output_mean_std': (XVout_mean, XVout_std), 'input_max': input_max,
                 'x_pred': x_pred.detach().cpu(), 'v_pred': v_pred.detach().cpu()}
    pickle.dump(save_file, open(f"{outdir}/save_file.pkl","wb"))
    torch.save(model.state_dict(), f"{outdir}/model.pth")
    







