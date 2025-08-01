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
    parser.add_argument('--num_layers', type=int, default=8) #[2,4,8,12] seems 8 is the best with hid_dim=4
    parser.add_argument('--hid_dim', type=int, default=4)
    parser.add_argument('--tie_weights', action='store_true', help='if used, then tie weights in the HenonLayer across 4 Henon Maps')
    parser.add_argument('--epsilon', type=float, default=1.0)

    parser.add_argument('--num_epochs', type=int, default=100) #100-500 converging
    parser.add_argument('--lr', type=float, default=1e-3) #1e-3 seems better than 1e-2
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--filename', type=str, default='clean_data_1D_dim=2.pkl')
    parser.add_argument('--save_path', type=str, default='results_clean_search/')
    parser.add_argument('--relativeMSE', action='store_true', help='relative MSE') 
    parser.add_argument('--weighted', action='store_true', help='using weighted MSE') 
    parser.add_argument('--weight_scale', type=float, default=5.0)
    parser.add_argument('--weight_velocity', type=float, default=1.0)

    parser.add_argument('--train_ratio', type=float, default=1.0)


    args = parser.parse_args()
    #save path
    outdir = f'{args.save_path}_layer={args.num_layers}_hid={args.hid_dim}_lr={args.lr}_weightMSE={args.weighted}'
    os.makedirs(outdir, exist_ok=True)

    #load data, generate split, and construct data loaders
    data_dict = pickle.load(open(args.filename, "rb"))
    XV, XV_mean, XV_std = load_variables(data_dict['X'], data_dict['V'])
    XV_out, XVout_mean, XVout_std = load_variables(data_dict['X_o'], data_dict['V_o'], standard=False)
    # input_max, _ = torch.max(XV, dim=0) 
    # output_max, _ = torch.max(XV_out, dim=0)
    # print(f'Input range: {input_max}, Output range:{output_max}')
    # print(f'standardizing both input and output...')
    # XV /= input_max #(2)
    # XV_out /= input_max
    # print(f'input range: {XV.max()}, output range: {XV_out.max()}')
    num_samples = XV.shape[0]
    train_indices, val_indices, test_indices = data_dict['splits']
    if args.train_ratio < 1.0:
        n_train = len(train_indices)
        sub_indices = np.random.choice(n_train, int(n_train*args.train_ratio), replace=False)
        train_indices = train_indices[sub_indices]

    train_dataset = TensorDataset(XV[train_indices, :], XV_out[train_indices, :])
    val_dataset = TensorDataset(XV[val_indices, :], XV_out[val_indices, :])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False) #full batch 

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
    loss_best = 1e9
    model_best = None
    for epoch in range(args.num_epochs):  
        model.train()
        for step, (xv, xv_out) in enumerate(train_loader):
            xv = xv.to(device)
            xv_out = xv_out.to(device)
            x_pred, v_pred = model(xv[:,0:1], xv[:,1:2])
            if args.weighted:
                weight = 0.5*(xv[:,0:1]**2 + xv[:,1:2]**2)  ##effectively Jhat
                weight = args.weight_scale * weight.repeat(1,2) #match input shape
            else:
                weight = torch.ones((xv.shape)).to(xv.device)
                weight[:,-1] *= args.weight_velocity #default: unweighted
            loss = loss_fn(torch.concatenate([x_pred,v_pred],axis=-1), xv_out, weight=weight)
            optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_history.append(loss.item())

        #eval
        model.eval()
        loss_all = 0
        xv, xv_out = next(iter(val_loader)) #full batch!
        xv = xv.to(device)
        xv_out = xv_out.to(device)
        x_pred, v_pred = model(xv[:,0:1], xv[:,1:2])
        if args.weighted:
            weight = 0.5*(xv[:,0:1]**2 + xv[:,1:2]**2)  ##effectively Jhat
            weight = args.weight_scale * weight.repeat(1,2) #match input shape #TODO: upweight the velocity part? it's learning a bit worse than the position
        else:
            weight = torch.ones((xv.shape)).to(xv.device)
            weight[:,-1] *= args.weight_velocity #default: unweighted
        loss_cur = loss_fn(torch.concatenate([x_pred,v_pred],axis=-1), xv_out, weight=weight).item()
        if loss_cur < loss_best:
            loss_best = loss_cur
            model_best = model 
        print(f'epoch = {epoch+1}, train_loss = {loss.item():.6f}, val_loss = {loss_cur:.6f}, best_val_loss = {loss_best:.6f}')
    
    #save training loss, model weights
    save_file = {'train_loss': loss_history, 'input_mean_std': (XV_mean, XV_std),
                 'output_mean_std': (XVout_mean, XVout_std), #'input_max': input_max,
                 'x_pred': x_pred.detach().cpu(), 'v_pred': v_pred.detach().cpu()}
    pickle.dump(save_file, open(f"{outdir}/save_file.pkl","wb"))
    torch.save(model_best.state_dict(), f"{outdir}/model.pth")
    







