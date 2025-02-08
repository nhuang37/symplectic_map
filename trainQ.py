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

from model import HenonNetsupQ
from utils import generate_split, to_standard, from_standard

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--tie_weights', action='store_true', help='if used, then tie weights in the HenonLayer across 4 Henon Maps')
    parser.add_argument('--Q_weight', type=float, default=1.0, help='regularization stength of the Q prediction')
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--filename', type=str, default='data_1D_dim=2.pkl')
    parser.add_argument('--save_path', type=str, default='results_supQ/')

    args = parser.parse_args()
    #save path
    outdir = f'{args.save_path}_layer={args.num_layers}_hid={args.hid_dim}_lr={args.lr}_tie={args.tie_weights}_ep={args.num_epochs}_bs={args.batch_size}'
    os.makedirs(outdir, exist_ok=True)

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
    Q = torch.from_numpy(data_dict['Q']).float().unsqueeze(1)
    XV_out_Q = torch.concatenate((XV_out, Q), dim=-1)
    num_samples = XV.shape[0]
    train_indices, val_indices, test_indices = data_dict['splits']

    train_dataset = TensorDataset(XV[train_indices, :], XV_out_Q[train_indices, :])
    val_dataset = TensorDataset(XV[val_indices, :], XV_out_Q[val_indices, :])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    #initialize model and optim
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HenonNetsupQ(input_dim=1, hid_dim=args.hid_dim, num_layers=args.num_layers, tie_weights=args.tie_weights)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


    #training
    print("Training model...")
    loss_history, loss_hist_xv, loss_hist_Q = [], [], []
    for epoch in range(args.num_epochs):  
        model.train()
        for step, (xv, xv_out_Q) in enumerate(train_loader):
            xv = xv.to(device)
            xv_out_Q = xv_out_Q.to(device)
            x_pred, v_pred, Q_pred = model(xv[:,0:1], xv[:,1:2])
            loss_xv = F.mse_loss(torch.concatenate([x_pred,v_pred],axis=-1), xv_out_Q[:,:2])
            loss_Q = F.mse_loss(Q_pred, xv_out_Q[:, 2:])
            loss = loss_xv + args.Q_weight * loss_Q
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0) ##seems important for numerical stability? 
            optimizer.step()
            optimizer.zero_grad()
            loss_history.append(loss.item())
            loss_hist_xv.append(loss_xv.item())
            loss_hist_Q.append(loss_Q.item())
        print(f'epoch = {epoch+1}, step={step}, loss = {loss.item():.4f}, loss_xv = {loss_xv.item():.4f}, loss_Q = {loss_Q.item():.4f}')
        if (epoch+1) % args.eval_every == 0:
            model.eval()
            loss_all, loss_xv, loss_Q = 0, 0, 0
            for (xv, xv_out_Q) in val_loader:
                xv = xv.to(device)
                xv_out_Q = xv_out_Q.to(device)
                x_pred, v_pred, Q_pred = model(xv[:,0:1], xv[:,1:2])
                loss_xv += F.mse_loss(torch.concatenate([x_pred,v_pred],axis=-1), xv_out_Q[:,:2]).item()
                loss_Q += F.mse_loss(Q_pred,  xv_out_Q[:,2:])
                loss_all += loss_xv + args.Q_weight * loss_Q
            print(f'epoch = {epoch+1}, validation_loss = {loss_all:.4f}, val_xv = {loss_xv:.4f}, val_Q = {loss_Q:.4f}')
    
    #save training loss, model weights
    save_file = {'train_loss': loss_history, 'loss_xv': loss_hist_xv, 'loss_Q': loss_hist_Q,
                 'data_mean_std': (XVout_mean, XVout_std),
                 'x_pred': x_pred.detach().cpu().numpy(), 'v_pred': v_pred.detach().cpu().numpy(),
                 'Q_pred': Q_pred.detach().cpu().numpy()}
    pickle.dump(save_file, open(f"{outdir}/save_file.pkl","wb"))
    torch.save(model.state_dict(), f"{outdir}/model.pth")
    







