import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn

#src: https://github.com/vduruiss/SymplecticGyroceptron/blob/main/NPMap_Learning.py
#streamline into Pytorch implementation

def HenonMap(X, Y, W_in, W_out, b_in, eta, epsilon=1):
    '''
    X, Y: original coordinates (assume X, Y have the same shape)
    W_in, W_out, b_in: define the 2-layer neural network V(Y) \in \R (output a scalar)
    X_out, Y_out: transformed coordinates, where
    - X_out = Y + eta 
    - Y_out = -X + \epsilon dV(Y)/dY, for V a two-layer MLP
    epsilon: default to 1 (regular Henon Map in https://arxiv.org/pdf/2007.04496)
    '''
    assert X.shape == Y.shape
    Y.requires_grad_()  # Track gradients for Y 
    V = torch.matmul(torch.tanh(torch.matmul(Y, W_in) + b_in), W_out) #tanh seems to explode gradient to loss nan
    
    X_out = Y + eta
    V_grad = torch.autograd.grad(V.sum(), Y, create_graph=True)[0]  # Compute the gradient of V with respect to Y
    
    Y_out = -X + epsilon * V_grad
    return X_out, Y_out

class HenonLayer(nn.Module): 
    def __init__(self, input_dim, hid_dim, epsilon=1, num_maps=4, tie_weights=False, xavier_init=False):
        super().__init__()
        self.input_dim = input_dim  #dimension of X, Y
        self.hid_dim = hid_dim #dimension of X_out, Y_out
        self.epsilon = epsilon
        self.num_maps = num_maps
        self.tie_weights = tie_weights #to share the weights across HenonMaps
        if self.tie_weights:
            self.num_maps = 1
        
        # Initialize weights: shared across 4 HenonMaps (break the weight sharing to see if it helps with nana)
        self.W_in = nn.Parameter(torch.rand(self.num_maps, self.input_dim, self.hid_dim)) 
        self.W_out = nn.Parameter(torch.rand(self.num_maps, self.hid_dim, 1))
        self.b_in = nn.Parameter(torch.zeros(self.num_maps, 1, self.hid_dim))  # Zero initializer
        self.eta = nn.Parameter(torch.zeros(self.num_maps,1, self.input_dim))  # Random init for eta

        if xavier_init:
            for weight in [self.W_in, self.W_out, self.b_in, self.eta]:
                nn.init.xavier_uniform_(weight)

    def forward(self, X, Y):
        for i in range(self.num_maps):
            if self.tie_weights:
                X, Y = HenonMap(X, Y, self.W_in[0], self.W_out[0], 
                            self.b_in[0], self.eta[0], self.epsilon)
            else:
                X, Y = HenonMap(X, Y, self.W_in[i], self.W_out[i], 
                            self.b_in[i], self.eta[i], self.epsilon)
        return X, Y


class HenonNet(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers=2, epsilon=1, num_maps=4, tie_weights=False):
        super().__init__()
        self.tie_weights = tie_weights
        self.hlayers = [HenonLayer(input_dim, hid_dim, epsilon, num_maps, tie_weights) for _ in range(num_layers)]
        self.hlayers = nn.Sequential(*self.hlayers)

    def forward(self, X, Y):
        for layer in self.hlayers:
            X, Y = layer(X, Y)
        return X, Y



def InverseHenonMap(X_out, Y_out, W_in, W_out, b_in, eta, epsilon=1):
    '''
    Compute the inverse of HenonMap
    - X_out = Y + eta 
    - Y_out = -X + \epsilon dV(Y)/dY, for V a two-layer MLP
    as
    - Y = X_out - eta
    - X = -Y_out + \epsilon dV(Y)/dY
    '''
    Y = X_out - eta 
    Y.requires_grad_(True)
    V = torch.matmul(torch.tanh(torch.matmul(Y, W_in) + b_in), W_out)
    #V_grad = torch.autograd.grad(V.sum(), Y, create_graph=True)[0]  # Compute the gradient of V with respect to Y
    V_grad = torch.autograd.grad(outputs=V, inputs=Y,
                                 grad_outputs=torch.ones_like(V),
                                create_graph=True)[0]
    X = -Y_out + epsilon * V_grad
    return X, Y


def InverseHenonLayer(X_out, Y_out, HenonLayer):
    '''
    Given the output coordinates (X_out, Y_out)
    and the forward HenonLayer
    compute the inverse of HenonLayer that maps (X_out, Y_out) to (X, Y) original coordinates
    '''
    for i in range(-1, -1-HenonLayer.num_maps, -1):
        if HenonLayer.tie_weights:
            X_out, Y_out = InverseHenonMap(X_out, Y_out, HenonLayer.W_in[0], HenonLayer.W_out[0], 
                            HenonLayer.b_in[0], HenonLayer.eta[0], HenonLayer.epsilon)
        else:
            X_out, Y_out = InverseHenonMap(X_out, Y_out, HenonLayer.W_in[i], HenonLayer.W_out[i], 
                            HenonLayer.b_in[i], HenonLayer.eta[i], HenonLayer.epsilon)
    return X_out, Y_out


def InverseHenonNet(X_out, Y_out, HenonNet: nn.Module):
    '''
    Given a forward (trained) HenonNet and points in output space
    Compute its inverse that maps (X_out, Y_out) to (X, Y) in the original input space
    '''
    #for layer in HenonNet.hlayers:
    depth = len(HenonNet.hlayers)
    for i in range(-1, -1-depth, -1):
        X_out, Y_out = InverseHenonLayer(X_out, Y_out, HenonNet.hlayers[i])
    return X_out, Y_out

    
class HenonNetsupQ(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers=2, epsilon=1, num_maps=4, tie_weights=False):
        super().__init__()
        self.tie_weights = tie_weights
        self.hlayers = [HenonLayer(input_dim, hid_dim, epsilon, num_maps, tie_weights) for _ in range(num_layers)]
        self.hlayers = nn.Sequential(*self.hlayers)
        self.Q_predictor = nn.Sequential(*[
            nn.Linear(input_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1)
        ])

    def forward(self, X, Y):
        for layer in self.hlayers:
            X, Y = layer(X, Y)
        Jhat = 0.5*(X**2 + Y**2) #X = \sqrt{2J} sin theta, Y = \sqrt{2J} cos theta
        #out = torch.concatenate((X,Y),dim=-1)
        Qhat = self.Q_predictor(Jhat)
        return X, Y, Qhat


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer=2):
        super(MLP, self).__init__()
        layers = []
        hidden_sizes = [hidden_size] * (num_layer - 1)
        sizes = [input_size] + hidden_sizes
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
