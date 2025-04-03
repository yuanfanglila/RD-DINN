import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['font.size'] = 15
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from scipy.io import loadmat
from collections import OrderedDict
from pyDOE import lhs
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
np.random.seed(1234)
torch.manual_seed(1234)
print(device)

class PhysicsInformedNN(nn.Module):
    def __init__(self, X_IC, X_BC, S_IC, I_IC, X_f, layers, beta=0.5, alpha=0.1):
        super(PhysicsInformedNN, self).__init__()

        self.x_ic = torch.tensor(X_IC[:, 0].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.t_ic = torch.tensor(X_IC[:, 1].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.x_bc = torch.tensor(X_BC[:, 0].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.t_bc = torch.tensor(X_BC[:, 1].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.x_f = torch.tensor(X_f[:, 0].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.t_f = torch.tensor(X_f[:, 1].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.S_ic = torch.tensor(S_IC, dtype=torch.float32).to(device)
        self.I_ic = torch.tensor(I_IC, dtype=torch.float32).to(device)

        self.beta = torch.tensor([beta], dtype=torch.float32).to(device)
        self.alpha = torch.tensor([alpha], dtype=torch.float32).to(device)

        self.null = torch.zeros((self.x_f.shape[0], 1), dtype=torch.float32, requires_grad=True).to(device)
        self.zero = torch.zeros((self.x_bc.shape[0], 1), dtype=torch.float32, requires_grad=True).to(device)

        self.layers = layers
        self.DINN = self.DNN(layers).to(device)
        self.params = list(self.DINN.parameters())
        self.loss = nn.MSELoss()

        self.losses = []
        self.loss_IC = []
        self.loss_BC = []
        self.loss_F = []

    class DNN(nn.Module):
        def __init__(self, layers):
            super(PhysicsInformedNN.DNN, self).__init__()
            self.depth = len(layers) - 1
            self.activation = nn.Tanh
            layer_list = list()
            for i in range(self.depth - 1):
                layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
                layer_list.append(('activation_%d' % i, self.activation()))
            layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
            layerdict = OrderedDict(layer_list)
            self.layers = torch.nn.Sequential(layerdict)

        def forward(self, x, t):
            inputs = torch.hstack((x, t))
            out = self.layers(inputs)
            return out

    def net_SIR(self, x, t):
        SIR = self.DINN(x, t)
        S = SIR[:,0].reshape(-1,1)
        I = SIR[:,1].reshape(-1,1)
        return S, I

    def net_SIR_x(self, x, t):
        S, I = self.net_SIR(x, t)
        S_x = torch.autograd.grad(S, x, grad_outputs=torch.ones_like(S), create_graph=True)[0]
        I_x = torch.autograd.grad(I, x, grad_outputs=torch.ones_like(I), create_graph=True)[0]
        return S_x, I_x

    def net_r_SIR(self, x, t):
        S, I = self.net_SIR(x, t)
        S_x, I_x = self.net_SIR_x(x, t)
        S_xx = torch.autograd.grad(S_x, x, grad_outputs=torch.ones_like(S_x), create_graph=True)[0]
        S_t = torch.autograd.grad(S, t, grad_outputs=torch.ones_like(S), create_graph=True)[0]
        pde1 = S_t - 0.001 * S_xx + self.beta * S * I - 0.1 + 0.1 * S

        I_xx = torch.autograd.grad(I_x, x, grad_outputs=torch.ones_like(I_x), create_graph=True)[0]
        I_t = torch.autograd.grad(I, t, grad_outputs=torch.ones_like(I), create_graph=True)[0]
        pde2 = I_t - 0.001 * I_xx - self.beta * S * I + self.alpha * I + 0.1 * I
        return pde1, pde2

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            S_ic_pred, I_ic_pred = self.net_SIR(self.x_ic, self.t_ic)
            S_xNbc, I_xNbc = self.net_SIR_x(self.x_bc, self.t_bc)
            f_pred_1, f_pred_2 = self.net_r_SIR(self.x_f, self.t_f)

            S_bc_loss = self.loss(S_xNbc, self.zero)
            I_bc_loss = self.loss(I_xNbc, self.zero)
            loss_BC = S_bc_loss + I_bc_loss

            S_ic_loss = self.loss(S_ic_pred, self.S_ic)
            I_ic_loss = self.loss(I_ic_pred, self.I_ic)
            loss_IC = S_ic_loss + I_ic_loss

            f_loss_1 = self.loss(f_pred_1, self.null)
            f_loss_2 = self.loss(f_pred_2, self.null)
            loss_F = f_loss_1 + f_loss_2

            loss = loss_IC + loss_BC + loss_F

            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()

            self.losses.append(loss.item())
            self.loss_IC.append(loss_IC.item())
            self.loss_BC.append(loss_BC.item())
            self.loss_F.append(loss_F.item())

            if epoch % 10 == 0:
                print(
                    'Iter %d, Loss: %.3e, Loss_IC: %.5e, Loss_BC: %.5e, Loss_F: %.5e' %
                    (
                        epoch, loss.item(), loss_IC.item(), loss_BC.item(), loss_F.item()
                    )
                )

    def predict(self, X):
        x = torch.tensor(X[:, 0], requires_grad=True).float().to(device)[:,None]
        t = torch.tensor(X[:, 1], requires_grad=True).float().to(device)[:,None]

        self.DINN.eval()
        S, I = self.net_SIR(x, t)
        S = S.detach().cpu().numpy()
        I = I.detach().cpu().numpy()
        return S, I

layers = [2, 100, 100, 100, 100, 100, 100, 2]
def sensitivity_analysis(param_range, param_name, data_path_template):
    mse_results = []
    mse_results_i = []
    for value in param_range:
        N_u = 60
        N_f = 3000
        data_path = data_path_template.format(beta=value, alpha=value)
        data = loadmat(data_path)
        t = data['t'].flatten()[:, None]
        x = data['x'].flatten()[:, None]
        I_Exact = np.real(data['u2'])
        S_Exact = np.real(data['u1'])

        X, T = np.meshgrid(x, t)
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        S_star = S_Exact.flatten()[:, None]
        I_star = I_Exact.flatten()[:, None]
        lb = X_star.min(0)
        ub = X_star.max(0)

        xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
        II1 = I_Exact[0:1, :].T
        SS1 = S_Exact[0:1, :].T

        xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
        xx3 = np.hstack((X[:, -1:], T[:, -1:]))

        X_SI_ic_train = xx1
        X_SI_bc_train = np.vstack([xx2, xx3])
        X_f_train = lb + (ub - lb) * lhs(2, N_f)
        X_f_train = np.vstack((X_f_train, X_SI_bc_train, X_SI_ic_train))
        S_ic_train = SS1
        I_ic_train = II1

        idx1 = np.random.choice(X_SI_ic_train.shape[0], N_u, replace=False)
        idx2 = np.random.choice(X_SI_bc_train.shape[0], N_u, replace=False)
        X_SI_ic_train = X_SI_ic_train[idx1, :]
        X_SI_bc_train = X_SI_bc_train[idx2, :]
        S_ic_train = S_ic_train[idx1, :]
        I_ic_train = I_ic_train[idx1, :]

        if param_name == 'beta':
            model = PhysicsInformedNN(X_SI_ic_train, X_SI_bc_train, S_ic_train, I_ic_train,
                                      X_f_train, layers, beta=value, alpha=0.5)
        elif param_name == 'alpha':
            model = PhysicsInformedNN(X_SI_ic_train, X_SI_bc_train, S_ic_train, I_ic_train,
                                      X_f_train, layers, beta=0.5, alpha=value)

        model.beta.requires_grad_(False)
        model.alpha.requires_grad_(False)

        optimizer = optim.Adam(model.DINN.parameters(), lr=1e-3)
        model.optimizer = optimizer
        model.train(1000)

        S_pred, I_pred = model.predict(X_star)
        mse = np.mean((S_star - S_pred) ** 2)
        mse_i = np.mean((I_star - I_pred) ** 2)
        mse_results.append(mse)
        mse_results_i.append(mse_i)
        print(f'{param_name}={value}, MSE={mse:.3e}')
        print(mse_results)
        print(mse_results_i)

    plt.plot(param_range, mse_results, marker='o')
    plt.xlabel(param_name)
    plt.ylabel('MSE')
    plt.title(f'Sensitivity to {param_name}')
    plt.show()

param_range = [0.1, 0.3, 0.5, 0.7, 0.9]
data_path_template = '../SIR_beta={beta}_alpha=0.5.mat'
sensitivity_analysis(param_range, 'beta', data_path_template)