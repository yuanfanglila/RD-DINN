import numpy as np
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.nn as nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DNN(nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
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

class Direct_problem:
    def __init__(self, X_IC, X_BC, S_IC, I_IC, X_f, layers):
        self.x_ic = torch.tensor(X_IC[:, 0].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.t_ic = torch.tensor(X_IC[:, 1].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.x_bc = torch.tensor(X_BC[:, 0].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.t_bc = torch.tensor(X_BC[:, 1].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.x_f = torch.tensor(X_f[:, 0].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.t_f = torch.tensor(X_f[:, 1].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.S_ic = torch.tensor(S_IC, dtype=torch.float32).to(device)
        self.I_ic = torch.tensor(I_IC, dtype=torch.float32).to(device)

        self.null = torch.zeros((self.x_f.shape[0], 1), dtype=torch.float32, requires_grad=True).to(device)
        self.zero = torch.zeros((self.x_bc.shape[0], 1), dtype=torch.float32, requires_grad=True).to(device)

        self.layers = layers
        self.DINN = DNN(layers).to(device)
        self.loss = nn.MSELoss()

        self.optimizer = torch.optim.LBFGS(
                                            self.DINN.parameters(),
                                            lr=0.01,
                                            max_iter=50000,
                                            max_eval=50000,
                                            history_size=50,
                                            tolerance_grad=1e-6,
                                            tolerance_change=1.0 * np.finfo(float).eps,
                                            line_search_fn="strong_wolfe"
                                        )
        self.iter = 0

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
        pde1 = S_t - 0.01 * S_xx + 0.8 * S * I - 0.5 + 0.1 * S

        I_xx = torch.autograd.grad(I_x, x, grad_outputs=torch.ones_like(I_x), create_graph=True)[0]
        I_t = torch.autograd.grad(I, t, grad_outputs=torch.ones_like(I), create_graph=True)[0]
        pde2 = I_t - 0.01 * I_xx - 0.8 * S * I + 0.5 * I + 0.1 * I
        return pde1, pde2

    def loss_func(self):
        self.optimizer.zero_grad()
        S_ic_pred, I_ic_pred = self.net_SIR(self.x_ic, self.t_ic)
        S_xNbc, I_xNbc = self.net_SIR_x(self.x_bc, self.t_bc)
        f_pred_1, f_pred_2 = self.net_r_SIR(self.x_f, self.t_f)

        S_bc_loss = self.loss(S_xNbc,self.zero)
        I_bc_loss = self.loss(I_xNbc,self.zero)
        loss_BC = S_bc_loss + I_bc_loss

        S_ic_loss = self.loss(S_ic_pred, self.S_ic)
        I_ic_loss = self.loss(I_ic_pred, self.I_ic)
        loss_IC = S_ic_loss + I_ic_loss

        f_loss_1 = self.loss(f_pred_1, self.null)
        f_loss_2 = self.loss(f_pred_2, self.null)
        loss_F = f_loss_1 + f_loss_2

        loss = loss_IC + loss_BC + loss_F
        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5f, Loss_IC: %.5f, Loss_BC: %.5f, Loss_F: %.5f' %
                (self.iter, loss.item(), loss_IC.item(), loss_BC.item(), loss_F.item())
                  )
        return loss

    def train(self):
        self.DINN.train()
        self.optimizer.step(self.loss_func)

    def predict(self, X):
        x = torch.tensor(X[:, 0], requires_grad=True).float().to(device)[:,None]
        t = torch.tensor(X[:, 1], requires_grad=True).float().to(device)[:,None]

        self.DINN.eval()
        S, I = self.net_SIR(x, t)
        S = S.detach().cpu().numpy()
        I = I.detach().cpu().numpy()
        return S, I


def savefig(filename, crop = True):
    if crop == True:
        plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig('{}.pdf'.format(filename))
