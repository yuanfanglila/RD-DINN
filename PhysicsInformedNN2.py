import numpy as np
import torch
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DNN(nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = nn.Tanh
        layer_list = list()

        # Define layers and activations
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))

        # Output layer (without activation)
        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))

        # Create the sequential model
        layerdict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerdict)

    def forward(self, x, t):
        # Concatenate x and t
        x = torch.hstack((x, t))

        # Save the input for residual connection
        residual = x

        # Pass through the layers
        out = self.layers(x)

        # Ensure the residual has the same shape as the output
        if out.size(1) != residual.size(1):
            residual = torch.nn.functional.pad(residual, (0, out.size(1) - residual.size(1)))

        # Add the residual connection
        out += residual

        return out

class Inverse_problem:
    def __init__(self, X, S, I, lb, ub, layers):
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        self.x = torch.tensor(X[:, 0].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.t = torch.tensor(X[:, 1].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.S = torch.tensor(S).float().to(device)
        self.I = torch.tensor(I).float().to(device)

        self.beta = torch.tensor([0.0], requires_grad=True).to(device)
        self.alpha = torch.tensor([0.0], requires_grad=True).to(device)
        self.beta = torch.nn.Parameter(self.beta)
        self.alpha = torch.nn.Parameter(self.alpha)

        self.null = torch.zeros((self.x.shape[0], 1), dtype=torch.float32, requires_grad=True).to(device)

        self.DINN = DNN(layers).to(device)
        self.DINN.register_parameter('beta', self.beta)
        self.DINN.register_parameter('alpha', self.alpha)
        self.loss = nn.MSELoss()

        self.optimizer = torch.optim.LBFGS(
                                            self.DINN.parameters(),
                                            lr=1.0,
                                            max_iter=2000,
                                            max_eval=2000,
                                            history_size=50,
                                            tolerance_grad=1e-6,
                                            tolerance_change=1.0 * np.finfo(float).eps,
                                            line_search_fn="strong_wolfe"
                                        )
        self.optimizer_Adam = torch.optim.Adam(self.DINN.parameters())
        self.losses = []
        self.loss_D = []
        self.loss_F = []
        self.beta_list = []
        self.alpha_list = []

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
        beta = self.beta
        alpha = torch.exp(self.alpha)
        S_xx = torch.autograd.grad(S_x, x, grad_outputs=torch.ones_like(S_x), create_graph=True)[0]
        S_t = torch.autograd.grad(S, t, grad_outputs=torch.ones_like(S), create_graph=True)[0]
        pde1 = S_t - 0.001 * S_xx + beta * S * I - 0.1 + 0.1 * S    ##- alpha * I

        I_xx = torch.autograd.grad(I_x, x, grad_outputs=torch.ones_like(I_x), create_graph=True)[0]
        I_t = torch.autograd.grad(I, t, grad_outputs=torch.ones_like(I), create_graph=True)[0]
        pde2 = I_t - 0.001 * I_xx - beta * S * I + alpha * I + 0.1 * I
        return pde1, pde2

    def loss_func(self):
        S_pred, I_pred = self.net_SIR(self.x, self.t)
        f_pred_1, f_pred_2 = self.net_r_SIR(self.x, self.t)
        S_loss = self.loss(S_pred, self.S)
        I_loss = self.loss(I_pred, self.I)
        loss_SIR = S_loss + I_loss

        f_loss_1 = self.loss(f_pred_1, self.null)
        f_loss_2 = self.loss(f_pred_2, self.null)
        loss_F = f_loss_1 + f_loss_2

        loss = loss_SIR + loss_F
        self.optimizer.zero_grad()
        loss.backward()

        self.iter += 1
        if self.iter % 10 == 0:
            print(
                'Loss: %e, beta: %.5f, alpha: %.5f' %
                (
                    loss.item(),
                    self.beta.item(),
                    torch.exp(self.alpha.detach()).item()
                )
            )
        return loss

    def train(self, nIter):
        self.DINN.train()
        for epoch in range(nIter):
            S_pred, I_pred = self.net_SIR(self.x, self.t)
            f_pred_1, f_pred_2 = self.net_r_SIR(self.x, self.t)
            S_loss = self.loss(S_pred, self.S)
            I_loss = self.loss(I_pred, self.I)
            loss_D = S_loss + I_loss

            f_loss_1 = self.loss(f_pred_1, self.null)
            f_loss_2 = self.loss(f_pred_2, self.null)
            loss_F = f_loss_1 + f_loss_2

            loss = loss_D + loss_F

            self.optimizer_Adam.zero_grad()
            loss.backward()

            self.optimizer_Adam.step()
            self.losses.append(loss.item())
            self.loss_D.append(loss_D.item())
            self.loss_F.append(loss_F.item())
            self.beta_list.append(self.beta.item())
            self.alpha_list.append(torch.exp(self.alpha).item())
            if epoch % 10 == 0:
                print(
                    'It: %d, Loss: %.3e, beta: %.3f, alpha: %.6f' %
                    (
                        epoch,
                        loss.item(),
                        self.beta.item(),
                        torch.exp(self.alpha).item()
                    )
                )
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
        plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0.02)
    else:
        plt.savefig('{}.pdf'.format(filename))
