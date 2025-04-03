import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import scipy.io
import matplotlib
from scipy.interpolate import griddata
from scipy.io import loadmat
from collections import OrderedDict
from pyDOE import lhs
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
np.random.seed(1234)
torch.manual_seed(1234)

class PhysicsInformedNN(nn.Module):
    def __init__(self, X_IC, X_BC, S_IC, E_IC, I_IC, X_f, layers):
        super(PhysicsInformedNN, self).__init__()
        self.x_ic = torch.tensor(X_IC[:, 0].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.t_ic = torch.tensor(X_IC[:, 1].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.x_bc = torch.tensor(X_BC[:, 0].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.t_bc = torch.tensor(X_BC[:, 1].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.x_f = torch.tensor(X_f[:, 0].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.t_f = torch.tensor(X_f[:, 1].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.S_ic = torch.tensor(S_IC, dtype=torch.float32).to(device)
        self.E_ic = torch.tensor(E_IC, dtype=torch.float32).to(device)
        self.I_ic = torch.tensor(I_IC, dtype=torch.float32).to(device)

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

    def net_SEIRD(self, x, t):
        SEIRD = self.DINN(x, t)
        S = SEIRD[:, 0].reshape(-1, 1)
        E = SEIRD[:, 1].reshape(-1, 1)
        I = SEIRD[:, 2].reshape(-1, 1)
        return S, E, I

    def net_SEIRD_x(self, x, t):
        S, E, I = self.net_SEIRD(x, t)
        S_x = torch.autograd.grad(S, x, grad_outputs=torch.ones_like(S), create_graph=True)[0]
        E_x = torch.autograd.grad(E, x, grad_outputs=torch.ones_like(E), create_graph=True)[0]
        I_x = torch.autograd.grad(I, x, grad_outputs=torch.ones_like(I), create_graph=True)[0]
        return S_x, E_x, I_x

    def net_r_SEIRD(self, x, t):
        S, E, I = self.net_SEIRD(x, t)
        S_x, E_x, I_x = self.net_SEIRD_x(x, t)
        S_xx = torch.autograd.grad(S_x, x, grad_outputs=torch.ones_like(S_x), create_graph=True)[0]
        E_xx = torch.autograd.grad(E_x, x, grad_outputs=torch.ones_like(E_x), create_graph=True)[0]
        I_xx = torch.autograd.grad(I_x, x, grad_outputs=torch.ones_like(I_x), create_graph=True)[0]

        S_t = torch.autograd.grad(S, t, grad_outputs=torch.ones_like(S), create_graph=True)[0]
        E_t = torch.autograd.grad(E, t, grad_outputs=torch.ones_like(E), create_graph=True)[0]
        I_t = torch.autograd.grad(I, t, grad_outputs=torch.ones_like(I), create_graph=True)[0]

        pde1 = S_t - 0.005 * S_xx - 1 + 1.0 * S * I + 0.1 * S
        pde2 = E_t - 0.005 * E_xx - 1.0 * S * I + 1.0 * E + 0.1 * E
        pde3 = I_t - 0.005 * I_xx - 1.0 * E + 1.0 * I + 0.1 * I
        return pde1, pde2, pde3

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            S_ic_pred, E_ic_pred, I_ic_pred = self.net_SEIRD(self.x_ic, self.t_ic)
            S_xNbc, E_xNbc, I_xNbc = self.net_SEIRD_x(self.x_bc, self.t_bc)
            f_pred_1, f_pred_2, f_pred_3 = self.net_r_SEIRD(self.x_f, self.t_f)

            S_bc_loss = self.loss(S_xNbc, self.zero)
            E_bc_loss = self.loss(E_xNbc, self.zero)
            I_bc_loss = self.loss(I_xNbc, self.zero)
            loss_BC = S_bc_loss + I_bc_loss + E_bc_loss

            S_ic_loss = self.loss(S_ic_pred, self.S_ic)
            E_ic_loss = self.loss(E_ic_pred, self.E_ic)
            I_ic_loss = self.loss(I_ic_pred, self.I_ic)
            loss_IC = S_ic_loss + I_ic_loss + E_ic_loss

            f_loss_1 = self.loss(f_pred_1, self.null)
            f_loss_2 = self.loss(f_pred_2, self.null)
            f_loss_3 = self.loss(f_pred_3, self.null)
            loss_F = f_loss_1 + f_loss_2 + f_loss_3

            loss = loss_IC + 10*loss_BC + 10*loss_F

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.losses.append(loss.item())
            self.loss_IC.append(loss_IC.item())
            self.loss_BC.append(loss_BC.item())
            self.loss_F.append(loss_F.item())

            if epoch % 50 == 0:
                print(
                    'Iter %d, Loss: %.5e, Loss_IC: %.5e, Loss_BC: %.5e, Loss_F: %.5e' %
                    (
                        epoch, loss.item(), loss_IC.item(), loss_BC.item(), loss_F.item()
                    )
                )

    def predict(self, X):
        x = torch.tensor(X[:, 0], requires_grad=True).float().to(device)[:,None]
        t = torch.tensor(X[:, 1], requires_grad=True).float().to(device)[:,None]

        self.DINN.eval()
        S, E, I = self.net_SEIRD(x, t)
        S = S.detach().cpu().numpy()
        E = E.detach().cpu().numpy()
        I = I.detach().cpu().numpy()
        return S, E, I

N_u = 50
N_f = 10000
data = loadmat('../data/SEIR_alpha_1.0.mat')
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
S_Exact = np.real(data['u1'])
E_Exact = np.real(data['u2'])
I_Exact = np.real(data['u3'])

X, T = np.meshgrid(x,t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
S_star = S_Exact.flatten()[:, None]
E_star = E_Exact.flatten()[:, None]
I_star = I_Exact.flatten()[:, None]
lb = X_star.min(0)
ub = X_star.max(0)

xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
II1 = I_Exact[0:1, :].T
SS1 = S_Exact[0:1, :].T
EE1 = E_Exact[0:1, :].T

xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
II2 = I_Exact[:, 0:1]
SS2 = S_Exact[:, 0:1]
EE2 = E_Exact[:, 0:1]
xx3 = np.hstack((X[:, -1:], T[:, -1:]))
II3 = I_Exact[:, -1:]
SS3 = S_Exact[:, -1:]
EE3 = E_Exact[:, -1:]

X_SI_ic_train = xx1
X_SI_bc_train = np.vstack([xx2, xx3])
X_f_train = lb + (ub-lb)*lhs(2, N_f)
X_f_train = np.vstack((X_f_train, X_SI_bc_train, X_SI_ic_train))
S_ic_train = SS1
I_ic_train = II1
E_ic_train = EE1

idx1 = np.random.choice(X_SI_ic_train.shape[0], N_u, replace=False)
idx2 = np.random.choice(X_SI_bc_train.shape[0], N_u, replace=False)
X_SI_ic_train = X_SI_ic_train[idx1, :]
X_SI_bc_train = X_SI_bc_train[idx2, :]
S_ic_train = S_ic_train[idx1, :]
E_ic_train = E_ic_train[idx1, :]
I_ic_train = I_ic_train[idx1, :]

layers = [2, 80, 80, 80, 80, 80, 3]
dinn = PhysicsInformedNN(X_SI_ic_train, X_SI_bc_train, S_ic_train, E_ic_train, I_ic_train, X_f_train, layers)

learning_rate = 1e-6
optimizer = optim.Adam(dinn.params, lr=learning_rate)
dinn.optimizer = optimizer
scheduler = torch.optim.lr_scheduler.CyclicLR(dinn.optimizer,
                                              base_lr=1e-5,
                                              max_lr=1e-3,
                                              step_size_up=500,
                                              mode="exp_range",
                                              gamma=0.85,
                                              cycle_momentum=False)
dinn.scheduler = scheduler
dinn.train(20000)

S_pred, E_pred, I_pred = dinn.predict(X_star)
error_S = np.linalg.norm(S_star-S_pred,2)/np.linalg.norm(S_star,2)
error_I = np.linalg.norm(I_star-I_pred,2)/np.linalg.norm(I_star,2)
print('Error S: %e' % (error_S))
print('Error I: %e' % (error_I))

fig = plt.figure(figsize=(11, 6))
left, bottom, width, height = 0.12, 0.13, 0.8, 0.8
ax1 = fig.add_axes([left, bottom, width, height])
j = 10000
average_loss_total = [np.mean(dinn.losses[i:i+j]) for i in range(0, len(dinn.losses), j)]
average_loss_IC = [np.mean(dinn.loss_IC[i:i+j]) for i in range(0, len(dinn.loss_IC), j)]
average_loss_BC = [np.mean(dinn.loss_BC[i:i+j]) for i in range(0, len(dinn.loss_BC), j)]
average_loss_F = [np.mean(dinn.loss_F[i:i+j]) for i in range(0, len(dinn.loss_F), j)]
# 绘制平均损失值
ax1.plot(average_loss_total, label='Total', color='teal')
ax1.plot(average_loss_IC, label='IC', color='red')
ax1.plot(average_loss_BC, label='BC', color='g')
ax1.plot(average_loss_F, label='Function', color='b')

plt.yscale('log')
ax1.set_xlabel('Epoch', fontdict={'fontsize': 20, 'fontfamily':'serif'})
ax1.set_ylabel('Relative $L^2$ Error', fontdict={'fontsize': 20, 'fontfamily':'serif'})

custom_ticks = [0,2,4,6,8,10,12,14]
custom_labels = ['0', '1', '2', '3', '4', '5', '6', '7']
ax1.set_xticks(custom_ticks)
ax1.set_xticklabels(custom_labels)
plt.legend()
plt.show()

i_pred = griddata(X_star, I_pred.flatten(), (X, T), method='cubic')
e_pred = griddata(X_star, E_pred.flatten(), (X, T), method='cubic')
s_pred = griddata(X_star, S_pred.flatten(), (X, T), method='cubic')
ErrorS = np.abs(S_Exact - s_pred)
ErrorI = np.abs(I_Exact - i_pred)
ErrorE = np.abs(E_Exact - e_pred)
print(ErrorS,ErrorE,ErrorI)

matplotlib.rcParams.update({'font.size': 25})
fig, axes = plt.subplots(3, 3, figsize=(28, 25))
Exact = [S_Exact, E_Exact, I_Exact]
Pred = [s_pred, e_pred, i_pred]
title1 = ['Reference S(x,t)', 'Reference E(x,t)', 'Reference I(x,t)']
title2 = ['Predicted S(x,t)', 'Predicted E(x,t)', 'Predicted I(x,t)']
title3 = ['Predicted S(x,t)', 'Predicted E(x,t)', 'Predicted I(x,t)']
for i in range(3):
    for j in range(3):
        if j == 0:
            ax = axes[i, j]
            pcm = ax.pcolormesh(T, X, Exact[i], cmap='jet')
            cbar = fig.colorbar(pcm)
            ax.set_xlabel(r'$t$')
            ax.set_ylabel(r'$x$')
            ax.set_title(title1[i])
        elif j == 1:
            ax = axes[i, j]
            pcm = ax.pcolormesh(T, X, Pred[i], cmap='jet')
            ax.plot(
                X_SI_bc_train[:, 1],
                X_SI_bc_train[:, 0],
                'kx',
                markersize=7,
                clip_on=False,
                alpha=1.0)
            ax.plot(
                X_SI_ic_train[:, 1],
                X_SI_ic_train[:, 0],
                'kx',
                markersize=7,
                clip_on=False,
                alpha=1.0
            )
            cbar = fig.colorbar(pcm)
            ax.set_xlabel(r'$t$')
            ax.set_ylabel(r'$x$')
            ax.set_title(title2[i])
        else:
            if i == 0:
                ax = axes[i, j]
                norm = matplotlib.colors.Normalize(vmin=0.00, vmax=0.035)
                pcm = ax.pcolormesh(T, X, np.abs(Exact[i] - Pred[i]), cmap='jet', norm=norm)
                cbar = fig.colorbar(pcm, format='%.0e')
                ax.set_xlabel(r'$t$')
                ax.set_ylabel(r'$x$')
                ax.set_title('Absolute Error')
            elif i == 1:
                ax = axes[i, j]
                norm = matplotlib.colors.Normalize(vmin=0.00, vmax=0.025)
                pcm = ax.pcolormesh(T, X, np.abs(Exact[i] - Pred[i]), cmap='jet', norm=norm)
                cbar = fig.colorbar(pcm, format='%.0e')
                ax.set_xlabel(r'$t$')
                ax.set_ylabel(r'$x$')
                ax.set_title('Absolute Error')
            elif i == 2:
                ax = axes[i, j]
                norm = matplotlib.colors.Normalize(vmin=0.00, vmax=0.025)
                pcm = ax.pcolormesh(T, X, np.abs(Exact[i] - Pred[i]), cmap='jet', norm=norm)
                cbar = fig.colorbar(pcm, format='%.0e')
                ax.set_xlabel(r'$t$')
                ax.set_ylabel(r'$x$')
                ax.set_title('Absolute Error')
plt.tight_layout()
plt.show()

fig_1 = plt.figure(1, figsize=(19, 15))
plt.subplot(3, 3, 1)
plt.plot(t, S_Exact[:, 25], "b-", linewidth=4, label="Exact")
plt.plot(t, s_pred[:, 25], "r--", linewidth=4, label="Prediction")
plt.ylabel('$S(x,t)$', fontdict={'fontsize': 30, 'fontfamily':'serif'})
plt.title('$x=0.25$', fontdict={'fontsize': 30, 'fontfamily':'serif'})

plt.subplot(3, 3, 2)
plt.plot(t, S_Exact[:, 50], "b-", linewidth=4, label="Exact")
plt.plot(t, s_pred[:, 50], "r--", linewidth=4, label="Prediction")
plt.title('$x=0.50$', fontdict={'fontsize': 30, 'fontfamily':'serif'})

plt.subplot(3, 3, 3)
plt.plot(t, S_Exact[:, 75], "b-", linewidth=4, label="Exact")
plt.plot(t, s_pred[:, 75], "r--", linewidth=4, label="Prediction")
plt.title('$x=0.75$', fontdict={'fontsize': 30, 'fontfamily':'serif'})

plt.subplot(3, 3, 4)
plt.plot(t, E_Exact[:, 25], "b-", linewidth=4, label="Exact")
plt.plot(t, e_pred[:, 25], "r--", linewidth=4, label="Prediction")
plt.xlabel('$t$', fontdict={'fontsize': 30, 'fontfamily':'serif'})
plt.ylabel('$E(x,t)$', fontdict={'fontsize': 30, 'fontfamily':'serif'})

plt.subplot(3, 3, 5)
plt.plot(t, E_Exact[:, 50], "b-", linewidth=4, label="Exact")
plt.plot(t, e_pred[:, 50], "r--", linewidth=4, label="Prediction")
plt.xlabel('$t$', fontdict={'fontsize': 30, 'fontfamily':'serif'})

plt.subplot(3, 3, 6)
plt.plot(t, E_Exact[:, 75], "b-", linewidth=4, label="Exact")
plt.plot(t, e_pred[:, 75], "r--", linewidth=4, label="Prediction")
plt.xlabel('$t$', fontdict={'fontsize': 30, 'fontfamily':'serif'})

plt.subplot(3, 3, 7)
plt.plot(t, I_Exact[:, 25], "b-", linewidth=4, label="Exact")
plt.plot(t, i_pred[:, 25], "r--", linewidth=4, label="Prediction")
plt.xlabel('$t$', fontdict={'fontsize': 30, 'fontfamily':'serif'})
plt.ylabel('$I(x,t)$', fontdict={'fontsize': 30, 'fontfamily':'serif'})

plt.subplot(3, 3, 8)
plt.plot(t, I_Exact[:, 50], "b-", linewidth=4, label="Exact")
plt.plot(t, i_pred[:, 50], "r--", linewidth=4, label="Prediction")
plt.xlabel('$t$', fontdict={'fontsize': 30, 'fontfamily':'serif'})

plt.subplot(3, 3, 9)
plt.plot(t, I_Exact[:, 75], "b-", linewidth=4, label="Exact")
plt.plot(t, i_pred[:, 75], "r--", linewidth=4, label="Prediction")
plt.xlabel('$t$', fontdict={'fontsize': 30, 'fontfamily':'serif'})

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
           fancybox=True, shadow=True, ncol=4, fontsize=20)

plt.tight_layout()
plt.show()

matplotlib.rcParams['font.size'] = 15

fig_1 = plt.figure(1, figsize=(18, 5))
plt.subplot(1, 3, 1)
# norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
plt.pcolor(T, X, S_Exact, cmap='jet')
plt.colorbar()
plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.title('Reference', fontdict={'fontsize': 30, 'fontfamily':'serif'})

plt.subplot(1, 3, 2)
# norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
plt.pcolor(T, X, s_pred, cmap='jet')
random_indices = np.random.choice(len(X_SI_bc_train), size=50, replace=True)
X_SI_bc_train = X_SI_bc_train[random_indices]
X_SI_ic_train = X_SI_ic_train[random_indices]
plt.plot(
    X_SI_bc_train[:, 1],
    X_SI_bc_train[:, 0],
    'kx',
    markersize=7,
    clip_on=False,
    alpha=1.0)
plt.plot(
    X_SI_ic_train[:, 1],
    X_SI_ic_train[:, 0],
    'kx',
    markersize=7,
    clip_on=False,
    alpha=1.0
)
plt.colorbar()
plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.title('Predicted', fontdict={'fontsize': 30,  'fontfamily':'serif'})

plt.subplot(1, 3, 3)
norm = matplotlib.colors.Normalize(vmin=0.00, vmax=0.035)
plt.pcolor(T, X, np.abs(S_Exact - s_pred), cmap='jet', norm=norm)
plt.colorbar(format='%.0e')
plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.title('Absolute Error', fontdict={'fontsize': 30, 'fontfamily':'serif'})

plt.tight_layout()
plt.show()

fig_2 = plt.figure(1, figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.pcolor(T, X, E_Exact, cmap='jet')
plt.colorbar()
plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.title('Reference', fontdict={'fontsize': 30, 'fontfamily':'serif'})

plt.subplot(1, 3, 2)
plt.pcolor(T, X, e_pred, cmap='jet')
random_indices = np.random.choice(len(X_SI_bc_train), size=50, replace=True)
X_SI_bc_train = X_SI_bc_train[random_indices]
X_SI_ic_train = X_SI_ic_train[random_indices]
plt.plot(
    X_SI_bc_train[:, 1],
    X_SI_bc_train[:, 0],
    'kx',
    markersize=7,  # marker size doubled
    clip_on=False,
    alpha=1.0)
plt.plot(
    X_SI_ic_train[:, 1],
    X_SI_ic_train[:, 0],
    'kx',
    markersize=7,  # marker size doubled
    clip_on=False,
    alpha=1.0
)
plt.colorbar()
plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.title('Predicted', fontdict={'fontsize': 30,  'fontfamily':'serif'})


plt.subplot(1, 3, 3)
norm = matplotlib.colors.Normalize(vmin=0.00, vmax=0.025)
plt.pcolor(T, X, np.abs(E_Exact - e_pred), cmap='jet', norm=norm)
plt.colorbar(format='%.0e')
plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.title('Absolute Error', fontdict={'fontsize': 30, 'fontfamily':'serif'})

plt.tight_layout()
plt.show()

fig_3 = plt.figure(1, figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.pcolor(T, X, I_Exact, cmap='jet')
plt.colorbar()
plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.title('Reference', fontdict={'fontsize': 30, 'fontfamily':'serif'})

plt.subplot(1, 3, 2)
plt.pcolor(T, X, i_pred, cmap='jet')
random_indices = np.random.choice(len(X_SI_bc_train), size=50, replace=True)
X_SI_bc_train = X_SI_bc_train[random_indices]
X_SI_ic_train = X_SI_ic_train[random_indices]
plt.plot(
    X_SI_bc_train[:, 1],
    X_SI_bc_train[:, 0],
    'kx',
    markersize=7,  # marker size doubled
    clip_on=False,
    alpha=1.0)
plt.plot(
    X_SI_ic_train[:, 1],
    X_SI_ic_train[:, 0],
    'kx',
    markersize=7,  # marker size doubled
    clip_on=False,
    alpha=1.0
)
plt.colorbar()
plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.title('Predicted', fontdict={'fontsize': 30,  'fontfamily':'serif'})

plt.subplot(1, 3, 3)
norm = matplotlib.colors.Normalize(vmin=0.00, vmax=0.025)
plt.pcolor(T, X, np.abs(I_Exact - i_pred), cmap='jet', norm=norm)
plt.colorbar(format='%.0e')
plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.title('Absolute Error', fontdict={'fontsize': 30, 'fontfamily':'serif'})

plt.tight_layout()
plt.show()