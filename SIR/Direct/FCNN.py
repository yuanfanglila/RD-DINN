import numpy as np
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import time
from scipy.interpolate import griddata
from scipy.io import loadmat
from collections import OrderedDict
from pyDOE import lhs
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
np.random.seed(1234)
torch.manual_seed(1234)

class PhysicsInformedNN(nn.Module):
    def __init__(self, X_IC, X_BC, S_IC, I_IC, X_f, layers):
        super(PhysicsInformedNN, self).__init__()
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
        pde1 = S_t - 0.001 * S_xx + 0.5 * S * I - 0.1 + 0.1 * S

        I_xx = torch.autograd.grad(I_x, x, grad_outputs=torch.ones_like(I_x), create_graph=True)[0]
        I_t = torch.autograd.grad(I, t, grad_outputs=torch.ones_like(I), create_graph=True)[0]
        pde2 = I_t - 0.001 * I_xx - 0.5 * S * I + 0.5 * I + 0.1 * I
        return pde1, pde2

    def train(self, n_epochs):
        self.epoch_predictions = []
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

            loss = loss_IC + loss_BC

            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()

            self.losses.append(loss.item())
            self.loss_IC.append(loss_IC.item())
            self.loss_BC.append(loss_BC.item())

            if epoch % 10 == 0:
                print(
                    'Iter %d, Loss: %.3e, Loss_IC: %.5e, Loss_BC: %.5e' %
                    (
                        epoch, loss.item(), loss_IC.item(), loss_BC.item()
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

N_u = 60
N_f = 3000
data = loadmat('../data/SIR_beta=0.5_alpha=0.7.mat')
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
I_Exact = np.real(data['u2'])
S_Exact = np.real(data['u1'])

X, T = np.meshgrid(x,t)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
S_star = S_Exact.flatten()[:,None]
I_star = I_Exact.flatten()[:,None]
lb = X_star.min(0)
ub = X_star.max(0)

xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
II1 = I_Exact[0:1,:].T
SS1 = S_Exact[0:1,:].T

xx2 = np.hstack((X[:,0:1], T[:,0:1]))
II2 = I_Exact[:,0:1]
SS2 = S_Exact[:,0:1]
xx3 = np.hstack((X[:,-1:], T[:,-1:]))
II3 = I_Exact[:,-1:]
SS3 = S_Exact[:,-1:]

X_SI_ic_train = xx1
X_SI_bc_train = np.vstack([xx2, xx3])
X_f_train = lb + (ub-lb)*lhs(2, N_f)
X_f_train = np.vstack((X_f_train, X_SI_bc_train, X_SI_ic_train))
S_ic_train = SS1
I_ic_train = II1

idx1 = np.random.choice(X_SI_ic_train.shape[0], N_u, replace=False)
idx2 = np.random.choice(X_SI_bc_train.shape[0], N_u, replace=False)
X_SI_ic_train = X_SI_ic_train[idx1, :]
X_SI_bc_train = X_SI_bc_train[idx2, :]
S_ic_train = S_ic_train[idx1,:]
I_ic_train = I_ic_train[idx1,:]

layers = [2, 100, 100, 100, 100, 100, 2]
dinn = PhysicsInformedNN(X_SI_ic_train, X_SI_bc_train, S_ic_train, I_ic_train, X_f_train, layers)

learning_rate = 1e-3
optimizer = optim.Adam(dinn.params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
dinn.optimizer = optimizer
# scheduler = torch.optim.lr_scheduler.CyclicLR(dinn.optimizer,
#                                               base_lr=1e-5,
#                                               max_lr=1e-3,
#                                               step_size_up=500,
#                                               mode="exp_range",
#                                               gamma=0.85,
#                                               cycle_momentum=False)
# dinn.scheduler = scheduler
start_time = time.time()
dinn.train(5000)
end_time = time.time()
time = end_time - start_time
print(f'total cost {time} seconds')

S_pred, I_pred = dinn.predict(X_star)
error_S = np.linalg.norm(S_star-S_pred,2)/np.linalg.norm(S_star,2)
error_I = np.linalg.norm(I_star-I_pred,2)/np.linalg.norm(I_star,2)
print('Error S: %e' % (error_S))
print('Error I: %e' % (error_I))

i_pred = griddata(X_star, I_pred.flatten(), (X, T), method='cubic')
s_pred = griddata(X_star, S_pred.flatten(), (X, T), method='cubic')
ErrorI = np.abs(I_Exact - i_pred)
ErrorS = np.abs(S_Exact - s_pred)

matplotlib.rcParams['font.size'] = 15
fig_1 = plt.figure(1, figsize=(18, 5))
plt.subplot(1, 3, 1)
norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
plt.pcolor(T, X, S_Exact, cmap='jet',norm=norm)
plt.colorbar()
plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.title('Reference', fontdict={'fontsize': 30, 'fontfamily':'serif'})

plt.subplot(1, 3, 2)
norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
plt.pcolor(T, X, s_pred, cmap='jet',norm=norm)
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
norm = matplotlib.colors.Normalize(vmin=0, vmax=0.2)
plt.pcolor(T, X, np.abs(S_Exact - s_pred), cmap='jet', norm=norm)
plt.colorbar(format='%.0e')
plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily':'serif'})
plt.title('Absolute Error', fontdict={'fontsize': 30, 'fontfamily':'serif'})


plt.tight_layout()
plt.show()

