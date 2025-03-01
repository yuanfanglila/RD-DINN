import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PhysicsInformedNN1 import Inverse_problem, savefig
import torch
import matplotlib
import matplotlib.gridspec as gridspec
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

N_u = 3000
data = loadmat('../Global （beta=0.5）.mat')

t = data['t'].flatten()[:, None]
t = t
x = data['x'].flatten()[:, None]
I_Exact = np.real(data['u2'])
S_Exact = np.real(data['u1'])

X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
S_star = S_Exact.flatten()[:, None]
I_star = I_Exact.flatten()[:, None]

lb = X_star.min(0)
ub = X_star.max(0)

noise = 0.0 

idx = np.random.choice(X_star.shape[0], N_u, replace=False)
X_u_train = X_star[idx, :]
S_train = S_star[idx, :]
I_train = I_star[idx, :]


layers = [2, 250, 250, 250, 250, 250, 250, 250, 2]
dinn = Inverse_problem(X_u_train, S_train, I_train, lb, ub, layers)
dinn.DINN.load_state_dict(torch.load('./saved_models/waitui_dnn_64.pth', map_location=device))
[dinn.beta, dinn.alpha] = torch.load('./saved_models/waitui_params_64.pt', map_location=device)

S_pred, I_pred = dinn.predict(X_star)
s_pred = griddata(X_star, S_pred.flatten(), (X, T), method='cubic')
i_pred = griddata(X_star, I_pred.flatten(), (X, T), method='cubic')

#%%
matplotlib.rcParams['font.size'] = 15
# Assume the first half of the data is used for training, and the second half for extrapolation
num_train = 0

# Calculate RMSE for each time step in the extrapolation period
rmse_S = []
rmse_I = []
for i in range(num_train, len(t)):
    rmse_S.append(np.sqrt(mean_squared_error(S_Exact[i, :], s_pred[i, :])))
    rmse_I.append(np.sqrt(mean_squared_error(I_Exact[i, :], i_pred[i, :])))

# Time steps for the extrapolation period
t_extrapolation = t[num_train:]

# Plot RMSE over time for S and I
plt.figure(figsize=(10, 6))
plt.plot(t_extrapolation, rmse_S, linewidth=4, label='RMSE of S')
plt.plot(t_extrapolation, rmse_I, linewidth=4, label='RMSE of I')
plt.xlabel('Time', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
plt.ylabel('RMSE', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
plt.yscale("log")
plt.title('RMSE of Extrapolation over Time', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
plt.legend()
plt.show()

# %%
fig_3 = plt.figure(1, figsize=(8, 5))

plt.plot(t, S_Exact[:, 50], "b-", linewidth=4, label="Exact")
plt.plot(t, s_pred[:, 50], "r--", linewidth=4, label="Prediction")
plt.plot(t, I_Exact[:, 50], "b-", linewidth=4, label="Exact")
plt.plot(t, i_pred[:, 50], "r--", linewidth=4, label="Prediction")
plt.axvline(x=10, color='k', linestyle='--', linewidth=4, label='t=10')
plt.title('$x=0.50$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})
plt.ylabel('$Value$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})
plt.xlabel('$t$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})



plt.tight_layout()
plt.show()

#%%

