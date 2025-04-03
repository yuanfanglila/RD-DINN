import numpy as np
from scipy.interpolate import griddata
from PhysicsInformedNN1 import Inverse_problem
import torch
import matplotlib
from scipy.io import loadmat
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    N_u = 1000
    data = loadmat('../data/SIR_beta=0.5_alpha=0.7.mat')

    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    I_Exact = np.real(data['u2'])
    S_Exact = np.real(data['u1'])

    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    S_star = S_Exact.flatten()[:,None]
    I_star = I_Exact.flatten()[:,None]

    lb = X_star.min(0)
    ub = X_star.max(0)

    noise = 0.0
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx, :]
    S_train = S_star[idx, :]
    I_train = I_star[idx, :]

    layers = [2, 100, 100, 100, 100, 100, 100, 100, 2]
    dinn = Inverse_problem(X_u_train, S_train, I_train, lb, ub, layers)
    iters = 500
    dinn.train(iters)

    S_pred, I_pred = dinn.predict(X_star)
    error_S = np.linalg.norm(S_star-S_pred,2)/np.linalg.norm(S_star,2)
    error_I = np.linalg.norm(I_star-I_pred,2)/np.linalg.norm(I_star,2)
    beta_value = dinn.beta.detach().cpu().numpy()
    alpha_value = dinn.alpha.detach().cpu().numpy()
    beta_value = beta_value
    alpha_value = np.exp(alpha_value)
    error_beta = np.abs(beta_value - 0.5) /0.5 * 100
    error_alpha = np.abs(alpha_value - 0.5) /0.5 * 100
    print('Error S: %e' % (error_S))
    print('Error I: %e' % (error_I))
    print('Error beta: %.3f%%' % (error_beta))
    print('Error alpha: %.3f%%' % (error_alpha))

    i_pred = griddata(X_star, I_pred.flatten(), (X, T), method='cubic')
    s_pred = griddata(X_star, S_pred.flatten(), (X, T), method='cubic')

    matplotlib.rcParams['font.size'] = 15
    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
    plt.pcolor(T, X, S_Exact, cmap='jet', norm=norm)

    plt.subplot(1, 3, 2)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
    plt.pcolor(T, X, s_pred, cmap='jet', norm=norm)
    random_indices = np.random.choice(len(X_u_train), size=1000, replace=True)
    X_u_train = X_u_train[random_indices]

    plt.subplot(1, 3, 3)
    norm = matplotlib.colors.Normalize(vmin=0.00, vmax=0.003)
    plt.pcolor(T, X, np.abs(S_Exact - s_pred), cmap='jet', norm=norm)
    plt.tight_layout()
    plt.show()
#
#     # %%
#     fig_2 = plt.figure(1, figsize=(18, 5))
#     plt.subplot(1, 3, 1)
#     norm = matplotlib.colors.Normalize(vmin=0, vmax=0.25)
#     plt.pcolor(T, X, I_Exact, cmap='jet', norm=norm)
#     plt.colorbar()
#     plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
#     plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
#     plt.title('Reference', fontdict={'fontsize': 30, 'fontfamily': 'serif'})
#
#     plt.subplot(1, 3, 2)
#     norm = matplotlib.colors.Normalize(vmin=0.0, vmax=0.25)
#     plt.pcolor(T, X, i_pred, cmap='jet', norm=norm)
#     random_indices = np.random.choice(len(X_u_train), size=1000, replace=True)
#     X_u_train = X_u_train[random_indices]
#     plt.plot(
#         X_u_train[:, 1],
#         X_u_train[:, 0],
#         'kx', label='Data (%d points)' % (I_train.shape[0]),
#         markersize=8,
#         clip_on=False,
#         alpha=0.8
#     )
#     plt.colorbar()
#     plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
#     plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
#     plt.title('Predicted', fontdict={'fontsize': 30, 'fontfamily': 'serif'})
#
#     plt.subplot(1, 3, 3)
#     norm = matplotlib.colors.Normalize(vmin=0.00, vmax=0.001)
#     plt.pcolor(T, X, np.abs(I_Exact - i_pred), cmap='jet', norm=norm)
#     plt.colorbar(format='%.0e')
#     plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
#     plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
#     plt.title('Absolute Error', fontdict={'fontsize': 30, 'fontfamily': 'serif'})
#
#     plt.tight_layout()
#     plt.show()
#
#     #%%
#     fig = plt.figure(figsize=(10, 6))
#     left, bottom, width, height = 0.11, 0.11, 0.8, 0.8
#     ax1 = fig.add_axes([left, bottom, width, height])
#     j = 2000
#     average_loss_total = [np.mean(dinn.losses[i:i + j]) for i in range(0, len(dinn.losses), j)]  ##(50,)
#     average_loss_D = [np.mean(dinn.loss_D[i:i + j]) for i in range(0, len(dinn.loss_D), j)]
#     average_loss_F = [np.mean(dinn.loss_F[i:i + j]) for i in range(0, len(dinn.loss_F), j)]

#     ax1.plot(average_loss_total, label='Total', color='teal')
#     ax1.plot(average_loss_D, label='Data', color='red')
#     ax1.plot(average_loss_F, label='Function', color='b')
#
#     plt.yscale('log')
#     ax1.set_xlabel('Epoch', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
#     ax1.set_ylabel('Relative $L^2$ Error', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
#
#     custom_ticks = [0, 5, 10, 15, 20, 25]
#     custom_labels = ['0', '1', '2', '3', '4', '5']
#     ax1.set_xticks(custom_ticks)
#     ax1.set_xticklabels(custom_labels)
#
#     plt.legend()
#     plt.show()
#
#     # %%
#     fig_3 = plt.figure(1, figsize=(19, 10))
#     plt.subplot(2, 3, 1)
#     plt.plot(t, S_Exact[:, 20], "b-", linewidth=4, label="Exact")
#     plt.plot(t, s_pred[:, 20], "r--", linewidth=4, label="Prediction")
#     plt.ylabel('$S(x,t)$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})
#     plt.title('$x=0.25$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})
#
#     plt.subplot(2, 3, 2)
#     plt.plot(t, S_Exact[:, 50], "b-", linewidth=4, label="Exact")
#     plt.plot(t, s_pred[:, 50], "r--", linewidth=4, label="Prediction")
#     plt.title('$x=0.50$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})
#
#     plt.subplot(2, 3, 3)
#     plt.plot(t, S_Exact[:, 75], "b-", linewidth=4, label="Exact")
#     plt.plot(t, s_pred[:, 75], "r--", linewidth=4, label="Prediction")
#     plt.title('$x=0.75$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})
#
#     plt.subplot(2, 3, 4)
#     plt.plot(t, I_Exact[:, 20], "b-", linewidth=4, label="Exact")
#     plt.plot(t, i_pred[:, 20], "r--", linewidth=4, label="Prediction")
#     plt.xlabel('$t$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})
#     plt.ylabel('$I(x,t)$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})
#
#     plt.subplot(2, 3, 5)
#     plt.plot(t, I_Exact[:, 50], "b-", linewidth=4, label="Exact")
#     plt.plot(t, i_pred[:, 50], "r--", linewidth=4, label="Prediction")
#     plt.xlabel('$t$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})
#
#     plt.subplot(2, 3, 6)
#     plt.plot(t, I_Exact[:, 75], "b-", linewidth=4, label="Exact")
#     plt.plot(t, i_pred[:, 75], "r--", linewidth=4, label="Prediction")
#     plt.xlabel('$t$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})
#
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
#                fancybox=True, shadow=True, ncol=4, fontsize=20)
#
#     plt.tight_layout()
#     plt.show()
#     #%%
#     fig = plt.figure(figsize=(10,6))
#     epoch = list(np.linspace(0,1300))
#     left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
#     ax1 = fig.add_axes([left, bottom, width, height])
#     ax1.plot(dinn.losses[200:], label='loss_total', color='teal')
#     ax1.plot(dinn.loss_D[200:], label='loss_data', color='red')
#     ax1.plot(dinn.loss_F[200:], label='loss_function', color='b')
#     plt.yscale('log')
#     ax1.set_xlabel('iterations')
#     ax1.set_ylabel('Loss')
#     ax1.set_ylim()
#     plt.legend()
#     plt.show()

#     #%%
#     fig_6 = plt.figure(figsize=(18,6))
#     plt.subplot(1, 2, 1)
#     iter = np.arange(iters)
#     beta_exact = np.full(iters, 0.5)
#     alpha_exact = np.full(iters, 0.5)
#     beta = np.array(dinn.beta_list)
#     alpha = np.array(dinn.alpha_list)
#     interval = 1000
#     avg_beta = [np.mean(beta[i:i + interval]) for i in range(0, len(beta), interval)]
#     avg_alpha = [np.mean(alpha[i:i + interval]) for i in range(0, len(alpha), interval)]
#     avg_iter = [np.mean(iter[i:i + interval]) for i in range(0, len(iter), interval)]
#
#     plt.plot(iter[:20000], beta_exact.T[:20000], "r--", label='beta_exact', linewidth=4, color='red')
#     plt.plot(avg_iter[:20], avg_beta[:20], label='beta_pred', linewidth=3, color='b')
#     plt.fill_between(avg_iter[:20], np.array(avg_beta[:20]) * (1.05), np.array(avg_beta[:20]) * (0.95), color='lightblue',
#                      alpha=0.7, label='Error Band')
#     plt.xlabel("Epoch", fontdict={'fontsize': 20, 'fontfamily': 'serif'})
#     plt.ylabel("Value", fontdict={'fontsize': 20, 'fontfamily': 'serif'})
#     plt.title("Predicted beta", fontdict={'fontsize': 30, 'fontfamily': 'serif'})
#     plt.grid(True)
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.plot(iter[:20000], alpha_exact.T[:20000], "r--", label='alpha_exact', linewidth=4, color='red')
#     plt.plot(avg_iter[:20], avg_alpha[:20], label='alpha_pred', linewidth=3, color='b')
#     plt.fill_between(avg_iter[:20], np.array(avg_alpha[:20]) * (1.05), np.array(avg_alpha[:20]) * (0.95), color='lightblue',
#                      alpha=0.7, label='Error Band')
#     plt.xlabel("Epoch", fontdict={'fontsize': 20, 'fontfamily': 'serif'})
#     plt.ylabel("Value", fontdict={'fontsize': 20, 'fontfamily': 'serif'})
#     plt.title("Predicted alpha", fontdict={'fontsize': 30, 'fontfamily': 'serif'})
#     plt.grid(True)
#     plt.legend()
#     plt.subplots_adjust(wspace=0.5)
#     plt.tight_layout()
#
#     plt.show()