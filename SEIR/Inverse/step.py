import numpy as np
from scipy.interpolate import griddata
import torch
from PhysicsInformedNN import Inverse_problem, savefig
import matplotlib
from scipy.io import loadmat
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 25})
# plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 使用微软雅黑字体
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

np.random.seed(1234)
torch.manual_seed(1234)

if __name__ == '__main__':
    N_u = 3000
    data = loadmat('D:\代码\python\扩散SIR\潜伏SEIR\SEIR方程解.mat')

    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    S_Exact = np.real(data['u1'])
    E_Exact = np.real(data['u2'])
    I_Exact = np.real(data['u3'])

    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    S_star = S_Exact.flatten()[:, None]
    E_star = E_Exact.flatten()[:, None]
    I_star = I_Exact.flatten()[:, None]

    lb = X_star.min(0)
    ub = X_star.max(0)

    noise = 0.0 # 噪声设置，随机扰动项
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx, :]
    S_train = S_star[idx, :]
    E_train = E_star[idx, :]
    I_train = I_star[idx, :]
    #%%
    layers = [2, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 3]
    dinn = Inverse_problem(X_u_train, S_train, E_train, I_train, lb, ub, layers)
    iters = 100000
    dinn.train(iters)
    #%%
    S_pred, E_pred, I_pred = dinn.predict(X_star)
    error_S = np.linalg.norm(S_star - S_pred, 2) / np.linalg.norm(S_star, 2)
    error_E = np.linalg.norm(E_star - E_pred, 2) / np.linalg.norm(E_star, 2)
    error_I = np.linalg.norm(I_star - I_pred, 2) / np.linalg.norm(I_star, 2)
    beta_value = dinn.beta.detach().cpu().numpy()
    xigema_value = dinn.xigema.detach().cpu().numpy()
    alpha_value = dinn.alpha.detach().cpu().numpy()
    beta_value = np.exp(beta_value)
    xigema_value = np.exp(xigema_value)
    alpha_value = np.exp(alpha_value)
    error_beta = np.abs(beta_value - 1) / 1 * 100
    error_xigema = np.abs(xigema_value - 1) / 1 * 100
    error_alpha = np.abs(alpha_value - 1) /1 * 100
    print('Error S: %e' % (error_S))
    print('Error S: %e' % (error_E))
    print('Error I: %e' % (error_I))
    print('Error beta: %.3f%%' % (error_beta))
    print('Error xigema: %.3f%%' % (error_xigema))
    print('Error alpha: %.3f%%' % (error_alpha))
    #%%
    s_pred = griddata(X_star, S_pred.flatten(), (X, T), method='cubic')
    e_pred = griddata(X_star, E_pred.flatten(), (X, T), method='cubic')
    i_pred = griddata(X_star, I_pred.flatten(), (X, T), method='cubic')
    matplotlib.rcParams.update({'font.size': 25})
    #%%
    print(beta_value, xigema_value, alpha_value)
    # %%
    matplotlib.rcParams['font.size'] = 15
    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(T, X, S_Exact, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.title('Reference', fontdict={'fontsize': 30, 'fontfamily': 'serif'})

    plt.subplot(1, 3, 2)
    plt.pcolor(T, X, s_pred, cmap='jet')
    random_indices = np.random.choice(len(X_u_train), size=100, replace=True)
    X_u_train = X_u_train[random_indices]
    plt.plot(
        X_u_train[:, 1],
        X_u_train[:, 0],
        'kx', label='Data (%d points)' % (I_train.shape[0]),
        markersize=8,
        clip_on=False,
        alpha=0.8
    )
    plt.colorbar()
    plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.title('Predicted', fontdict={'fontsize': 30, 'fontfamily': 'serif'})

    plt.subplot(1, 3, 3)
    norm = matplotlib.colors.Normalize(vmin=0.00, vmax=0.001)
    plt.pcolor(T, X, np.abs(S_Exact - s_pred), cmap='jet', norm=norm)
    plt.colorbar(format='%.0e')
    plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.title('Absolute Error', fontdict={'fontsize': 30, 'fontfamily': 'serif'})

    plt.tight_layout()
    plt.show()


    # %%
    fig_2 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(T, X, E_Exact, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.title('Reference', fontdict={'fontsize': 30, 'fontfamily': 'serif'})

    plt.subplot(1, 3, 2)
    plt.pcolor(T, X, e_pred, cmap='jet')
    random_indices = np.random.choice(len(X_u_train), size=100, replace=True)
    X_u_train = X_u_train[random_indices]
    plt.plot(
        X_u_train[:, 1],
        X_u_train[:, 0],
        'kx', label='Data (%d points)' % (I_train.shape[0]),
        markersize=8,
        clip_on=False,
        alpha=0.8
    )
    plt.colorbar()
    plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.title('Predicted', fontdict={'fontsize': 30, 'fontfamily': 'serif'})

    plt.subplot(1, 3, 3)
    norm = matplotlib.colors.Normalize(vmin=0.00, vmax=0.001)
    plt.pcolor(T, X, np.abs(E_Exact - e_pred), cmap='jet', norm=norm)
    plt.colorbar(format='%.0e')
    plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.title('Absolute Error', fontdict={'fontsize': 30, 'fontfamily': 'serif'})

    plt.tight_layout()
    plt.show()


    # %%
    fig_3 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(T, X, I_Exact, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.title('Reference', fontdict={'fontsize': 30, 'fontfamily': 'serif'})

    plt.subplot(1, 3, 2)
    plt.pcolor(T, X, i_pred, cmap='jet')
    random_indices = np.random.choice(len(X_u_train), size=100, replace=True)
    X_u_train = X_u_train[random_indices]
    plt.plot(
        X_u_train[:, 1],
        X_u_train[:, 0],
        'kx', label='Data (%d points)' % (I_train.shape[0]),
        markersize=8,
        clip_on=False,
        alpha=0.8
    )
    plt.colorbar()
    plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.title('Predicted', fontdict={'fontsize': 30, 'fontfamily': 'serif'})

    plt.subplot(1, 3, 3)
    norm = matplotlib.colors.Normalize(vmin=0.00, vmax=0.001)
    plt.pcolor(T, X, np.abs(I_Exact - i_pred), cmap='jet', norm=norm)
    plt.colorbar(format='%.0e')
    plt.xlabel(r'$t$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.ylabel(r'$x$', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.title('Absolute Error', fontdict={'fontsize': 30, 'fontfamily': 'serif'})

    plt.tight_layout()
    plt.show()

    # %%
    fig_4 = plt.figure(1, figsize=(19, 15))
    plt.subplot(3, 3, 1)
    plt.plot(t, S_Exact[:, 25], "b-", linewidth=4, label="Exact")
    plt.plot(t, s_pred[:, 25], "r--", linewidth=4, label="Prediction")
    plt.ylabel('$S(x,t)$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})
    plt.title('$x=0.25$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})

    plt.subplot(3, 3, 2)
    plt.plot(t, S_Exact[:, 50], "b-", linewidth=4, label="Exact")
    plt.plot(t, s_pred[:, 50], "r--", linewidth=4, label="Prediction")
    plt.title('$x=0.50$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})

    plt.subplot(3, 3, 3)
    plt.plot(t, S_Exact[:, 75], "b-", linewidth=4, label="Exact")
    plt.plot(t, s_pred[:, 75], "r--", linewidth=4, label="Prediction")
    plt.title('$x=0.75$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})

    plt.subplot(3, 3, 4)
    plt.plot(t, E_Exact[:, 25], "b-", linewidth=4, label="Exact")
    plt.plot(t, e_pred[:, 25], "r--", linewidth=4, label="Prediction")
    plt.xlabel('$t$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})
    plt.ylabel('$E(x,t)$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})

    plt.subplot(3, 3, 5)
    plt.plot(t, E_Exact[:, 50], "b-", linewidth=4, label="Exact")
    plt.plot(t, e_pred[:, 50], "r--", linewidth=4, label="Prediction")
    plt.xlabel('$t$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})

    plt.subplot(3, 3, 6)
    plt.plot(t, E_Exact[:, 75], "b-", linewidth=4, label="Exact")
    plt.plot(t, e_pred[:, 75], "r--", linewidth=4, label="Prediction")
    plt.xlabel('$t$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})

    plt.subplot(3, 3, 7)
    plt.plot(t, I_Exact[:, 25], "b-", linewidth=4, label="Exact")
    plt.plot(t, i_pred[:, 25], "r--", linewidth=4, label="Prediction")
    plt.xlabel('$t$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})
    plt.ylabel('$I(x,t)$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})

    plt.subplot(3, 3, 8)
    plt.plot(t, I_Exact[:, 50], "b-", linewidth=4, label="Exact")
    plt.plot(t, i_pred[:, 50], "r--", linewidth=4, label="Prediction")
    plt.xlabel('$t$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})

    plt.subplot(3, 3, 9)
    plt.plot(t, I_Exact[:, 75], "b-", linewidth=4, label="Exact")
    plt.plot(t, i_pred[:, 75], "r--", linewidth=4, label="Prediction")
    plt.xlabel('$t$', fontdict={'fontsize': 30, 'fontfamily': 'serif'})

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
               fancybox=True, shadow=True, ncol=4, fontsize=20)

    plt.tight_layout()
    plt.show()

    #%%
    fig = plt.figure(figsize=(10, 6))
    left, bottom, width, height = 0.11, 0.11, 0.8, 0.8
    ax1 = fig.add_axes([left, bottom, width, height])
    j = 2000
    average_loss_total = [np.mean(dinn.losses[i:i + j]) for i in range(0, len(dinn.losses), j)]  ##(50,)
    average_loss_D = [np.mean(dinn.loss_D[i:i + j]) for i in range(0, len(dinn.loss_D), j)]
    average_loss_F = [np.mean(dinn.loss_F[i:i + j]) for i in range(0, len(dinn.loss_F), j)]
    ax1.plot(average_loss_total, label='Total', color='teal')
    ax1.plot(average_loss_D, label='Data', color='red')
    ax1.plot(average_loss_F, label='Function', color='b')

    plt.yscale('log')
    ax1.set_xlabel('Epoch', fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    ax1.set_ylabel('Relative $L^2$ Error', fontdict={'fontsize': 20, 'fontfamily': 'serif'})

    # 自定义 x 轴刻度和标签
    custom_ticks = [0, 10, 20, 30, 40, 50]  # 自定义刻度位置
    custom_labels = ['0', '1', '2', '3', '4', '5']  # 自定义刻度标签
    ax1.set_xticks(custom_ticks)
    ax1.set_xticklabels(custom_labels)

    plt.legend()
    plt.show()
    # %%
    fig_8 = plt.figure(figsize=(27, 6))
    iter = np.arange(iters)
    beta_exact = np.full(iters, 1.0)
    alpha_exact = np.full(iters, 1.0)
    sigma_exact = np.full(iters, 1.0)
    beta = np.array(dinn.beta_list)
    alpha = np.array(dinn.alpha_list)
    sigma = np.array(dinn.xigema_list)
    interval = 10
    avg_beta = [np.mean(beta[i:i + interval]) for i in range(0, len(beta), interval)]
    avg_alpha = [np.mean(alpha[i:i + interval]) for i in range(0, len(alpha), interval)]
    avg_sigma = [np.mean(sigma[i:i + interval]) for i in range(0, len(sigma), interval)]
    avg_iter = [np.mean(iter[i:i + interval]) for i in range(0, len(iter), interval)]

    plt.subplot(1, 3, 1)
    plt.plot(iter[:5000], beta_exact.T[:5000], "r--", label='beta_exact', linewidth=4, color='red')
    plt.plot(avg_iter[:500], avg_beta[:500], label='beta_pred', linewidth=3, color='b')  # 取前20个间隔
    plt.fill_between(avg_iter[:500], np.array(avg_beta[:500]) * (1.001), np.array(avg_beta[:500]) * (0.999),
                     color='lightblue',
                     alpha=0.7, label='Error Band')
    plt.xlabel("Epoch", fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.ylabel("Value", fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.title("Predicted beta", fontdict={'fontsize': 30, 'fontfamily': 'serif'})
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(iter[:5000], alpha_exact.T[:5000], "r--", label='sigma_exact', linewidth=4, color='red')
    plt.plot(avg_iter[:500], avg_sigma[:500], label='sigma_pred', linewidth=3, color='b')  # 取前20个间隔
    plt.fill_between(avg_iter[:500], np.array(avg_sigma[:500]) * (1.001), np.array(avg_sigma[:500]) * (0.999),
                     color='lightblue',
                     alpha=0.7, label='Error Band')
    plt.xlabel("Epoch", fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.ylabel("Value", fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.title("Predicted sigma", fontdict={'fontsize': 30, 'fontfamily': 'serif'})
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(iter[:5000], alpha_exact.T[:5000], "r--", label='alpha_exact', linewidth=4, color='red')
    plt.plot(avg_iter[:500], avg_alpha[:500], label='alpha_pred', linewidth=3, color='b')  # 取前20个间隔
    plt.fill_between(avg_iter[:500], np.array(avg_alpha[:500]) * (1.001), np.array(avg_alpha[:500]) * (0.999),
                     color='lightblue',
                     alpha=0.7, label='Error Band')
    plt.xlabel("Epoch", fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.ylabel("Value", fontdict={'fontsize': 20, 'fontfamily': 'serif'})
    plt.title("Predicted alpha", fontdict={'fontsize': 30, 'fontfamily': 'serif'})
    plt.grid(True)
    plt.legend()


    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()

    plt.show()

