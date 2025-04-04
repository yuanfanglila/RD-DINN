import numpy as np
import torch
from PhysicsInformedNN import Inverse_problem
from scipy.io import loadmat

np.random.seed(1234)
torch.manual_seed(1234)

if __name__ == '__main__':
    N_u = 666
    data = loadmat(r'../../data/SEIR_alpha_1.0.mat')

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

    noise = 0.00
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx, :]
    S_train = S_star[idx, :]
    E_train = E_star[idx, :]
    I_train = I_star[idx, :]
    S_train = S_train + noise * np.std(S_train) * np.random.randn(S_train.shape[0], S_train.shape[1])
    I_train = I_train + noise * np.std(I_train) * np.random.randn(I_train.shape[0], I_train.shape[1])
    E_train = E_train + noise * np.std(E_train) * np.random.randn(E_train.shape[0], E_train.shape[1])

    layers = [2, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 3]
    dinn = Inverse_problem(X_u_train, S_train, E_train, I_train, lb, ub, layers)
    iters = 10000
    dinn.train(iters)

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
    error_alpha = np.abs(alpha_value - 1) / 1 * 100
    print('Error S: %e' % (error_S))
    print('Error S: %e' % (error_E))
    print('Error I: %e' % (error_I))
    print('Error beta: %.3f%%' % (error_beta))
    print('Error xigema: %.3f%%' % (error_xigema))
    print('Error alpha: %.3f%%' % (error_alpha))

    print(beta_value, xigema_value, alpha_value)