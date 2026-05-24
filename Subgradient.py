#%%
import numpy as np
import pandas as pd


# f(x) = |x| 
def abs_calcfg(x):
    f = abs(x[0])
    g = np.array([np.sign(x[0])])
    if g[0] == 0:
        g[0] = 1.0
    return f, g

# f(x) = x1^2 + 100 x2^2
def quadratic_calfg(x):
    f = x[0]**2 + 100 * x[1]**2
    g = np.array([
        2 * x[0],
        200 * x[1]
    ])
    return f, g


# f(x) = |x1| + 10|x2|
def abstwo_calfg(x):
    f = np.abs(x[0]) + 10 * np.abs(x[1])
    g = np.array([np.sign(x[0]), 10 * np.sign(x[1])], dtype=float)
    g[g == 0] = 1.0   # choose one valid subgradient at 0
    return f, g



# epsilon-SVR 
def epsilon_svr_calcfg(w, X, y, eps, lam, method):

    n = X.shape[0]

    # residuals
    z = y - X @ w   

    # epsilon-insensitive loss
    loss = np.maximum(np.abs(z) - eps, 0)

    # ===== Subgradient of epsilon loss =====
    mask = np.abs(z) > eps
    g_eps = - X[mask].T @ np.sign(z[mask]) if np.any(mask) else np.zeros_like(w)

    # ===== Regularization =====
    if method == 2:  # L2
        f = (1/n) * np.sum(loss) + (lam/2) * np.sum(w**2)
        g = (1/n) * g_eps + lam * w

    elif method == 1:  # L1
        f = (1/n) * np.sum(loss) + lam * np.sum(np.abs(w))
        g_w = np.sign(w)
        g = (1/n) * g_eps + lam * g_w

    return f, g


# nu-SVR 
def nu_svr_calcfg(w, X, y, alpha, lam, method):
    n = X.shape[0]

    # residuals
    z = y - X @ w   

    # quantile threshold
    C = np.quantile(np.abs(z), alpha) 

    # loss
    loss = np.maximum(np.abs(z) - C, 0)

    # ===== Subgradient of nu-SVR loss =====
    mask = np.abs(z) > C
    g_c = - X[mask].T @ np.sign(z[mask]) if np.any(mask) else np.zeros_like(w)

    # ===== Regularization =====
    if method == 2:  # L2
        f = C*(1 - alpha) + (1/n) * np.sum(loss) + (lam/2) * np.sum(w[1:]**2)
        g = (1/n) * g_c + lam * np.hstack((0, w[1:]))

    elif method == 1:  # L1
        f = C*(1 - alpha) + (1/n) * np.sum(loss) + lam * np.sum(np.abs(w[1:]))
        g_w = np.sign(w)
        g_w[0] = 0
        g = (1/n) * g_c + lam * g_w

    return f, g


# regression error
# z(w) = y - Xw
#Err(z(w)) = max(E[max(0,z(w))],E[max(0,-z(w))]), E = 1/n sum_i=1^n
#          |  (-X_i if z_i(w)>0) (0 if z_i(w)<=0),         E[max(0,z(w))] > E[max(0,-z(w))])
#          |
# g_i(w) = <  (X_i if z_i(w)<0) (0 if z_i(w)>=0),          E[max(0,z(w))]) < E[max(0,-z(w))])
#          |
#          |  0,                                           E[max(0,z(w))]) = E[max(0,-z(w))])
# g(w) = 1/n sum_{i=1}^n g_i(w).
def error_calcfg(w, X, y):
    n = X.shape[0]
    # residuals
    z = y - X @ w   
    
    # subgradient
    g_c = np.zeros_like(w)

    # expectations 
    E_pos = np.mean(np.maximum(0,z))
    E_neg = np.mean(np.maximum(0,-z))

    for i in range(n):
        if E_pos > E_neg:
            if z[i] > 0:
                g_c += -X[i]
            #elif z[i] <= 0:
                #g_c += 0
        elif E_pos < E_neg:
            if z[i] < 0:
                g_c += X[i]
            #elif z[i] >= 0:
                #g_c += 0    
        else:
                g_c += 0 

    f = max(E_pos, E_neg)
    g = (1/n) * g_c

    return f, g 

# CVaR portfolio optimization
def proj_simplex(w):
    u = np.sort(w)[::-1] # sort the weights desc order. 
    cssv = np.cumsum(u) # sum of the weights 
    k = np.where(u - (cssv - 1) / (np.arange(len(w)) + 1) > 0)[0][-1] # find th biggest feasible k， where k = 1,2,3,...n
    lam = (cssv[k] - 1) / (k + 1) # calculate the lambda by using this k 
    return np.maximum(w - lam, 0) 

def cvaropt_calcfg(w, X, lam=1.0, alpha=0.95, M=1e3):
    n = X.shape[0]
    L = - X @ w
    E_loss = np.mean(L)

    # VaR (quantile)
    C = np.quantile(L, alpha)

    #CVaR 
    excess = np.maximum(L - C, 0)  # (L - C)+
    cvar = C + (1/(1 - alpha)) * np.mean(excess)   

    f = E_loss + lam * cvar

    #subgradient
    # mean part E(L(w))
    g_EL = np.zeros_like(w)
    for i in range(n):
        g_EL += -X[i]

    # CVaR part lambda * CVaR(L(w))
    g_cvar = np.zeros_like(w)
    for i in range(n):
        if L[i] > C:
            g_cvar += -X[i]

    proj_term = M * (w-proj_simplex(w)) / np.linalg.norm(w-proj_simplex(w)) if np.linalg.norm(w-proj_simplex(w)) != 0 else np.zeros_like(w)  # projection term to ensure w in simplex
    g = (1/n) * g_EL + lam * (1/((1 - alpha)*n)) * g_cvar + proj_term


    return f, g


# bad conditional convex quadratic function (Quad) from the book 
def bad_quad_calcfg(x):
    x = np.asarray(x, dtype=float)
    f = sum(10**(i-1) * x[i-1]**2 for i in range(1, len(x)+1))
    g = np.array([2 * 10**(i-1) * x[i-1] for i in range(1, len(x)+1)])
    return f, g


# convex piece-wise linear function (PWL) from the book
def pwl_calcfg(x):
    x = np.asarray(x, dtype=float)
    f = np.sum(10**(i-1) * abs(x[i-1]) for i in range(1, len(x)+1))
    g = np.array([10**(i-1) * np.sign(x[i-1]) for i in range(1, len(x)+1)], dtype=float)
    return f, g


# Rozenbrock function 
def rozenbrock_calcfg(x):
    x = np.asarray(x, dtype=float)
    f = (x[0]-1)**2 + 100*np.sum((x[i] - x[i-1]**2)**2 for i in range(1, len(x)))
    g = np.zeros_like(x)
    g[0] = 2 * (x[0] - 1) - 400 * x[0] * (x[1] - x[0]**2)
    for i in range(1, len(x)-1):
        g[i] = 200 * (x[i] - x[i-1]**2) - 400 * x[i] * (x[i+1] - x[i]**2)
    g[-1] = 200 * (x[-1] - x[-2]**2)
    
    return f, g


# Relative Entropy Minimization
def entropyr_proj(X, l=1e-13, u=1):
    eigvals, Q = np.linalg.eigh(X)
    eigvals_clipped = np.clip(eigvals, l, u)
    return Q @ np.diag(eigvals_clipped) @ Q.T

def entropyr_calcfg(x, Y, M):
    """
    f(X) = log det(X) - trace(XY)
    """
    X = x.reshape(Y.shape)  # Reshape x to the original matrix shape
    sign, logdet = np.linalg.slogdet(X)
    #if sign <= 0:
        #return np.inf
    f = logdet - np.trace(X @ Y)

    #subgradient
    """
    g(X) = X^{-1}^T - Y^T
    """
    proj_term_entro = M * (X-entropyr_proj(X)) / np.linalg.norm(X-entropyr_proj(X)) if np.linalg.norm(X-entropyr_proj(X)) != 0 else np.zeros_like(X)
    g = np.linalg.inv(X).transpose() - Y.transpose() + proj_term_entro
    return f, g


def beale_calcfg(x):
    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    f = (1.5 - x1 + x1 * x2)**2 + (2.25 - x1 + x1 * x2**2)**2 + (2.625 - x1 + x1 * x2**3)**2
    g1 = 2 * (1.5 - x1 + x1 * x2) * (-1 + x2) + 2 * (2.25 - x1 + x1 * x2**2) * (-1 + x2**2) + 2 * (2.625 - x1 + x1 * x2**3) * (-1 + x2**3)
    g2 = 2 * (1.5 - x1 + x1 * x2) * (x1) + 4 * (2.25 - x1 + x1 * x2**2) * (x1 * x2) + 6 * (2.625 - x1 + x1 * x2**3) * (x1 * x2**2)
    g = np.array([g1, g2])
    return f, g


def entropy_dual_calcfg(z, A, b, p, l, u):
    """
    Dual entropy problem.

    z = [alpha, lambda_1, ..., lambda_m]

    Dual problem:
        maximize L(x*, alpha, lambda)

    Since r-alg / Adam usually minimize, return:
        f = -dual_value
        g = -dual_gradient
    """

    z = np.asarray(z, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    p = np.asarray(p, dtype=float)
    l = np.asarray(l, dtype=float)
    u = np.asarray(u, dtype=float)

    alpha = z[0]
    lamb = z[1:]      # length m

    # x_hat_j = p_j exp(-1 - alpha - lambda^T a_col^j)
    x_hat = p * np.exp(-1 - alpha - A.T @ lamb)

    # x_star_j = projection of x_hat_j onto [l_j, u_j]
    x_star = np.clip(x_hat, l, u) # np.clip(x, lower_bound, upper_bound)

    # L(x*, alpha, lambda)
    eps = 1e-12
    x_safe = np.maximum(x_star, eps) # avoid log(0) by ensuring x_star is at least eps

    t1 = np.sum(x_safe * np.log(x_safe / p))
    t2 = alpha * (np.sum(x_star) - 1)
    t3 = lamb @ (A @ x_star - b)

    dual_value = t1 + t2 + t3

    # gradient of dual objective
    # by envelope theorem:
    grad_alpha = np.sum(x_star) - 1
    grad_lamb = A @ x_star - b

    grad_dual = np.concatenate([[grad_alpha], grad_lamb])

    # minimize negative dual
    f = -dual_value
    g = -grad_dual

    return f, g

# %%
