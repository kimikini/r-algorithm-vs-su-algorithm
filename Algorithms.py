#%%
import numpy as np
import pandas as pd


# Projection
# find a labda that staticfy sum of np.maximum(w - lambda, 0) = 1  which is sum of weights = 1 
def proj_simplex(w):
    u = np.sort(w)[::-1] # sort the weights desc order. 
    cssv = np.cumsum(u) # sum of the weights 
    k = np.where(u - (cssv - 1) / (np.arange(len(w)) + 1) > 0)[0][-1] # find th biggest feasible k， where k = 1,2,3,...n
    lam = (cssv[k] - 1) / (k + 1) # calculate the lambda by using this k 
    return np.maximum(w - lam, 0) 


# B-form r-algorithm  
def ralgb5a(calcfg, x0, alpha=2.5,
    h0= 1.0,
    q1= 0.9,
    q2 = 1.2,
    nh = 2.5,
    epsg=1e-8,
    epsx=1e-8,
    maxitn=1000,
    intp=50
    
    ):
    """
    B-form r-algorithm with adaptive step (ralgb5 type)
    Parameters
    ----------
    calcfg :Function that returns (f, g) given x
    x0 : Initial starting point
    alpha :  Space dilation coefficient (alpha > 0)
    h0 : Initial step size
    q1 <= 1: Step shrinking factor
    q2 >= 1: step expand factor
    epsg : Stopping tolerance for gradient norm
    epsx : Stopping tolerance for step size
    maxitn :  Maximum number of iterations
    intp : Print info every intp iterations

    Returns
    -------
    xr : best point found
    fr : best function value
    itn : number of iterations used
    nfg : number of function evaluations
    istop : stopping code
        2 = gradient small
        3 = step small
        4 = max iterations reached
        5 = line search error
    """
    # history
    history = {'f':[], 'fr(local opt)':[], 'x':[], 'B':[]}  # here x is trajectory

    # Current iterate (copy to avoid modifying original input)
    x = x0.copy()
    #if projection:
        #x = proj_simplex(x0.copy())
    
    
    # Dimension of variable space
    n = len(x)

    # Initialize transformation matrix B = I (identity matrix)
    # This corresponds to B0 in the paper
    B = np.eye(n)
    history['B'].append(B)

    # Initialize step size
    hs = h0

    # Counters for line search statistics
    lsa = 0   # accumulated line search steps
    lsm = 0   # max line search steps

    # Best solution found so far
    xr = x.copy()
    history['x'].append(xr)
    # Evaluate function and gradient at initial point， f is the function, g0 is the gradient, this is y? 
    fr, g0 = calcfg(x)
    f = fr
    history['f'].append(fr)

    # Count function evaluations
    nfg = 1

    # Print initial information
    print(f"itn {0:4d}  f {fr:15.6e}  fr {fr:15.6e}  nfg {nfg:4d}")

    # Stopping condition: gradient norm small  
    if np.linalg.norm(g0) < epsg:
        return xr, fr, 0, nfg,  2, history

    # =======================
    # Main iteration loop (do normal subgradident descent)
    # =======================
    for itn in range(1, maxitn + 1):  # allowing how many times greaident update and space dilation update 

        # Compute transformed gradient to current space: B^T g_k  
        g_trans = B.T @ g0

        # Normalize transformed gradient
        g_trans_norm = np.linalg.norm(g_trans)

        # Descent direction:
        # dx = B * (B^T g_k / ||B^T g_k||)
        # This is normalized anti-subgradient in transformed space
        xi_k = (g_trans / g_trans_norm)  #xi_k
        dx = B @ xi_k  #B_k @ xi_k
        

        # Initialize line search variables
        d = 1          # control variable for descent
        ls = 0         # number of line search steps
        ddx = 0        # total step movement

        # =======================
        # Adaptive line search to find a good hk (step size)  not need to be the best   hk >= hk*
        # =======================
        while d > 0:  # if true keep looping

            # Move along descent direction  xk+1 = xk - hs * dx , x1 is drawn by using the steepest descent direction
            x = x - hs * dx
            #if projection:
                #x = proj_simplex(x)
            history['x'].append(x)
            
            # Accumulate movement size
            ddx += hs * np.linalg.norm(dx)

            # Evaluate function and gradient at new point
           
            f, g1 = calcfg(x)
            nfg += 1
            history['f'].append(f)

            # Update best solution if improved
            if f < fr:
                fr = f
                xr = x.copy()
            history['fr(local opt)'].append(fr)

            # Stop if gradient small
            if np.linalg.norm(g1) < epsg:
                return xr, fr, itn, nfg, 2, history

            # Increase line search counter
            ls += 1

            # Every nh steps, slightly increase step size by q2
            if ls % nh == 0:
                hs *= q2

            # Emergency stop if too many LS steps （line search failed too many steps) so 500 is the maximum line search steps allowed
            if ls > 500:  #k_max = 500
                return xr, fr, itn, nfg, 5, history
                # break

            # Check descent condition:
            # d = dx^T g_{k+1} pg700
            # calculate the new d 
            # If d <= 0, descent condition satisfied so stop, the step size is good enough for current x 
            # If d > 0, keep looping to find a step size such that the direction becomes a descent direction.
            d = dx @ g1
            # d = f_prev - f  # this is the sufficient decrease condition, if not satisfied, keep shrinking step size

        # If descent finished in one step, shrink step size
        #在 while 里找到一个“合适的 hk”
        #如果 ls = 1，就把 hk 稍微缩小一点
        #但 不会重新做这次 line search
        if ls == 1:
            hs *= q1

        # Update LS statistics
        #ls	当前这一次 line search 走了多少步 第一个iteration k=1, ls = 2,  k=2, ls=3
        #lsa	line search steps accumulated（累计总步数） 走到d不满足loop条件停下来，算lsa = 2+3 = 5 
        #lsm	line search max（当前区间内最大步数）   看最差（最多次line search）的记录
        lsa += ls
        lsm = max(lsm, ls)

        # Print progress every intp iterations, uf intp = 10, 每10次打印一次迭代信息
        if itn % intp == 0:
            print(f"itn {itn:4d}  f {f:14.6e}  fr {fr:14.6e}  "
                  f"nfg {nfg:4d}  lsa {lsa:3d}  lsm {lsm:3d}")
            lsa = 0
            lsm = 0

        # Stop if movement between two steps are small, ddx is the total step movement accumulated during line search
        if ddx < epsx:  # ddx = xk+1 - xk 
            return xr, fr, itn, nfg, 3, history

        # =======================
        # Space dilation update (updates B every iteration, but meaningful space dilation only occurs when the gradient differences have a consistent direction)
        # =======================
        # Compute gradient difference in transformed space:
        # dg = B^T (g_{k+1} - g_k), rk = g_{k+1} - g_k
        dg = B.T @ (g1 - g0)

        # Normalize direction xi
        xi = dg / np.linalg.norm(dg)

        # Update B matrix:
        # B_{k+1} = B_k + (1/alpha - 1) * (B_k xi) xi^T
        # This implements:
        # R_k = I + (1/alpha - 1) xi xi^T
        # B_{k+1} = B_k R_k
        B = B + (1/alpha - 1) * np.outer(B @ xi, xi)  # the space dilation operator formula
        history['B'].append(B)
        
        # Update gradient for next iteration
        g0 = g1

    # Max iterations reached
    return xr, fr, maxitn, nfg, 4, history


#**********************************************************************************************************************************************************

# su-algorithm  
def sualg(calcfg, x0, 
    h0=1.0,
    q1=0.9,
    q2 = 1.5,
    nh = 2.5,
    epsg=1e-8,
    epsx=1e-8,
    maxitn=1000,
    intp=50,
    theta = 0.001):
    """
    Parameters
    ----------
    calcfg :Function that returns (f, g) given x
    x0 :Initial starting point
    theta: step size of combination of B-update and x-update # bstep in paper recommend 0.65, (0<theta<1)
    h0 : Initial step size
    q1 <= 1 : Step shrinking factor
    q2 >= 1: step expand factor
    nh : Number of steps to increase step size by q2
    epsg : Stopping tolerance for gradient norm
    epsx : Stopping tolerance for step size
    maxitn :Maximum number of iterations
    intp : Print info every intp iterations
    flevel :Target function value; algorithm stops if f(x) ≤ flevel    New!!!

    Returns
    -------
    xr : best point found
    fr : best function value
    itn : number of iterations used
    nfg : number of function evaluations
    istop : stopping code
        1 = function value reached flevel
        2 = gradient norm is samll small (||g|| < epsg)
        3 = step size is small  (||x_{k+1} - x_k|| < epsx)
        4 = max iterations reached 
        5 = line search error
    """
    
    # history
    history = {'f':[], 'fr(local opt)':[], 'x':[], 'B':[]}

    # Current iterate (copy to avoid modifying original input)
    x = x0.copy()
    #if projection == True:
    #    x = proj_simplex(x0.copy())
    
    # Dimension of variable space
    n = len(x)

    # Initialize transformation matrix B = I (identity matrix)
    # This corresponds to B0 in the paper
    B = np.eye(n)
    history['B'].append(B)
    # Initialize step size
    hs = h0

    # Counters for line search statistics
    lsa = 0   # accumulated line search steps
    lsm = 0   # max line search steps

    # Best solution found so far
    xr = x.copy()
    history['x'].append(xr)
    # Evaluate function and gradient at initial point， f is the function, g0 is the gradient, this is y? 
    fr, g0 = calcfg(x)
    history['f'].append(fr)

    # Count function evaluations
    nfg = 1

    # Print initial information
    print(f"itn {0:4d}  f {fr:15.6e}  fr {fr:15.6e}  nfg {nfg:4d}")

    # Stopping condition: gradient norm small  
    if np.linalg.norm(g0) < epsg:
        return xr, fr, 0, nfg,  2, history

    # =======================
    # Main iteration loop (do normal subgradident descent)
    # =======================
    for itn in range(1, maxitn + 1):

        # Compute transformed gradient to current space: B^T gf_(xk)  
        g_trans = B.T @ g0

        # Normalize transformed gradient
        g_trans_norm = np.linalg.norm(g_trans)

        # Descent direction:
        # dx = B * (B^T g_k / ||B^T g_k||)
        # This is normalized anti-subgradient in transformed space
        xi_k = (g_trans / g_trans_norm)  #xi_k
        dx = B @ xi_k  #B_k @ xi_k

        # Initialize line search variables
        d = 1          # control variable for descent
        ls = 0         # number of line search steps
        ddx = 0        # total step movement

        # =======================
        # Adaptive line search to find a good hk (step size)  not need to be the best   hk >= hk*
        # =======================
        while d > 0:  # if true keep looping

            # Move along descent direction  xk+1 = xk - hs * dx  update x
            x = x - hs * dx
            #if projection:
            #    x = proj_simplex(x)
            history['x'].append(x)
            
            # Accumulate movement size
            ddx += hs * np.linalg.norm(dx)

            # Evaluate function and gradient at new point
            f, g1 = calcfg(x)
            nfg += 1
            history['f'].append(f)

            # Update best solution if improved
            if f < fr:
                fr = f
                xr = x.copy()
            history['fr(local opt)'].append(fr)
            
            # Stop if gradient small
            if np.linalg.norm(g1) < epsg:
                return xr, fr, itn, nfg, 2, history

            # Increase line search counter
            ls += 1

            # Every nh steps, slightly increase step size by q2
            if ls % nh == 0:
                hs *= q2

            # Emergency stop if too many LS steps （line search failed too many steps)
            if ls > 500:
                return xr, fr, itn, nfg, 5, history

            # Check descent condition:
            # d = dx^T g_{k+1} pg700
            # calculate the new d 
            # If d <= 0, descent condition satisfied so stop, the step size is good enough for current x 
            # If d > 0, keep looping to find a step size such that the direction becomes a descent direction.
            d = dx @ g1

        # If descent finished in one step, shrink step size
        #在 while 里找到一个“合适的 hk”
        #如果 ls = 1，就把 hk 稍微缩小一点
        #但 不会重新做这次 line search
        if ls == 1:
            hs *= q1

        # Update LS statistics
        #ls	当前这一次 line search 走了多少步 第一个iteration k=1, ls = 2,  k=2, ls=3
        #lsa	line search steps accumulated（累计总步数） 走到d不满足loop条件停下来，算lsa = 2+3 = 5 
        #lsm	line search max（当前区间内最大步数）   看最差（最多次line search）的记录
        lsa += ls
        lsm = max(lsm, ls)

        # Print progress every intp iterations, uf intp = 10, 每10次打印一次迭代信息
        if itn % intp == 0:
            print(f"itn {itn:4d}  f {f:14.6e}  fr {fr:14.6e}  "
                  f"nfg {nfg:4d}  lsa {lsa:3d}  lsm {lsm:3d}")
            lsa = 0
            lsm = 0

        # Stop if movement between two steps are small, ddx is the total step movement accumulated during line search
        if ddx < epsx:  # ddx = xk+1 - xk 
            return xr, fr, itn, nfg, 3, history

        # =======================
        # updates B every iteration, but meaningful space dilation only occurs when the gradient differences have a consistent direction)

        # Normalize direction xi
        g1_norm = np.linalg.norm(g1)
        if g1_norm < epsg:
            return xr, fr, itn, nfg, 2, history

        xi = g1 / g1_norm
 

        # Update B matrix:
        # B_{k+1} = B_k + gamma_si * (xi * g_k^T  + g_k * xi^T) * B_k
        # same space, but change the direction of the gradient (transform the gradient by using B)
        B = B +  theta * (np.outer(xi, g0) + np.outer(g0, xi)) @ B
        history['B'].append(B)

        # Update gradient for next iteration
        g0 = xi
        
    # Max iterations reached
    return xr, fr, maxitn, nfg, 4, history







#**********************************************************************************************************************************************************
# r-alg save memory 
def eye_with_tail_pivot(n, m, pivot_col=None):
    if pivot_col is None:
        pivot_col = m - 1 
    
    A = np.zeros((n, m))
    
    # 前 m 行：identity
    for i in range(min(n, m)):
        A[i, i] = 1
    
    # 后面的行：全放在 pivot_col
    for i in range(m, n):
        A[i, pivot_col] = 1
    
    return A

# B-form r-algorithm  
def ralgb5a_m(calcfg, x0, alpha=2.5,
    h0= 1.0,
    q1= 0.9,
    q2 = 1.5,
    nh = 3,
    epsg=1e-8,
    epsx=1e-8,
    maxitn=1000,
    intp=50,
    projection=False,
    m=5):
    """
    B-form r-algorithm with adaptive step (ralgb5 type)
    Parameters
    ----------
    calcfg :Function that returns (f, g) given x
    x0 : Initial starting point
    alpha :  Space dilation coefficient (alpha > 0)
    h0 : Initial step size
    q1 <= 1: Step shrinking factor
    q2 >= 1: step expand factor
    epsg : Stopping tolerance for gradient norm
    epsx : Stopping tolerance for step size
    maxitn :  Maximum number of iterations
    intp : Print info every intp iterations

    Returns
    -------
    xr : best point found
    fr : best function value
    itn : number of iterations used
    nfg : number of function evaluations
    istop : stopping code
        2 = gradient small
        3 = step small
        4 = max iterations reached
        5 = line search error
    """
    # history
    history = {'f':[], 'fr(local opt)':[], 'x':[], 'B':[]}  # here x is trajectory

    # Current iterate (copy to avoid modifying original input)
    x = x0.copy()
    if projection:
        x = proj_simplex(x0.copy())
    
    
    # Dimension of variable space
    n = len(x)

    # Initialize transformation matrix B = I (identity matrix)
    # This corresponds to B0 in the paper
    B = eye_with_tail_pivot(n, m)
    history['B'].append(B)

    # Initialize step size
    hs = h0

    # Counters for line search statistics
    lsa = 0   # accumulated line search steps
    lsm = 0   # max line search steps

    # Best solution found so far
    xr = x.copy()
    history['x'].append(xr)
    # Evaluate function and gradient at initial point， f is the function, g0 is the gradient, this is y? 
    fr, g0 = calcfg(x)
    f = fr
    history['f'].append(fr)

    # Count function evaluations
    nfg = 1

    # Print initial information
    print(f"itn {0:4d}  f {fr:15.6e}  fr {fr:15.6e}  nfg {nfg:4d}")

    # Stopping condition: gradient norm small  
    if np.linalg.norm(g0) < epsg:
        return xr, fr, 0, nfg,  2, history

    # =======================
    # Main iteration loop (do normal subgradident descent)
    # =======================
    for itn in range(1, maxitn + 1):  # allowing how many times greaident update and space dilation update 

        # Compute transformed gradient to current space: B^T g_k  
        g_trans = B.T @ g0

        # Normalize transformed gradient
        g_trans_norm = np.linalg.norm(g_trans)

        # Descent direction:
        # dx = B * (B^T g_k / ||B^T g_k||)
        # This is normalized anti-subgradient in transformed space
        xi_k = (g_trans / g_trans_norm)  #xi_k
        dx = B @ xi_k  #B_k @ xi_k
        

        # Initialize line search variables
        d = 1          # control variable for descent
        ls = 0         # number of line search steps
        ddx = 0        # total step movement

        # =======================
        # Adaptive line search to find a good hk (step size)  not need to be the best   hk >= hk*
        # =======================
        while d > 0:  # if true keep looping

            # Move along descent direction  xk+1 = xk - hs * dx , x1 is drawn by using the steepest descent direction
            x = x - hs * dx
            if projection:
                x = proj_simplex(x)
            history['x'].append(x)
            
            # Accumulate movement size
            ddx += hs * np.linalg.norm(dx)

            # Evaluate function and gradient at new point
            f, g1 = calcfg(x)
            nfg += 1
            history['f'].append(f)

            # Update best solution if improved
            if f < fr:
                fr = f
                xr = x.copy()
            history['fr(local opt)'].append(fr)

            # Stop if gradient small
            if np.linalg.norm(g1) < epsg:
                return xr, fr, itn, nfg, 2, history

            # Increase line search counter
            ls += 1

            # Every nh steps, slightly increase step size by q2
            if ls % nh == 0:
                hs *= q2

            # Emergency stop if too many LS steps （line search failed too many steps) so 500 is the maximum line search steps allowed
            if ls > 500:  #k_max = 500
                return xr, fr, itn, nfg, 5, history
                # break

            # Check descent condition:
            # d = dx^T g_{k+1} pg700
            # calculate the new d 
            # If d <= 0, descent condition satisfied so stop, the step size is good enough for current x 
            # If d > 0, keep looping to find a step size such that the direction becomes a descent direction.
            d = dx @ g1
            # d = f_prev - f  # this is the sufficient decrease condition, if not satisfied, keep shrinking step size

        # If descent finished in one step, shrink step size
        #在 while 里找到一个“合适的 hk”
        #如果 ls = 1，就把 hk 稍微缩小一点
        #但 不会重新做这次 line search
        if ls == 1:
            hs *= q1

        # Update LS statistics
        #ls	当前这一次 line search 走了多少步 第一个iteration k=1, ls = 2,  k=2, ls=3
        #lsa	line search steps accumulated（累计总步数） 走到d不满足loop条件停下来，算lsa = 2+3 = 5 
        #lsm	line search max（当前区间内最大步数）   看最差（最多次line search）的记录
        lsa += ls
        lsm = max(lsm, ls)

        # Print progress every intp iterations, uf intp = 10, 每10次打印一次迭代信息
        if itn % intp == 0:
            print(f"itn {itn:4d}  f {f:14.6e}  fr {fr:14.6e}  "
                  f"nfg {nfg:4d}  lsa {lsa:3d}  lsm {lsm:3d}")
            lsa = 0
            lsm = 0

        # Stop if movement between two steps are small, ddx is the total step movement accumulated during line search
        if ddx < epsx:  # ddx = xk+1 - xk 
            return xr, fr, itn, nfg, 3, history

        # =======================
        # Space dilation update (updates B every iteration, but meaningful space dilation only occurs when the gradient differences have a consistent direction)
        # =======================
        # Compute gradient difference in transformed space:
        # dg = B^T (g_{k+1} - g_k), rk = g_{k+1} - g_k
        dg = B.T @ (g1 - g0)

        # Normalize direction xi
        xi = dg / np.linalg.norm(dg)

        # Update B matrix:
        # B_{k+1} = B_k + (1/alpha - 1) * (B_k xi) xi^T
        # This implements:
        # R_k = I + (1/alpha - 1) xi xi^T
        # B_{k+1} = B_k R_k
        B = B + (1/alpha - 1) * np.outer(B @ xi, xi)  # the space dilation operator formula
        history['B'].append(B)
        
        # Update gradient for next iteration
        g0 = g1

    # Max iterations reached
    return xr, fr, maxitn, nfg, 4, history




#**********************************************************************************************************************************************************
# su-alg save memory 
def sualg_m(calcfg, x0, h0=1.0,
    q1=0.9,
    q2 = 1.5,
    nh = 3,
    epsg=1e-8,
    epsx=1e-8,
    maxitn=1000,
    intp=50,
    theta = 0.01, 
    projection=False,
    m =5):
    """
    Parameters
    ----------
    calcfg :Function that returns (f, g) given x
    x0 :Initial starting point
    gamma: step size of combination of B-update and x-update # bstep in paper
    h0 : Initial step size
    q1 <= 1 : Step shrinking factor
    q2 >= 1: step expand factor
    nh : Number of steps to increase step size by q2
    epsg : Stopping tolerance for gradient norm
    epsx : Stopping tolerance for step size
    maxitn :Maximum number of iterations
    intp : Print info every intp iterations
    flevel :Target function value; algorithm stops if f(x) ≤ flevel    New!!!

    Returns
    -------
    xr : best point found
    fr : best function value
    itn : number of iterations used
    nfg : number of function evaluations
    istop : stopping code
        1 = function value reached flevel
        2 = gradient norm is samll small (||g|| < epsg)
        3 = step size is small  (||x_{k+1} - x_k|| < epsx)
        4 = max iterations reached 
        5 = line search error
    """
    
    # history
    history = {'f':[], 'fr(local opt)':[], 'x':[], 'B':[]}

    # Current iterate (copy to avoid modifying original input)
    x = x0.copy()
    if projection == True:
        x = proj_simplex(x0.copy())
    
    # Dimension of variable space
    n = len(x)

    # Initialize transformation matrix B = I (identity matrix)
    # This corresponds to B0 in the paper
    B = eye_with_tail_pivot(n,m)
    history['B'].append(B)
    # Initialize step size
    hs = h0

    # Counters for line search statistics
    lsa = 0   # accumulated line search steps
    lsm = 0   # max line search steps

    # Best solution found so far
    xr = x.copy()
    history['x'].append(xr)
    # Evaluate function and gradient at initial point， f is the function, g0 is the gradient, this is y? 
    fr, g0 = calcfg(x)
    history['f'].append(fr)

    # Count function evaluations
    nfg = 1

    # Print initial information
    print(f"itn {0:4d}  f {fr:15.6e}  fr {fr:15.6e}  nfg {nfg:4d}")

    # Stopping condition: gradient norm small  
    if np.linalg.norm(g0) < epsg:
        return xr, fr, 0, nfg,  2, history

    # =======================
    # Main iteration loop (do normal subgradident descent)
    # =======================
    for itn in range(1, maxitn + 1):

        # Compute transformed gradient to current space: B^T gf_(xk)  
        g_trans = B.T @ g0

        # Normalize transformed gradient
        g_trans_norm = np.linalg.norm(g_trans)

        # Descent direction:
        # dx = B * (B^T g_k / ||B^T g_k||)
        # This is normalized anti-subgradient in transformed space
        xi_k = (g_trans / g_trans_norm)  #xi_k
        dx = B @ xi_k  #B_k @ xi_k

        # Initialize line search variables
        d = 1          # control variable for descent
        ls = 0         # number of line search steps
        ddx = 0        # total step movement

        # =======================
        # Adaptive line search to find a good hk (step size)  not need to be the best   hk >= hk*
        # =======================
        while d > 0:  # if true keep looping

            # Move along descent direction  xk+1 = xk - hs * dx  update x
            x = x - hs * dx
            if projection:
                x = proj_simplex(x)
            history['x'].append(x)
            
            # Accumulate movement size
            ddx += hs * np.linalg.norm(dx)

            # Evaluate function and gradient at new point
            f, g1 = calcfg(x)
            nfg += 1
            history['f'].append(f)

            # Update best solution if improved
            if f < fr:
                fr = f
                xr = x.copy()
            history['fr(local opt)'].append(fr)
            
            # Stop if gradient small
            if np.linalg.norm(g1) < epsg:
                return xr, fr, itn, nfg, 2, history

            # Increase line search counter
            ls += 1

            # Every nh steps, slightly increase step size by q2
            if ls % nh == 0:
                hs *= q2

            # Emergency stop if too many LS steps （line search failed too many steps)
            if ls > 500:
                return xr, fr, itn, nfg, 5, history

            # Check descent condition:
            # d = dx^T g_{k+1} pg700
            # calculate the new d 
            # If d <= 0, descent condition satisfied so stop, the step size is good enough for current x 
            # If d > 0, keep looping to find a step size such that the direction becomes a descent direction.
            d = dx @ g1

        # If descent finished in one step, shrink step size
        #在 while 里找到一个“合适的 hk”
        #如果 ls = 1，就把 hk 稍微缩小一点
        #但 不会重新做这次 line search
        if ls == 1:
            hs *= q1

        # Update LS statistics
        #ls	当前这一次 line search 走了多少步 第一个iteration k=1, ls = 2,  k=2, ls=3
        #lsa	line search steps accumulated（累计总步数） 走到d不满足loop条件停下来，算lsa = 2+3 = 5 
        #lsm	line search max（当前区间内最大步数）   看最差（最多次line search）的记录
        lsa += ls
        lsm = max(lsm, ls)

        # Print progress every intp iterations, uf intp = 10, 每10次打印一次迭代信息
        if itn % intp == 0:
            print(f"itn {itn:4d}  f {f:14.6e}  fr {fr:14.6e}  "
                  f"nfg {nfg:4d}  lsa {lsa:3d}  lsm {lsm:3d}")
            lsa = 0
            lsm = 0

        # Stop if movement between two steps are small, ddx is the total step movement accumulated during line search
        if ddx < epsx:  # ddx = xk+1 - xk 
            return xr, fr, itn, nfg, 3, history

        # =======================
        # updates B every iteration, but meaningful space dilation only occurs when the gradient differences have a consistent direction)

        # Normalize direction xi
        g1_norm = np.linalg.norm(g1)
        if g1_norm < epsg:
            return xr, fr, itn, nfg, 2, history

        xi = g1 / g1_norm

        # Update B matrix:
        # B_{k+1} = B_k + gamma_si * (xi * g_k^T  + g_k * xi^T) * B_k
        # same space, but change the direction of the gradient (transform the gradient by using B)
        B = B +  theta * (np.outer(xi, g0) + np.outer(g0, xi)) @ B
        history['B'].append(B)

        # Update gradient for next iteration
        g0 = xi

    # Max iterations reached
    return xr, fr, maxitn, nfg, 4, history
#%% md
# 