import numpy as np
import decimal, time
import tensorflow as tf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from itertools import combinations_with_replacement
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from matplotlib.ticker import MaxNLocator
from scipy import special
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
from tqdm import tqdm
import seaborn as sns
import pysindy as ps

def fn_lib(X, poly_order, trig, ex, u_or_not=0):
    if u_or_not == 1:
        u = X[:,-1]
        if u.shape[0]==u.size: u = u.reshape(-1,1)
        X = np.delete(X, -1, 1)
    m = np.size(X, axis=0)
    n = np.size(X, axis=1)
    r = np.arange(n)
    
    a1 = ["x" + str(j) for j in np.arange(n)]
    b = [j + " dot" for j in a1]
    a = ["1"]
    lib_store = np.ones((m, 1))
    for P in range(1,poly_order+1): # 4 if till cubic, 6 if till quintic
        ind = np.array(np.meshgrid(*[r]*P, indexing='ij')).T.reshape(-1,P)
        ind = np.flip(ind, axis=1)
        ind = ind[np.all(np.diff(ind, axis=1) >= 0, axis=1)]
        lib = np.prod(X[:,ind.T].reshape((m, P, -1)), 1)
        lib_store = np.append(lib_store, lib, axis=1)
        a_loop =  list(combinations_with_replacement(a1, P))
        a_loop = [''.join(i) for i in a_loop]
        a.append(a_loop)
    a = [item for sublist in a for item in sublist] # Flatten the list
    if trig == 1:
        lib_store = np.append(lib_store, np.sin(X), axis=1)
        lib_store = np.append(lib_store, np.cos(X), axis=1)
        lib_store = np.append(lib_store, np.tan(X), axis=1)
        lib_store = np.append(lib_store, np.tanh(X), axis=1)
        a2 = ["sin " + j for j in a1]
        a3 = ["cos " + j for j in a1]
        a4 = ["tan " + j for j in a1]
        a5 = ["tanh " + j for j in a1]
        a = a+a2+a3+a4+a5
    if ex == 1:
        lib_store = np.append(lib_store, np.exp(X), axis=1)
        lib_store = np.append(lib_store, np.exp(-X), axis=1)
        a6 = ["exp " + j for j in a1]
        a7 = ["exp -" + j for j in a1]
        a = a+a6+a7
    if u_or_not == 1:
        lib_store = np.append(lib_store, u, axis=1)
        a.append('u')
    col_labels = ["Term",*b]
    return lib_store, a, col_labels

def sparse_iden(lam, Theta, dXdt):
    n = dXdt.shape[1]
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0] # initial guess: Least-squares
    for i in range(10):
        smallinds = (np.abs(Xi)<lam) # find small coefficients
        Xi[smallinds] = 0 # and threshold
        for ind in range(n): # n is state dimension
            biginds = ~smallinds[:,ind] # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds,ind] = np.linalg.lstsq(Theta[:,biginds],dXdt[:,ind], rcond=None)[0]
    np.set_printoptions(suppress=False)
    return Xi

def ode_recon(t, y, poly_order, trig, ex, Xi, u=None):
    n = Xi.shape[1]
    r = np.arange(n)
    lib_store = 1
    
    for P in range(1,poly_order+1): # 4 if till cubic, 6 if till quintic
        ind = np.array(np.meshgrid(*[r]*P, indexing='ij')).T.reshape(-1,P)
        ind = np.flip(ind, axis=1)
        ind = ind[np.all(np.diff(ind, axis=1) >= 0, axis=1)]
        lib = np.prod(y[ind], 1)
        lib_store = np.append(lib_store, lib)
    if trig == 1:
        lib_store = np.append(lib_store, np.sin(y))
        lib_store = np.append(lib_store, np.cos(y))
        lib_store = np.append(lib_store, np.tan(y))
        lib_store = np.append(lib_store, np.tanh(y))
    if ex == 1:
        lib_store = np.append(lib_store, np.exp(y))
        lib_store = np.append(lib_store, np.exp(-y))
    if u is not None: lib_store = np.append(lib_store, u)
    dydt =  np.dot(Xi.T,lib_store)
    return dydt

def batch(t, y):
    k1, k2, kr1 = 5.0, 0.5, 1.0
    C_B, C_C, C_A = y
    dydt = np.zeros(3)
    dydt[2] = -k1*C_A+kr1*C_B
    dydt[0] = k1*C_A-kr1*C_B-k2*C_B
    dydt[1] = k2*C_B
    return dydt

def data_generator_batch(t, num_ICs):
    k1, k2, kr1 = 5.0, 0.5, 1.0
    t_0 = t[0]
    t_f = t[-1]
    num_points = len(t)
    delta_t = t[1] - t[0]
    
    sol_store = np.zeros((0,3), float)
    grad_fd_store = np.zeros((0,3), float)
    grad_ex_store = np.zeros((0,3), float)
    
    for i in range(num_ICs):
        # np.random.seed(i*5)
        # y0 = np.random.randint(5, size=3)
        y0 = np.array([9-i, 0, i])
        solution = solve_ivp(batch, [t_0, t_f], y0, t_eval = t, method='LSODA')
        sol = solution.y.T
        grad_ex_3 = -k1*sol[:,2]+kr1*sol[:,0]
        grad_ex_1 = k1*sol[:,2]-kr1*sol[:,0]-k2*sol[:,0]
        grad_ex_2 = k2*sol[:,0]
        grad_ex = np.vstack((grad_ex_1, grad_ex_2, grad_ex_3)).T
        grad_fd = np.gradient(sol, delta_t, axis=0, edge_order=1)
        sol_store = np.vstack((sol_store, sol))
        grad_ex_store = np.vstack((grad_ex_store, grad_ex))
        grad_fd_store = np.vstack((grad_fd_store, grad_fd))
        
        plt.figure(1, figsize=(8,10))
        plt.subplot(5,2,i+1)
        plt.plot(solution.t, solution.y.T[:,2], label='$C_A$')
        plt.plot(solution.t, solution.y.T[:,0], label='$C_B$')
        plt.plot(solution.t, solution.y.T[:,1], label='$C_C$')
        plt.xlabel('$t$')
        plt.ylabel('$C$')
        plt.legend(loc=0)        
    plt.tight_layout()
    plt.savefig('./fig/Data.png', format='png')
    plt.close('all')
    
    plt.figure(2, figsize=(8,10))
    for i in range(num_ICs):
        plt.subplot(5,2,i+1)
        plt.plot(t, grad_ex)
    plt.tight_layout()
    plt.savefig('./fig/Data Gradient.png', format='png')
    plt.close('all')
    
    X_original = sol_store
    means = np.mean(X_original, axis=0)
    stds = np.std(X_original, axis=0)
    X_original_scaled = (X_original - means)/stds
    
    ind_del = np.array([], dtype=np.int32)
    for i in range(0, num_points*num_ICs, 100): ind_del = np.append(ind_del, np.arange(i,i+10))
    X_trun = np.delete(X_original, ind_del, axis=0)
    X_trun_scaled = np.delete(X_original_scaled, ind_del, axis=0)

    num_points_total = X_original.shape[0]
    return X_original, X_original_scaled, X_trun, X_trun_scaled, num_points_total, means, stds, ind_del, grad_ex_store, grad_fd_store

# %% Nonlinear Principal Component Analysis

# Point-to-line distance calculator
def point_to_line_segment(start, end, point):  # x3,y3 is the point
    delta = end - start
    norm =  np.sum(delta**2)
    u = (np.dot(point - start, delta)) / float(norm)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    intersection = start + u*delta
    dist =  np.sum((intersection - point)**2)
    return dist, intersection

def prin_curve(X, more_iter, num_PC, spn):
    num_points = X.shape[0]
    num_var = X.shape[1]
    min_iter = 5
    max_iter = 20
    eps = 0.01

    lowess = sm.nonparametric.lowess
    
    D2_old, D2_new, D2_new_by_old = np.zeros((3, max_iter))
    d_ij = np.zeros(num_points - 1)
    min_d_ij = np.zeros(num_points)
    index = np.zeros(num_points, dtype=np.uint8)
    lambdas = np.zeros([num_points, max_iter + 1])
    lambdas_f = np.zeros([num_points, max_iter + 1])
    lambdas_ordered = np.zeros([num_points, max_iter + 1])
    X_new = np.zeros([num_points, num_var, max_iter])
    X_ordered = np.zeros([num_points, num_var, max_iter])
    X_iter = np.zeros([num_points - 1, num_var])
    X_proj_onto_pca = np.zeros([num_points, num_var])
    X_proj = np.zeros([num_points, num_var])
    
    # Initialization: Find first linear principal component, f^(0)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    X_for_lin = X - np.mean(X, axis=0)
    pca = PCA(n_components=1)
    pca.fit(X_for_lin)
    scores = pca.transform(X_for_lin)
    linear_PCA_line = pca.inverse_transform(scores)
    linear_PCA_line = linear_PCA_line  + np.mean(X, axis=0)
    pca_vec = pca.components_
    pca_vec_T = np.transpose(pca_vec)
    # Set lambda^(0) (x) = lambdas_{f^(0)} (x) or find projections of data points onto first linear principal component
    for i in range(0, num_points):
        point = X[i,:]
        X_proj_onto_pca[i,:] = np.dot(point,pca_vec_T)*pca_vec # No need to divide by norm of line because norm=1
    # Assign lambdas to projections onto f^(0). Find "first" data or most "left" data
    pos = np.argmin(X_proj_onto_pca[:,0])
    # Find lambdas_f values as Euclidean distance from this point so cumulative
    lambdas_f[:,0] = np.sqrt(np.sum((X_proj_onto_pca - X_proj_onto_pca[pos,:])**2, axis=1))
    lambdas[:,0] = lambdas_f[:,0]
    # Sort lambdas and data points accordingly. This is lambda^(0) (x) = lambdas_{f^(0)} (x)
    lambdas_ordered[:,0] = np.sort(lambdas[:,0])
    sort_order = np.argsort(lambdas[:,0])
    X_ordered[:,:,0] = X[sort_order]
    
    # Iteration 1 to end
    for k in tqdm(range(0, max_iter)):
        # Step 1: Set f^(k)(lambda) = E(X | lambda_{f^(k-1)} (x) = lambda)
        # f^(k) is the curve of (X_new[:, 0, k], X_new[:, 1, k], X_new[:, 2, k],...)
        for l in range(0, num_var): X_new[:, l, k] = lowess(X_ordered[:,l,k], lambdas_ordered[:, k], frac=spn, return_sorted=False)
        if k==0:
            D2_old[k] = np.mean((X-X_proj_onto_pca)**2) # Distance to first linear PCA line
        else:
            D2_old[k] = np.mean(min_d_ij)        
    
        # Step 2: Define lambda^(k) (x) = lambda_{f^(k)} (x) but need to find the new arc lengths because f^(k) is a new curve
        # Find lambdas for f^(k) to get f^(k) (lambda)
        # for i in range(1, num_points):
        lambdas[1:, k+1] = np.sqrt(np.sum(np.diff(X_new[:,:,k], axis=0)**2,axis=1))
        lambdas[:, k+1] = np.cumsum(lambdas[:, k+1])
    
        # Now project data points onto f^(k) and find lambda_f^(k) for each point
        for i in range(0, num_points):
            for j in range(0, num_points - 1):
                [d_ij[j], X_iter[j,:]] = point_to_line_segment(X_new[j,:,k], X_new[j+1,:,k], X_ordered[i,:,k])
            min_d_ij[i] = np.amin(d_ij)
            index[i] = np.argmin(d_ij)
            X_proj[i,:] = X_iter[index[i],:]
            
        # Assign lambdas to projections onto f^(k)
        # Check if point's closest point to f^(k) is a vertex and set it to that vertex, otherwise interpolate
        for i in range(0, num_points):
            if any(np.absolute(X_proj[i,0] - X_new[:, 0, k]) < 1.0e-3): # Not ==0 for Python
                lambdas_f[i, k+1] = lambdas[i, k+1]
            else:
                seg_no = index[i]
                lambdas_f[i, k+1] = lambdas[seg_no, k+1] + (lambdas[seg_no + 1, k+1] - lambdas[seg_no, k+1]) / (X_new[seg_no + 1, 0, k] - X_new[seg_no, 0, k]) * (X_proj[i,0] - X_new[seg_no, 0, k])
        # Set lambda^(k) (x) = lambda_{f^(k)} (x)
        lambdas[:, k+1] = lambdas_f[:, k+1]
        
        # Step 3: Error calculation
        D2_new[k] = np.mean(min_d_ij)
        D2_new_by_old[k] = abs(1 - D2_new[k] / D2_old[k])
        if more_iter == 0:
            if D2_new_by_old[k] < eps: break
        else:
            if any(t < 0 for t in D2_new_by_old) < eps and k>= min_iter: break
    
        # If continuing, reorder X_ordered according to lambda again for next iteration
        if k != max_iter-1:
            lambdas_ordered[:, k+1] = np.sort(lambdas[:, k+1])
            sort_order = np.argsort(lambdas[:, k+1])
            X_ordered[:,:,k+1] = X_ordered[:,:,k][sort_order]
            
    pca = PCA(n_components=num_PC).fit(X)
    perc_var_linear = pca.explained_variance_ratio_[num_PC-1]
    
    perc_var_PC1 = np.sum(np.var(X_new[:,:,k],axis=0))/np.sum(np.var(X,axis=0))
    residual = X_ordered[:,:,k] - X_new[:,:,k]
    return linear_PCA_line, X_new[:,:,k], X_ordered[:,:,k], lambdas_ordered[:,k], residual, perc_var_linear, perc_var_PC1, k

def NN_trainer_2_hidden_layers(X, y, num_hidden_1, num_hidden_2, num_in, num_out, hyper_params):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=6)
    opt = tf.keras.optimizers.Adam(
    learning_rate=hyper_params[0],
    beta_1=hyper_params[1],
    beta_2=hyper_params[2],
    epsilon=hyper_params[3],
    amsgrad=False
    )
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(num_hidden_1, input_dim=num_in, activation=hyper_params[4]),
        tf.keras.layers.Dense(num_hidden_2, activation=hyper_params[5]),
        tf.keras.layers.Dense(num_out, activation='linear')
    ])
    model.compile(optimizer=opt,
                   loss='mean_squared_error',
                   metrics=['mae'])
    model.fit(X_train, y_train, epochs=50, validation_data=[X_val, y_val], verbose=0)
    return model