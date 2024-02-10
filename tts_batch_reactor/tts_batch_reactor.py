import numpy as np
import os, sys, pickle
import matplotlib as mpl
import matplotlib.animation as animation
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import special
from scipy.integrate import odeint, solve_ivp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate

sys.path.append(os.path.abspath("/workspaces/PhD/"))
from subfunctions import fn_lib, sparse_iden, ode_recon, batch, data_generator_batch, prin_curve, NN_trainer_2_hidden_layers

class ProcessSystemsEngineering:
    def __init__(self):
        self.t_0 = 0
        self.t_f = 10
        self.delta_t = 0.1
        self.t = np.arange(self.t_0, self.t_f, step=self.delta_t)
        self.num_points = len(self.t)
        self.num_ICs = 10
        self.more_iter = 0
        self.lambda_jump_tol = 0.4
        self.trun_NN = 0
        self.slow_index = 2  # index is 1 higher so 0:3 is 0,1,2
        self.X_original, self.X_original_scaled, self.X_trun, self.X_trun_scaled, self.num_points_total, self.means, self.stds, self.ind_del, self.grad_ex, self.grad_fd = data_generator_batch(self.t, self.num_ICs)
        self.fast_index = len(self.means) - self.slow_index # number of fast variables
        self.num_points_total_trun = self.X_trun.shape[0]
        self.y0 = np.array([5.5,0,3.5])
        self.solution = solve_ivp(batch, [self.t_0, self.t_f], self.y0, t_eval = self.t, method='LSODA')
        self.sol_diff = self.solution.y.T
        self.sol_diff_scaled = (self.sol_diff - self.means)/self.stds

    def train_neural_network(self, NLPCA_output_file):
        with open(NLPCA_output_file, 'rb') as f:
            X_new, X_ordered, X_trun, X_PCA, residual, linear_PCA_line, lambdas, lambdas_trun, lambdas_fixed, lambdas_fixed_trun, perc_var_linear, perc_var_PC1, end_k, X_new_dist, res_dist = pickle.load(f)

        X_NN = X_new[:,:,0]
        lambdas_NN = lambdas_fixed
        json_file = open('tts_batch_reactor_NN_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = tf.keras.models.model_from_json(loaded_model_json)
        self.model.load_weights("tts_batch_reactor_NN_model.h5")

    def sparse_identification(self, lam, num_SINDy_states): # lam is our sparsification knob
        # Your sparse identification logic here
        # For example:
        self.x_SI = self.X_original[:,0:num_SINDy_states]
        self.poly_order = 4
        self.trig = 1
        self.ex = 1
        self.dXdt = self.grad_fd[:,0:num_SINDy_states]
        self.Theta, self.row_labels, self.col_labels = fn_lib(self.x_SI, self.poly_order, self.trig, self.ex)
        self.Xi = sparse_iden(lam, self.Theta, self.dXdt)
        print(tabulate(np.c_[self.row_labels, self.Xi], headers=self.col_labels, tablefmt="fancy_grid"))
        return self.Xi
    
    def predictor(self, num_SINDy_states):
        solution = solve_ivp(ode_recon, [self.t_0, self.t_f], self.y0[0:num_SINDy_states], t_eval=self.t, method='LSODA', args=(self.poly_order, self.trig, self.ex, self.Xi))
        if num_SINDy_states == self.slow_index:
            print('Reduced-order slow subsystem: ' + solution.message)
            self.x_recon = solution.y.T
            x_recon_scaled = (self.x_recon - self.means[0:self.slow_index])/self.stds[0:self.slow_index]
            z_pred_scaled = self.model.predict(x_recon_scaled)
            self.z_pred = z_pred_scaled*self.stds[self.slow_index:] + self.means[self.slow_index:]
        else:
            print('Full order model: ' + solution.message)
            self.x_recon = solution.y.T

def plot_results(t, sol_diff, ROM, Full):
    plt.rc('text', usetex=0)
    plt.figure(figsize=(4,4))
    
    p1, = plt.plot(t, sol_diff[:,0], '-', label='$C_B$ ODE')
    p4, = plt.plot(t, sol_diff[:,1], '-', label='$C_C$ ODE')
    p7, = plt.plot(t, sol_diff[:,2], '-', label='$C_A$ ODE')
    plt.gca().set_prop_cycle(None)

    p2, = plt.plot(t, ROM.x_recon[:,0], '--', label='$C_B$ NLPCA-SI')
    p5, = plt.plot(t, ROM.x_recon[:,1], '--', label='$C_C$ NLPCA-SI')
    p8, = plt.plot(t, ROM.z_pred, '--', label='$C_A$ NLPCA-SI')
    plt.gca().set_prop_cycle(None)

    p3, = plt.plot(t, Full[:,0], ':', label='$C_B$ SI')
    p6, = plt.plot(t, Full[:,1], ':', label='$C_C$ SI')
    p9, = plt.plot(t, Full[:,2], ':', label='$C_A$ SI')

    plt.xlabel('$t$ (hr)')
    plt.ylabel('$C~\mathrm{(kmol/m^3)}$')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45),
              ncol=3, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig('./fig/Results.png', format='png')

if __name__ == "__main__":
    SINDy_ROM = ProcessSystemsEngineering()
    SINDy_ROM.train_neural_network('tts_batch_reactor_NLPCA_output.pickle')
    SINDy_ROM.sparse_identification(lam=0.5, num_SINDy_states=2)
    SINDy_ROM.predictor(num_SINDy_states=2)
    
    SINDy_Full = ProcessSystemsEngineering()
    SINDy_Full.train_neural_network('tts_batch_reactor_NLPCA_output.pickle')
    SINDy_Full.sparse_identification(lam=0.13, num_SINDy_states=3)
    SINDy_Full.predictor(num_SINDy_states=3)
    
    plot_results(SINDy_ROM.t, SINDy_ROM.sol_diff, SINDy_ROM, SINDy_Full.x_recon)