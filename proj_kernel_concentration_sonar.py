#Imports
# Classical Imports
import numpy as np
import matplotlib.pyplot as plt
#from lib.data import import_csv, sample_balanced, generate_name
import os

from datetime import datetime

import pandas as pd

from lib.projected_kernel import projected_kernel

# Quantum Imports
#from qiskit.circuit.library import ZZFeatureMap
#from qiskit.primitives import Sampler
#from qiskit.algorithms.state_fidelities import ComputeUncompute
#from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Data
path_2_data = 'datasets/sonar.all-data.csv'

# Constants
ZZ_reps = 4
#m = 100
N_FEATURES_MIN = 2
N_FEATURES_MAX = 11
N_FEATURES_LIST = [int(i) for i in np.arange(N_FEATURES_MIN,N_FEATURES_MAX+1,1)]
ent_type = "linear"

data_array = pd.read_csv(path_2_data).to_numpy()
X = data_array[:,:-1]
Y = data_array[:,-1]

m = X.shape[0]

# Center and normalize data
X_normalized = (X - X.min(axis=0))
X_normalized = X_normalized/X_normalized.max(axis=0)

# Reshuffle columns in descending variance order
X_var = X_normalized.var(axis=0)
X_var_ordered = X_normalized[:,np.argsort(-X_var)] # negated array to sort descending


means = []
stds = []
kernel_entries = np.zeros((len(N_FEATURES_LIST),int(m*(m-1)/2)))

for i in range(len(N_FEATURES_LIST)):
    N_FEATURES = N_FEATURES_LIST[i]
    print(N_FEATURES)

    X_kernel = X_var_ordered[:,:N_FEATURES]

    gram_matrix = projected_kernel(X_kernel, ZZ_reps=ZZ_reps, ent_type=ent_type)

    mask = np.triu_indices(m,k=1)
    independent_entries = gram_matrix[mask]
    kernel_entries[i] = independent_entries

path_2_export = 'results/sonar/'
pd.DataFrame(kernel_entries).to_csv(path_2_export+'projected_entries/'+'NF_'+str(N_FEATURES_MAX)+'_ZZ_'+str(ZZ_reps)+'.csv', index = False, header = False)
