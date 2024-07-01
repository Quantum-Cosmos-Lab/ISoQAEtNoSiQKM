#Imports
# Classical Imports
import numpy as np
import matplotlib.pyplot as plt
from lib.data import import_csv, sample_balanced, generate_name
import os
from tqdm.auto import tqdm

from datetime import datetime

import pandas as pd

# Quantum Imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Data
path_2_data = 'datasets/sonar.all-data.csv'

# Constants
ZZ_reps = 6

N_FEATURES_MIN = 2
N_FEATURES_MAX = 10 #8
N_FEATURES_LIST = [int(i) for i in np.arange(N_FEATURES_MIN,N_FEATURES_MAX+1,1)]
ent_type = "full" #linear, circular, full

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

kernel_entries = np.zeros((len(N_FEATURES_LIST),int(m*(m-1)/2)))

for i in tqdm(range(len(N_FEATURES_LIST))):
    N_FEATURES = N_FEATURES_LIST[i]
    print(N_FEATURES)

    X_kernel = X_var_ordered[:,:N_FEATURES]

    feature_map = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler=sampler)
    kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

    gram_matrix = kernel.evaluate(X_kernel)

    mask = np.triu_indices(m,k=1)
    independent_entries = gram_matrix[mask]
    kernel_entries[i] = independent_entries

path_2_export = 'results/sonar/'
pd.DataFrame(kernel_entries).to_csv(path_2_export+'new_entries/full/'+'recalculated_NF_'+str(N_FEATURES_MAX)+'_ZZ_'+str(ZZ_reps)+'.csv', index = False, header = False)
