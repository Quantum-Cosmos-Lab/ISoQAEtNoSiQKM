#Imports
# Classical Imports
import numpy as np
import matplotlib.pyplot as plt
from lib.data import import_csv, sample_balanced, generate_name
import os

from datetime import datetime

import pandas as pd

# Quantum Imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Data
path_2_data = 'prepared_data/twonorm'

# Constants
ZZ_reps = 2
m = 100
N_FEATURES_MIN = 2
N_FEATURES_MAX = 12
N_FEATURES_LIST = [int(i) for i in np.arange(N_FEATURES_MIN,N_FEATURES_MAX+1,1)]
ent_type = "linear"


files = []

for file in os.listdir(path_2_data):
    if file.endswith(".csv"):
        files.append(os.path.join(path_2_data, file))

files.sort()

N_files = len(files)

# Main loop


file_id = 0
for file_path in files:
# inner loop
    print('file: ', file_path)
    means = []
    stds = []
    kernel_entries = np.zeros((len(N_FEATURES_LIST),int(m*(m-1)/2)))

    for i in range(len(N_FEATURES_LIST)):
        N_FEATURES = N_FEATURES_LIST[i]
        print(N_FEATURES)

        X, Y = import_csv(path_2_file=file_path, header=False)

        X_normalized = (X - X.min(axis=0))
        X_normalized = X_normalized/X_normalized.max(axis=0)

        X_kernel = X_normalized[:,:N_FEATURES]

        feature_map = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
        sampler = Sampler()
        fidelity = ComputeUncompute(sampler=sampler)
        kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

        gram_matrix = kernel.evaluate(X_kernel)

        mask = np.triu_indices(m,k=1)
        independent_entries = gram_matrix[mask]
        kernel_entries[i] = independent_entries

        means.append(independent_entries.mean())
        stds.append(independent_entries.std())

    means_array = np.array(means)
    stds_array = np.array(stds)

    path_2_export = 'results/twonorm/'
    pd.DataFrame(kernel_entries).to_csv(path_2_export+'entries/'+'NF_'+str(N_FEATURES_MAX)+'_ZZ_'+str(ZZ_reps)+'_'+file_path[22:], index = False, header = False)

    pd.DataFrame(means_array).to_csv(path_2_export+'statistical_measures/'+'means_NF_'+str(N_FEATURES_MAX)+'_ZZ_'+str(ZZ_reps)+'_'+file_path[22:], index = False, header = False)
    pd.DataFrame(stds_array).to_csv(path_2_export+'statistical_measures/'+'stds_NF_'+str(N_FEATURES_MAX)+'_ZZ_'+str(ZZ_reps)+'_'+file_path[22:], index = False, header = False)

# /inner loop

# /Main loop