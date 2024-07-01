from lib.projected_kernel import get_all_partial_traces
import numpy as np
import pandas as pd

def nfeature_2_str(n):
    if(n<10):
        return('0'+str(n))
    else:
        return(str(n))

ZZ_reps = 4
# Constants

ent_type = "full" #linear, circular, full
# Data
path_2_data = 'datasets/sonar.all-data.csv'
# Data preprocessing
data_array = pd.read_csv(path_2_data).to_numpy()
X_import = data_array[:,:-1]
Y = data_array[:,-1]
m = X_import.shape[0]
# Center and normalize data
X_normalized = (X_import - X_import.min(axis=0))
X_normalized = X_normalized/X_normalized.max(axis=0)
# Reshuffle columns in descending variance order
X_var = X_normalized.var(axis=0)
X_var_ordered = X_normalized[:,np.argsort(-X_var)] # negated array to sort descending

for N_FEATURES in range(11, 12):
    print('N_FEATURES: ', N_FEATURES)
    X = X_var_ordered[:,:N_FEATURES]
    densities = get_all_partial_traces(X, ZZ_reps=ZZ_reps, ent_type=ent_type)
    if(ent_type=='linear'):
        np.save('./results/sonar/reduced_density_matrices/linear/'+'densities_'+nfeature_2_str(N_FEATURES)+'_ZZ_'+str(ZZ_reps), densities)
    if(ent_type=='full'):
        np.save('./results/sonar/reduced_density_matrices/full/'+'densities_'+nfeature_2_str(N_FEATURES)+'_ZZ_'+str(ZZ_reps), densities)