#Imports
# Classical Imports
#import matplotlib.pyplot as plt
#from lib.data import import_csv, sample_balanced, generate_name
import os

import numpy as np
import pandas as pd

from scipy.special import rel_entr
from scipy.linalg import logm

from lib.projected_kernel import projected_kernel, quantum_relative_entropy, Gamma
from lib.data import import_prepare_sonar

# Data
X = import_prepare_sonar()
m = X.shape[0]

# Constants
ZZ_reps = 2
#m = 100
N_FEATURES_MIN = 2
N_FEATURES_MAX = 5
N_FEATURES_LIST = [int(i) for i in np.arange(N_FEATURES_MIN,N_FEATURES_MAX+1,1)]
ent_type = "full"

entropy_entries = np.zeros((len(N_FEATURES_LIST),int(m*(m-1)/2)))

for i in range(len(N_FEATURES_LIST)):
    N_FEATURES = N_FEATURES_LIST[i]
    print(N_FEATURES)

    X_kernel = X[:,:N_FEATURES]

    entropies_matrix = np.zeros((m,m))
    for row in range(entropies_matrix.shape[0]):
        for column in range(row+1,entropies_matrix.shape[1]):
            entropies_matrix[row, column] = Gamma(X_kernel[row], X_kernel[column])

    entropies_matrix += entropies_matrix.T

    mask = np.triu_indices(m,k=1)
    independent_entries = entropies_matrix[mask]
    entropy_entries[i] = independent_entries

path_2_export = 'results/sonar/'
pd.DataFrame(entropy_entries).to_csv(path_2_export+'entropies/'+ent_type+'/'+'NF_'+str(N_FEATURES_MAX)+'_ZZ_'+str(ZZ_reps)+'.csv', index = False, header = False)
