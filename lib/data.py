import pandas as pd
import numpy as np

def import_csv(path_2_file, header=True):
    if(header):
        data = pd.read_csv(path_2_file)
    else:
        data = pd.read_csv(path_2_file, header=None)

    data_array = np.array(data)
    X = data_array[:,:-1]
    Y = data_array[:,-1]
    return(X,Y)


def sample_balanced(X, Y, m, shuffle=False):

    X_first_class = X[Y==np.unique(Y)[0]]
    X_second_class = X[Y==np.unique(Y)[1]]

    m_per_class = m//2

    first_class_ids = np.random.choice(X_first_class.shape[0], size=(m_per_class,), replace=False)
    second_class_ids = np.random.choice(X_second_class.shape[0], size=(m_per_class,), replace=False)

    X_1 = X_first_class[first_class_ids]
    X_2 = X_second_class[second_class_ids]

    Y_1 = np.unique(Y)[0]*np.ones((m_per_class,))
    Y_2 = np.unique(Y)[1]*np.ones((m_per_class,))

    X_new = np.vstack((X_1,X_2))
    Y_new = np.concatenate((Y_1,Y_2))

    if(shuffle):
        shuffled_ids = np.random.choice(m, size=(m,), replace=False)
        X_new = X_new[shuffled_ids]
        Y_new = Y_new[shuffled_ids]

    return(X_new, Y_new)

def generate_name(i):
    if(i<10):
        return('0'+str(i)+'.csv')
    else:
        return(str(i)+'.csv')

def import_prepare_sonar():
    path_2_data = 'datasets/sonar.all-data.csv'
    data_array = pd.read_csv(path_2_data).to_numpy()
    X = data_array[:,:-1]
    #Y = data_array[:,-1]

    # Center and normalize data
    X_normalized = (X - X.min(axis=0))
    X_normalized = X_normalized/X_normalized.max(axis=0)

    # Reshuffle columns in descending variance order
    X_var = X_normalized.var(axis=0)
    X_var_ordered = X_normalized[:,np.argsort(-X_var)] # negated array to sort descending
    return(X_var_ordered)    