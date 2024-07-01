import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import DensityMatrix,partial_trace
from scipy.linalg import logm


def Schatten_2_norm(X):
    X_squared = np.matmul(X.conj().T, X)
    norm = np.sqrt(np.trace(X_squared))
    if(np.imag(norm)>1e-10):
        print("Non-zero imaginary part of norm")
        return('Error')
    return(np.real(norm))

def kernel_value_from_exponent_norms(exponent_norms, gamma = 1):
    x = np.power(exponent_norms,2).sum()
    return(np.exp(-gamma*x))

def kernel_value(x1_id, x2_id, all_partial_traces_matrix, gamma = 1):
    exponent_norms = []
    partial_rhos_1 = all_partial_traces_matrix[x1_id]
    partial_rhos_2 = all_partial_traces_matrix[x2_id]

    for i in range(all_partial_traces_matrix.shape[1]):
        partial_rho_1 = partial_rhos_1[i]
        partial_rho_2 = partial_rhos_2[i]
        exponent_norms.append(Schatten_2_norm(partial_rho_1-partial_rho_2))
    exponent_norms = np.array(exponent_norms)
    
    return(kernel_value_from_exponent_norms(exponent_norms, gamma=gamma))

def densities_2_kernel(densities, gamma = 1):
    m  = densities.shape[0]
    N_FEATURES = densities.shape[1]
    kernel_matrix = np.zeros((m,m))

    for row in range(kernel_matrix.shape[0]):
        for column in range(row+1,kernel_matrix.shape[1]):
            kernel_matrix[row, column] = kernel_value(x1_id = row, x2_id = column, all_partial_traces_matrix = densities, gamma=gamma)
    kernel_matrix += kernel_matrix.T
    kernel_matrix += np.identity(kernel_matrix.shape[0])
    return(kernel_matrix)

def get_independent_kernel_entries(gram_matrix):
    mask = np.triu_indices(gram_matrix.shape[0],k=1)
    independent_entries = gram_matrix[mask]
    return(independent_entries)

def quantum_relative_entropy(X,Y):
    temp = np.matmul(X, logm(X) - logm(Y))
    return(np.real(temp.trace()))

def projected_kernel(X, ZZ_reps=1, ent_type='linear'):
    N_FEATURES = X.shape[1]
    all_partial_traces = np.empty((X.shape[0],N_FEATURES,2,2), dtype = 'complex128') # four dimensions: 0-number of data points, 1-number of partial trace matrices, 2,3 - dimension of partial trace matrix
    qubit_list = [i for i in range(N_FEATURES)]

    for data_point in range(X.shape[0]):
        qc = QuantumCircuit(N_FEATURES)
        fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
        fm_bound = fm.bind_parameters(X[data_point])
        qc.append(fm_bound, range(N_FEATURES))
        rho = DensityMatrix.from_instruction(qc)

        partial_rhos = np.empty((N_FEATURES, 2, 2), dtype = 'complex128')

        for current_qubit in qubit_list:
            list_to_trace_out = [qubit_id for qubit_id in qubit_list if qubit_id != current_qubit]
            partial_rho=partial_trace(rho,list_to_trace_out).data
            partial_rhos[current_qubit] = partial_rho

        all_partial_traces[data_point] = partial_rhos
    
    kernel_matrix = np.zeros((X.shape[0],X.shape[0]))

    for row in range(kernel_matrix.shape[0]):
        for column in range(row+1,kernel_matrix.shape[1]):
            kernel_matrix[row, column] = kernel_value(x1_id = row, x2_id = column, all_partial_traces_matrix = all_partial_traces)

    kernel_matrix += kernel_matrix.T
    kernel_matrix += np.identity(kernel_matrix.shape[0])
    return(kernel_matrix)

def Gamma(x1,x2, ZZ_reps=1, ent_type = 'linear'):
    
    MMS = np.identity(2)/2
    N_FEATURES = len(x1)
    qubit_list = [i for i in range(N_FEATURES)]

    qc = QuantumCircuit(N_FEATURES)
    fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
    fm_bound = fm.bind_parameters(x1)
    qc.append(fm_bound, range(N_FEATURES))
    rho1 = DensityMatrix.from_instruction(qc)

    qc = QuantumCircuit(N_FEATURES)
    fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
    fm_bound = fm.bind_parameters(x2)
    qc.append(fm_bound, range(N_FEATURES))
    rho2 = DensityMatrix.from_instruction(qc)

    partial_rhos1 = np.empty((N_FEATURES, 2, 2), dtype = 'complex128')
    for current_qubit in qubit_list:
        list_to_trace_out = [qubit_id for qubit_id in qubit_list if qubit_id != current_qubit]
        partial_rho=partial_trace(rho1,list_to_trace_out).data
        partial_rhos1[current_qubit] = partial_rho
    
    partial_rhos2 = np.empty((N_FEATURES, 2, 2), dtype = 'complex128')
    for current_qubit in qubit_list:
        list_to_trace_out = [qubit_id for qubit_id in qubit_list if qubit_id != current_qubit]
        partial_rho=partial_trace(rho2,list_to_trace_out).data
        partial_rhos2[current_qubit] = partial_rho

    entropies_1 = np.empty((N_FEATURES,))
    for i in range(partial_rhos1.shape[0]):
        entropy = quantum_relative_entropy(partial_rhos1[i], MMS)
        entropies_1[i] = entropy

    entropies_2 = np.empty((N_FEATURES,))
    for i in range(partial_rhos2.shape[0]):
        entropy = quantum_relative_entropy(partial_rhos2[i], MMS)
        entropies_2[i] = entropy

    Gamma = np.power(np.sqrt(entropies_1)+np.sqrt(entropies_2),2).sum()
    return(Gamma)

def get_all_partial_traces(X, ZZ_reps = 1, ent_type = 'linear'):
    N_FEATURES = X.shape[1]
    all_partial_traces = np.empty((X.shape[0],N_FEATURES,2,2), dtype = 'complex128') # four dimensions: 0-number of data points, 1-number of partial trace matrices, 2,3 - dimension of partial trace matrix
    qubit_list = [i for i in range(N_FEATURES)]

    for data_point in range(X.shape[0]):
        qc = QuantumCircuit(N_FEATURES)
        fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
        fm_bound = fm.bind_parameters(X[data_point])
        qc.append(fm_bound, range(N_FEATURES))
        rho = DensityMatrix.from_instruction(qc)

        partial_rhos = np.empty((N_FEATURES, 2, 2), dtype = 'complex128')

        for current_qubit in qubit_list:
            list_to_trace_out = [qubit_id for qubit_id in qubit_list if qubit_id != current_qubit]
            partial_rho=partial_trace(rho,list_to_trace_out).data
            partial_rhos[current_qubit] = partial_rho

        all_partial_traces[data_point] = partial_rhos
    return(all_partial_traces)

def densities_MMS_entropy(densities):
    m = densities.shape[0]
    N_FEATURES = densities.shape[1]
    d = densities.shape[2]
    
    MMS = np.identity(d)/d # Maximally mixed state

    relative_entropies = np.zeros((m,N_FEATURES))
    for data_id in range(m):
        for qubit_id in range(N_FEATURES):
            relative_entropies[data_id, qubit_id] = quantum_relative_entropy(densities[data_id, qubit_id], MMS)
    
    return(relative_entropies)

def Theta(x_rhos, y_rhos, gamma = 1):
    diff = x_rhos - y_rhos
    diff_00 = np.real(diff[:,0,0])
    diff_r = np.real(diff[:,0,1])
    diff_i = np.imag(diff[:,0,1])
    rhos = np.power(diff_00,2).sum() + np.power(diff_r,2).sum() + np.power(diff_i,2).sum()
    root_rhos = np.sqrt(rhos)
    factor = gamma*np.sqrt(32)
    return(factor*root_rhos)