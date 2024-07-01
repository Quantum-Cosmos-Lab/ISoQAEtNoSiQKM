import numpy as np

def entries_2_matrix(entries):
    """Takes the one-dimensional array of independent kernel entries and returns the kernel Gram matrix"""
    N = len(entries)
    m = int((1+np.sqrt(1+8*N))/2) # positive root of the quadratic equation m(m-1)/2 = N
    tri = np.zeros((m, m))
    tri[np.triu_indices(m, 1)] = entries
    tri = tri+tri.T + np.identity(m)
    return(tri)

def get_x_info(x, x_all):
    """
    Create a matrix consisting of information about the number of qubits that were calculated and simulated
    The first column lists all qubit numbers, the second column tell if this qubit number was calculated.
    """
    x_info = np.empty((x_all.shape[0], 2))
    x_info[:,0] = x_all
    for i in range(len(x)):
        x_info[i,1] = 1
    return(x_info)

def get_N_estimates(N_s, N_sr):
    """
    Create a three-index matrix consisting of information about the estimates of N. First element is N_spread, second - N_successRate, third - maximum of previous two.
    """
    N_tot = np.empty(N_s.shape)

    for ZZ_rep in range(N_s.shape[0]):
        N_tot[ZZ_rep] = np.maximum(N_s[ZZ_rep], N_sr[ZZ_rep])
    
    N_estimates = np.empty((3, N_s.shape[0], N_s.shape[1]))
    N_estimates[0] = N_s
    N_estimates[1] = N_sr
    N_estimates[2] = N_tot

    return(N_estimates)

def filter_n(n, files):
    index = (files[0].find('_ZZ'))
    files_n = [f for f in files if int(f[index-2:index])==n]
    return(files_n)

def filter_r(r, files):
    files_r = [f for f in files if int(f[-5])==r]
    return(files_r)