import numpy as np

def bin_std_num(x):
    return(np.sqrt(x*(1-x)))

def V_k(k, x_densities, y_densities):
    x_density = x_densities[k]
    y_density = y_densities[k]
    
    D_x = np.real(x_density[0,0])
    R_x = np.real(x_density[0,1])+0.5
    I_x = 0.5-np.imag(x_density[0,1])
    D_y = np.real(y_density[0,0])
    R_y = np.real(y_density[0,1])+0.5
    I_y = 0.5-np.imag(y_density[0,1])

    covariances_vec = np.array([bin_std_num(D_x),bin_std_num(R_x),bin_std_num(I_x),bin_std_num(D_y),bin_std_num(R_y),bin_std_num(I_y)])
    covariances_matrix = np.outer(covariances_vec, covariances_vec)

    differences_vec = np.array([np.abs(D_x-D_y), np.abs(R_x-R_y), np.abs(I_x-I_y),np.abs(D_x-D_y), np.abs(R_x-R_y), np.abs(I_x-I_y)])
    differences_matrix = np.outer(differences_vec, differences_vec)

    V_matrix = np.multiply(covariances_matrix, differences_matrix)
    return(16*V_matrix.sum())

def V_k_noisy(k, x_densities, y_densities):
    x_density = x_densities[k]
    y_density = y_densities[k]
    
    D_x = np.real(x_density[0,0])
    R_x = np.real(x_density[0,1])+0.5
    I_x = 0.5-np.imag(x_density[0,1])
    D_y = np.real(y_density[0,0])
    R_y = np.real(y_density[0,1])+0.5
    I_y = 0.5-np.imag(y_density[0,1])

    differences_vec = np.array([np.abs(D_x-D_y), np.abs(R_x-R_y), np.abs(I_x-I_y),np.abs(D_x-D_y), np.abs(R_x-R_y), np.abs(I_x-I_y)])
    differences_matrix = np.outer(differences_vec, differences_vec)

    V_matrix = differences_matrix
    return(16*V_matrix.sum())

def kernel_value_from_densities(x_density, y_density, gamma=1):
    dens_diff = x_density - y_density
    argument = np.array([np.power(np.linalg.norm(rho),2) for rho in dens_diff]).sum()
    return(np.exp(-gamma*argument))

def N_spread(x_density, y_density, Delta, P_target, gamma=1, epsilon = 1):
    n = x_density.shape[0]
    k = kernel_value_from_densities(x_density, y_density, gamma=gamma)
    V = np.array([V_k(k,x_density,y_density) for k in range(n)]).sum()
    
    N = (np.power(gamma*k,2)*n*V)/(np.power(epsilon*Delta,2)*(1-P_target))
    return(N)

def N_spread_noisy(x_density, y_density, Delta, P_target, gamma=1, epsilon = 1):
    n = x_density.shape[0]
    k = kernel_value_from_densities(x_density, y_density, gamma=gamma)
    V = np.array([V_k_noisy(k,x_density,y_density) for k in range(n)]).sum()
    
    N = (np.power(gamma*k,2)*n*V)/(np.power(epsilon*Delta,2)*(1-P_target))
    return(N)