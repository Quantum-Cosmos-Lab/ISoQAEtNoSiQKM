import numpy as np
from scipy.special import betainc

def CDF(N, mu, p):
    """
    Cumulative distribution function for a binomial distribution
    """
    return(betainc(N-np.floor(N*mu), 1+np.floor(N*mu), 1-p))

def P_over_mu(N,mu,p):
    """
    Probability of obtaining measured value greater than mu in N trials
    """
    return(1-CDF(N,mu,p))

def P_under_mu(N,mu,p):
    """
    Probability of obtaining measured value lower than mu in N trials
    """
    return(betainc(N-np.floor(N*mu-1), 1+np.floor(N*mu-1), 1-p))

def P_mu(N,mu,p):
    """
    Universal function for computing probability of obtaining measured value different than mu in N trials
    """
    if(p>mu):
        return(P_over_mu(N,mu,p))
    if(p<mu):
        return(P_under_mu(N,mu,p))
    else:
        return(1)
    
# ----

def find_interval_P_mu(P,mu,p):
    """
    Find the interval boundaries for N for further bisection. The values are increased 10 fold with every loop iteration.
    """
    lower = 1
    upper = 10
    
    if(np.abs(mu-p)<1.e-20):
        return(lower,upper)

    while(upper < 1.e20):
        if((P_mu(lower,mu,p)<P) and (P_mu(upper,mu,p)>=P)):
            return(lower, upper)
        else:
            lower = upper
            upper = 10*upper
    print('N not found withing the given accuracy')
    return(0,0)

def bisection_P_mu(P,mu,p,N_lower,N_upper):
    """
    Get N from the expected probability of measuring different value than mu.
    Returns 0 for the case when p==mu
    """
    if(N_lower==N_upper):
        return(0)
    
    lower = N_lower
    upper = N_upper

    while(np.abs(upper-lower)>1):
        mid = (lower+upper)/2
        P_lower = P_mu(lower,mu,p)
        P_upper = P_mu(upper,mu,p)
        P_mid = P_mu(mid,mu,p)

        if( (P_lower-P)*(P_mid-P)<0 ):
            upper = mid
        else:
            lower = mid
    return(np.ceil(mid))

# -- one function

# -- single value of reduced density matrix

def get_N_success_rate(P,mu,p):
    N_lower, N_upper = find_interval_P_mu(P,mu,p)
    N = bisection_P_mu(P,mu,p,N_lower,N_upper)
    return(N)

# -- single reduced density matrix

def get_N_success_rate_rdm(rho_k, P_SR, mu):
    """
    returns: (3,) array
    """
    rho_D = np.real(rho_k[0,0])
    rho_R = np.real(rho_k[0,1])
    rho_I = np.imag(rho_k[0,1])

    M_D = rho_D
    M_R = rho_R + 1/2
    M_I = 1/2 - rho_I

    Ns = [get_N_success_rate(P_SR, mu, M) for M in [M_D, M_R, M_I]]
    return(np.array(Ns))

# -- single data point - n 1-qubit rdms

def get_N_success_rate_x(rhos_x, P_SR, mu):
    """
    Standard input + an iterable collection of 1-qubit rdms for a given data point.
    returns: (n,3) array of N_SR for a given data point.
    """
    return(np.array([get_N_success_rate_rdm(M, P_SR, mu) for M in rhos_x]))

# -- whole data set

def get_N_success_rate_array(rhos, P_SR, mu):
    """
    input: rhos is an (m,n,2,2) array
    returns: (m,n,3) array of N_SRs
    """
    return(np.array([get_N_success_rate_x(rhos_x, P_SR, mu) for rhos_x in rhos]))

# -- representative average kernel argument

def get_rho_n(k_ie, n, gamma=1):
    av_rho = np.sqrt(-1.0*(np.log(k_ie))/(6*n*gamma))
    return(np.median(av_rho))

def get_rho(k_array, n_range):
    representative_rhos = [get_rho_n(k_array[i], n_range[i], gamma=1) for i in range(len(n_range)) ]
    representative_rhos = np.array(representative_rhos)
    return(representative_rhos)