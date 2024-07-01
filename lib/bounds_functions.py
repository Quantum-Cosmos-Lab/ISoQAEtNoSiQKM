import numpy as np

def beta_Haar(n):
    return(1.0/(np.power(2,n-1)*(np.power(2,n)+1)))

def expressibility_bound(n,e):
    return(beta_Haar(n)+e*(e+2*np.sqrt(beta_Haar(n))))

def beta_Haar_projected(n):
    return(3.0/(np.power(2,n+1)+2))