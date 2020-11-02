import autograd.numpy as np 

def rbf_kernel(x,xstar,hyp):
    """
    Implements the radial basis function for Gaussian Process

    x: with shape (N,d)
    xstar: with shape (Nstar,d)
    hyp: (log(sigma_f),log(l1),log(l2),...) with shape (d+1,)

    returns:
        a covariance matrix with shape (N,Nstar)
    """
    sigma_f = np.exp(hyp[0])
    l = np.exp(hyp[1:]) #shape (d,)

    diff = np.expand_dims(x/l,1) - np.expand_dims(xstar/l,0) #result of shape (N,Nstar,d)

    return sigma_f*np.exp(-0.5*(diff**2).sum(axis=2)) # should be of shape (N,Nstar)

    
def periodic_kernel(x,xstar,hyp):
    """
    Implements the periodic kernel function for Gaussian Process

    x: input data with shape (N,d)
    xstar: inpt data with data (Nstar,d)
    hyp: (log(sigma_f),log(l1),log(l2),...,log(period)) with shape (d+2,)
    
    returns:
        a covariance matrix with shape (N,Nstar)
    """
    sigma_f = np.exp(hyp[0])
    N = x.shape[0]
    Nstar = xstar.shape[0]
    l = np.exp(hyp[1:-1]) #shape (d,)
    l = np.repeat(np.repeat(l[np.newaxis,:],Nstar,axis=0)[np.newaxis,:],N,axis=0) #shape (N,Nstar,d)
    period = np.exp(hyp[-1])

    diff = np.sin(np.pi*np.abs(np.expand_dims(x,1) - np.expand_dims(xstar,0))/period)/l #result of shape (N,Nstar,d)
    K = sigma_f*np.exp(-2*(diff**2).sum(axis=2)) #should be of shape (N,Nstar)

    return K
    
