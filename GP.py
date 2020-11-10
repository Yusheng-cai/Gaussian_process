import numpy as np
from scipy.optimize import minimize
from autograd.numpy.linalg import lstsq

class GP:
    def __init__(self,X,y,kernel,hyp):
        """
        X: input array with shape (N,d)
        y: measurement with shape (N,1), the data should be normalized with a mean of 0 and std of 1 
        """
        self.X,self.y = X,y
        self.D = X.shape[1]
        self.kernel = kernel
        self.hyp = hyp

    def NLL(self,hyp):
        """
        calculates the negative log likelihood for a GP process p(y|x) = \int p(y|f,x)p(f|x)df
        hyp: hyper parameters in shape (d+2,) where hyp = (sigma_f,l1,l2,...,ld,sigma_n)

        returns: 
            Negative log likelihood as a float
        """
        X = self.X
        y = self.y
        N = self.X.shape[0]
        jitter = 1e-8

        sigma_n = hyp[-1]
        
        K = self.kernel(X,X,hyp[:-1])+np.exp(sigma_n)*np.eye(N)
        L = np.linalg.cholesky(K+jitter*np.eye(N))
        
        K_inv_y = np.linalg.lstsq(L.T,np.linalg.lstsq(L,y,rcond=-1)[0],rcond=-1)[0] #(N,1)

        # log(det(S)) = 2*sum(log(diag(L)))
        return (np.sum(np.log(np.diag(L))) + 0.5*np.dot(y.T,K_inv_y) + 0.5*np.log(2*np.pi)*N)[0,0]
    
    def train(self,verbose=False):
        """
        optimize NLL equation using LBFGS method

        returns:
            parameters (d+2,1) [sigma_f,l1,...,ld,sigma_n]
        """
        if verbose:
            results = minimize(self.NLL,self.hyp,method='L-BFGS-B',callback=self.callback)
        else:
            results = minimize(self.NLL,self.hyp,method='L-BFGS-B')

        self.hyp = results.x

        return results.x

    def predict(self,xstar):
        """
        predict GP output with the conditional distribution p(f(x*)|y,x)
                mu(f(x*)) = k(x*,x)K^{-1}y
                cov(f(x*)) = k(x*,x*)-k(x*,x)K^{-1}k(x,x*)

        xstar: of shape (Nstar,d)
        
        returs:
            mu(f(x*)),cov(f(x*))
        """
        X = self.X
        y = self.y
        N = X.shape[0]
        hyp = self.hyp[:-1]
        sigma_n = self.hyp[-1]
        jitter = 1e-8

        K = self.kernel(X,X,hyp)+np.exp(sigma_n)*np.eye(N)
        L = np.linalg.cholesky(K+jitter*np.eye(N))

        K_inv = np.linalg.solve(L.T,np.linalg.solve(L,np.eye(N)))
        kxs_x = self.kernel(xstar,X,hyp)
        mu = kxs_x.dot(K_inv).dot(y)

        kxs_xs = self.kernel(xstar,xstar,hyp)
        cov = kxs_xs - kxs_x.dot(K_inv).dot(kxs_x.T)

        return mu,cov
        
    def draw_prior(self,x,num_samples=100):
        """
        draw samples from prior distribution

        x: the x points that we want to calculate prior at (N,1) 
        num_samples: the number of samples that we want to draw from the prior

        returns:
            a matrix containing all prior samples with shape (num_samples,N)
        """
        N = x.shape[0]
        mu = np.zeros((N,))
        cov = self.kernel(x,x,self.hyp[:-1])
        samples = np.random.multivariate_normal(mu,cov,num_samples)

        return samples

    def draw_posterior(self,xstar,num_samples=100):
        """
        draw from the posterior Gaussian p(f(x*)|y)

        xstar: input data of shape (Nstar,d)

        output:
            samples drawn from posterior distribution (num_samples,Nstar)
        """
        mu,cov = self.predict(xstar)
        samples = np.random.multivariate_normal(mu[:,0],cov,size=num_samples)

        return (mu,cov,samples)

    def callback(self,params):
        print("Log likelihood {}".format(self.NLL(params)))


class multifidelity_GP:
    def __init__(self,x_l,y_l,x_h,y_h,kernel,hyp):
        """
        x_l: x data of the low fidelity model (shape (N_l,d))
        y_l: y data of the low fidelity model (shape (N_l,1))
        
        x_h: x data of the high fidelity model (shape (N_h,d))
        y_h: y data of the high fidelity model (shape (N_h,1))

        kernel: the kernel that will be used with multifidelity_GP
        hyp: in the form of (ln(sigma_fl),ln(theta1_l),...,ln(thetad_l),\
                            ln(sigma_fh),ln(theta1_h),....,ln(thetad_h),\
                            ln(sigma_nl),ln(sigma_nh),rho) (shape (2d+5,))
        """
        self.D = x_l.shape[1]
        self.nl = x_l.shape[0]
        self.nh = x_h.shape[0]

        self.xl = x_l
        self.yl = y_l
        self.xh = x_h
        self.yh = y_h
        
        self.kernel = kernel
        self.hyp = hyp

    def NLL(self,hyp):
        """
        calculates the negative log likelihood of multifidelity_GP
        """
        sigma_nl = np.exp(hyp[-3])
        sigma_nh = np.exp(hyp[-2])
        rho = hyp[-1]
        jitter = 1e-8
        N = self.nl+self.nh

        hyp_l = hyp[:self.D+1]
        hyp_h = hyp[self.D+1:2*self.D+2]

        kll = self.kernel(self.xl,self.xl,hyp_l)+\
                sigma_nl*np.eye(self.nl) # kl(xl,xl,thetal)+sigma_nl^2*I
        klh = rho*self.kernel(self.xl,self.xh,hyp_l) #rho*kl(xl,xh,thetal)
        khl = klh.T
        khh = (rho**2)*self.kernel(self.xh,self.xh,hyp_l)+\
                self.kernel(self.xh,self.xh,hyp_h)+\
                sigma_nh*np.eye(self.nh) # rho^2*kl(xh,xh,thetal)+kh(xh,xh,thetah)+sigmanh^2I

        K = np.vstack((np.hstack((kll,klh)),np.hstack((khl,khh))))
        
        y = np.vstack((self.yl,self.yh)) #shape(NL+NH,1)

        L = np.linalg.cholesky(K+jitter*np.eye(N))
        Kinv_y = np.linalg.lstsq(L.T,np.linalg.lstsq(L,y,rcond=-1)[0],rcond=-1)[0] 

        logdet_K = np.sum(np.log(np.diag(L)))

        return (logdet_K+0.5*np.dot(y.T,Kinv_y))[0,0]

    def train(self,verbose=False):
        if verbose:
            results = minimize(self.NLL,self.hyp,method='L-BFGS-B',callback=self.callback)
        else:
            results = minimize(self.NLL,self.hyp,method='L-BFGS-B')

        self.hyp = results.x

        return results.x

    def predict(self,xstar):
        """
        Function to predict high fidelity outputs

        xstar: the new points that needs to be predicted (Nstar,1)

        returns: 
            mu and covariance of the posterior distribution
        """
        sigma_nl = np.exp(self.hyp[-3])
        sigma_nh = np.exp(self.hyp[-2])
        rho = self.hyp[-1]
        jitter = 1e-8
        N = self.nl+self.nh
        y = np.vstack((self.yl,self.yh)) 

        hyp_l = self.hyp[:self.D+1]
        hyp_h = self.hyp[self.D+1:2*self.D+2]

        kxs_xs = (rho**2)*self.kernel(xstar,xstar,hyp_l)+self.kernel(xstar,xstar,hyp_h)
        kxs_X = np.hstack((rho*self.kernel(xstar,self.xl,hyp_l),\
                rho**2*self.kernel(xstar,self.xh,hyp_l)+self.kernel(xstar,self.xh,hyp_h)))

        kll = self.kernel(self.xl,self.xl,hyp_l)+\
                sigma_nl*np.eye(self.nl) # kl(xl,xl,thetal)+sigma_nl^2*I
        klh = rho*self.kernel(self.xl,self.xh,hyp_l) #rho*kl(xl,xh,thetal)
        khl = klh.T
        khh = (rho**2)*self.kernel(self.xh,self.xh,hyp_l)+\
                self.kernel(self.xh,self.xh,hyp_h)+\
                sigma_nh*np.eye(self.nh) # rho^2*kl(xh,xh,thetal)+kh(xh,xh,thetah)+sigmanh^2I
        
        K = np.vstack((np.hstack((kll,klh)),np.hstack((khl,khh))))

        L = np.linalg.cholesky(K+jitter*np.eye(N)) 
        Kinv_y = np.linalg.lstsq(L.T,np.linalg.lstsq(L,y,rcond=-1)[0],rcond=-1)[0]
        Kinv_kxs_X = np.linalg.lstsq(L.T,np.linalg.lstsq(L,kxs_X.T,rcond=-1)[0],rcond=-1)[0]

        mu = kxs_X.dot(Kinv_y) #(Nstar,1)
        cov = kxs_xs - kxs_X.dot(Kinv_kxs_X) #(Nstar,Nstar)

        return mu,cov

    def draw_posterior(self,xstar,num_samples=100):
        """
        function that draws the posterior f(xstar|X,y)

        xstar: a matrix of data point with shape (Nstar,d)
        num_samples: number of samples that one wants to draw

        returns: 
                (mu,cov,samples)
                where samples is with shape (num_samples,len(xstar))
        """
        mu,cov = self.predict(xstar)
        samples = np.random.multivariate_normal(mu[:,0],cov,size=num_samples)

        return (mu,cov,samples)

    def callback(self,params):
        print("Log likelihood {}".format(self.NLL(params)))

