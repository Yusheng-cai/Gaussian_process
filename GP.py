import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize

class GP:
    def __init__(self,X,y,kernel):
        """
        X: input array with shape (N,d)
        y: measurement with shape (N,1), the data should be normalized with a mean of 0 and std of 1 
        """
        self.X,self.y = X,y
        self.D = X.shape[1]
        self.kernel = kernel

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
        
        K_inv_y = np.linalg.solve(L.T,np.linalg.solve(L,y)) #(N,1)

        # log(det(S)) = 2*sum(log(diag(L)))
        return (np.sum(np.log(np.diag(L))) + 0.5*np.dot(y.T,K_inv_y) - 0.5*np.log(2*np.pi)*N)[0,0]
    
    def train(self,hyp0):
        """
        optimize NLL equation using LBFGS method

        returns:
            parameters (d+2,1) [sigma_f,l1,...,ld,sigma_n]
        """
        results = minimize(value_and_grad(self.NLL),hyp0,method='L-BFGS-B',jac=True,callback=self.callback)
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

    def draw_posterior(self,xp,num_samples=100):
        """
        draw from the posterior Gaussian p(f(x*)|y)

        xp: input data of shape (N*,d)

        output:
            samples drawn from posterior distribution (num_samples,N*)
        """
        mu,cov = self.predict(xp)
        samples = np.random.multivariate_normal(mu[:,0],cov,size=num_samples)

        return samples

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

    def NLL(self):
        """
        calculates the negative log likelihood of multifidelity_GP
        """
        sigma_nl = np.exp(self.hyp[-3])
        sigma_nh = np.exp(self.hyp[-2])
        rho = np.exp(self.hyp[-1])
        jitter = 1e-8
        N = self.nl+self.nh

        hyp_l = self.hyp[:self.D+1]
        hyp_h = self.hyp[self.D+1:2*self.D+2]

        kll = self.kernel(self.xl,self.xl,hyp_l)+\
                (sigma_nl**2)*np.eye(self.nl) # kl(xl,xl,thetal)+sigma_nl^2*I
        klh = rho*self.kernel(self.xl,self.xh,hyp_l) #rho*kl(xl,xh,thetal)
        khl = klh.T
        khh = (rho**2)*self.kernel(self.xh,self.xh,hyp_l)+\
                self.kernel(self.xh,self.xh,hyp_h)+\
                (sigma_nh**2)*np.eye(self.nh) # rho^2*kl(xh,xh,thetal)+kh(xh,xh,thetah)+sigmanh^2I
        K = np.block([
            [kll,klh],
            [khl,khh]]) #shape(NL+NH,NL+NH)
        
        y = np.vstack((self.yl,self.yh)) #shape(NL+NH,1)

        L = np.linalg.cholesky(K+jitter*np.eye(N))

        Kinv_y = np.linalg.solve(L,np.linalg.solve(L.T,y))
        logdet_K = np.sum(np.log(np.diag(L)))

        return (0.5*logdet_K+0.5*np.dot(y.T,Kinv_y)+0.5*N*np.log(2*np.pi))[0,0]
    

