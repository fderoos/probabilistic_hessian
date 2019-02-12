
from __future__ import division, print_function, absolute_import


from scipy._lib.six import callable


import scipy.linalg as slinalg
import scipy.sparse.linalg as splinalg
import numpy as np




def Estimate_Scalar_Parameters(x0, jac=None, hessp=None,args=(), nbr_est_runs=4):
    """Empirically find scalar estimates of mean, variance and noise"""

    grad = np.zeros(x0.shape)  
    grad_squared = np.zeros(x0.shape)  
    sTs = 0
    sHs = 0
    sHHs = 0
    prior_mean = 0
    acc_sTs = 0

    for i in np.arange(1,nbr_est_runs+1):
        s=jac(x0)
        Hs=hessp(x0,s,*args)

        sTs = np.linalg.norm(s)**2
        sHs += s.T.dot(Hs)  # /sTs
        sHHs += Hs.T.dot(Hs)  # /sTs
        acc_sTs+=sTs

        grad += s
        grad_squared += s**2



    var_grad = np.percentile(np.abs(((grad/i)**2-grad_squared/i)), q=50)
    var_grad /= np.sqrt(acc_sTs)/(i)

    prior_scalar_variance = np.abs((sHs/acc_sTs)).flatten()

    prior_mean = np.sqrt(sHs/sHHs).flatten()

    print('prior variance:  %3.2f prior mean:  %3.2f noise variance:  %3.2f'%(prior_scalar_variance,1/prior_mean,var_grad))

    return prior_mean, prior_scalar_variance, var_grad


def Valid_Cov(W,N):

    if W.ndim==2 and np.allclose(W.shape,N) and np.allclose(W,W.T):
        return True
    else:
        return False



    


class MemoizeJacHessp(object):
    """ Decorator that caches the gradient and Hesssian-vector product of function each time it
    is called. """

    def __init__(self, fun,jac,hessp):
        self.fun = fun
        if jac:
            self.jac = None
        if hessp:
            self.hessp = None
            self.v = None
        self.x = None
        

    def __call__(self, x, v=None,*args):
        self.x = np.asarray(x).copy()
        # print(v.__class__)
        # if v is None:
        #     print(v)
        if v is None:
            fg = self.fun(x, v=None, *args)
        else:
            self.v = np.asarray(v).copy()
            fg = self.fun(x,v, *args)
            self.hessp=fg[2]
        self.jac = fg[1]
        return fg[0]

    def derivative(self, x, *args):
        if self.jac is not None and np.alltrue(x == self.x):
            return self.jac
        else:
            self(x, *args)
            return self.jac

    def hessian_product(self, x, v, *args):
        if self.hessp is not None and np.alltrue(x == self.x) and np.alltrue(v == self.v):
            return self.hessp
        else:
            self(x, v,*args)
            return self.hessp
            


def Estimate_Hessian(x0, fun=None,args=(),hess_rank=1, nbr_hv=3,prior_mean=None,prior_var=None, noise_est=None,jac=None, hessp=None):
    """ 
    Estimation of 'hess_rank' leading eigen-values and eigen-vectors from 'nbr_hv' noisy Hessian-vector products.
    The arguments 'prior_mean', 'prior_var' and 'noise_est' can be set to scalar values or matrices and will be estimated if not provided.  
    Note that a handle to `jac` and `hessp` parameters (Jacobian, Hessian-vector product) is required or
    should indicate as Boolean if it is returned by 'fun'.
    """
    memoizeJac = False
    memoizeHessp = False

    # Check required handles 
    if not callable(jac):
        if bool(jac):
            if callable(fun):
                memoizeJac=True
            else:
                raise ValueError('fun must be callable for boolean jac')
        else:
            jac = None

    if not callable(hessp):
        if bool(hessp):
            if callable(fun):
                memoizeHessp=True
            else:
                raise ValueError('fun must be callable for boolean hessp')
        else:
            hessp = None

    if jac is None:
        raise ValueError('Jacobian is required for Hessian estimation')
    if hessp is None:
        raise ValueError('Hessian-vector product handle is required for Hessian estimation')

    if memoizeHessp or memoizeJac:
        fun = MemoizeJacHessp(fun,jac=memoizeJac,hessp=memoizeHessp)
        if memoizeJac:
            jac=fun.derivative

        if memoizeHessp:
            hessp=fun.hessian_product


    fprime = jac
    fhess_p = hessp


    x0 = np.asarray(x0).flatten()
    M = nbr_hv
    N = x0.shape[0]

    if prior_mean is None or  prior_var is None or noise_est is None: 
        # Estimate parameters
        prior_inv_mean_est,prior_var_est,noise_var_est = Estimate_Scalar_Parameters(x0, jac=fprime, hessp=fhess_p)
        print(prior_inv_mean_est)


    if prior_mean is None:
        prior_scalar_mean = 1/prior_inv_mean_est
    else:
        prior_scalar_mean=np.float(prior_mean)



    if prior_var is None:
        prior_var = prior_var_est
        W = prior_var*np.eye(N)
    elif np.ndim(prior_var)==0:
        if prior_var>0:
            W = prior_var*np.eye(N)
        else:
            raise ValueError(
                'scalar prior_var must be positive')
    elif np.ndim(prior_var)==1 and prior_var.shape[0]==N:
        W=np.diag(np.abs(prior_var))
    elif Valid_Cov(prior_var,N):
        W=np.copy(prior_var)
    else:
        raise ValueError(
            'prior_var must be a positive scalar or a positive definite matrix of same dimension as x0')

    if noise_est is None:
        noise_est = noise_var_est
        L = noise_est*np.eye(N)
    elif np.ndim(noise_est) == 0:
        if noise_est > 0:
            L = noise_est*np.eye(N)
        else:
            raise ValueError(
                'noise_est must be positive')
    elif Valid_Cov(noise_est, N):
        L = np.copy(noise_est)
    else:
        raise ValueError(
            'noise_est must be a positive scalar or a positive definite matrix of same dimension as x0')


    # Start of algorithm
    S = np.zeros((N, M))
    Y = np.zeros((N, M))
    WS = np.zeros((N, M))

    STWS = np.zeros((M, M))
    STLS_diag = np.zeros(M)

    A0 = prior_scalar_mean*np.eye(N)
    A0_inv = 1.0/prior_scalar_mean*np.eye(N)


    L = noise_var_est*np.eye(N)


    [Dl, Vl] = slinalg.eigh(W, L)
    g = -fprime(x0)

    # First direction along gradient
    si = A0_inv.dot(g)

    for i in (np.arange(M)+1):


        g=-fprime(x0,*args)
        yi=fhess_p(x0,si,*args)


        S[:, i-1] = si  
        Y[:, i-1] = yi  


        WS[:, i-1] = W.dot(si)
        STWS[i-1,i-1] = si.dot(WS[:,i-1])

        if i !=1:
            STWS_irow = si.T.dot(WS[:, :i])

            STWS[i-1, :i] = STWS_irow
            STWS[:i, i-1] = STWS_irow.T

        Lsi = L.dot(si)
        STLS_diag[i-1] = si.T.dot(Lsi)

        [Dr, Vr] = slinalg.eigh(STWS[:i, :i], np.diag(STLS_diag[:i]))

        Delta = (Y[:, :i]-A0.dot(S[:, :i]))


        Hadamard = 1.0/(Dl[:, np.newaxis]*Dr[np.newaxis, :]+1)

        X_had_sol = Vl.dot((Hadamard*(Vl.T.dot(Delta.dot(Vr)))).dot(Vr.T))

        U = W.dot(X_had_sol)
        V = WS[:, :i].T

        # Apply estimated inverse to gradient for new search direction
        si = 1/prior_scalar_mean * \
            (g-U.dot(np.linalg.solve(np.eye(i) +
                                         V.dot(A0_inv.dot(U)), V.dot(1/prior_scalar_mean*g))))


    # Use stored vectors to build low-rank estimate and 
    def mv(v):
        return U.dot((V.dot(v)))   + prior_scalar_mean*v

    def mvt(v):
        return V.T.dot((U.T.dot(v)))   + prior_scalar_mean*v

    Hessian_op = splinalg.LinearOperator((N, N), matvec=mv, rmatvec=mvt)

    U_svd, S_svd, _ = splinalg.svds(Hessian_op, k=hess_rank)

    return U_svd, S_svd, U_svd.T
   
    

