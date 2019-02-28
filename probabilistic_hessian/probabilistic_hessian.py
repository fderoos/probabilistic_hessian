
from __future__ import division, print_function, absolute_import


from scipy._lib.six import callable


import scipy.linalg as slinalg
import scipy.sparse.linalg as splinalg
import numpy as np


def Estimate_Scalar_Parameters(jac=None, hessp=None, nbr_est_runs=12, **kwargs):
    """Empirically find scalar estimates of mean, variance and noise"""
    sTs = 0
    sHs = 0
    sHHs = 0
    prior_mean = 0
    acc_sTs = 0
    for i in np.arange(1, nbr_est_runs+1):
        s = jac(**kwargs)
        Hs = hessp(s, **kwargs)

        if i == 1:
            N = s.flatten().shape[0]
            grad = np.zeros_like(s)
            grad_squared = np.zeros_like(s)

        sTs = np.linalg.norm(s)**2
        sHs += s.T.dot(Hs)  # /sTs
        sHHs += Hs.T.dot(Hs)  # /sTs
        acc_sTs += sTs

        grad += s
        grad_squared += s**2

    var_grad = np.max(np.abs(((grad/i)**2-grad_squared/i)))
    var_grad /= np.sqrt(acc_sTs)/(i)

    prior_scalar_variance = np.abs((sHs/acc_sTs)).flatten()

    prior_mean = np.sqrt(sHs/sHHs).flatten()

    print('prior variance:  %3.2e prior mean:  %3.2e noise variance:  %3.2e' %
          (prior_scalar_variance, 1/prior_mean, var_grad))

    return prior_mean, prior_scalar_variance, var_grad, N


def Valid_Cov(W,N):

    if W.ndim==2 and np.allclose(W.shape,N) and np.allclose(W,W.T):
        return True
    else:
        return False


class MemoizeJacHessp(object):
    """ Decorator that caches the gradient and Hesssian-vector product of function each time it
    is called. """

    def __init__(self, fun, jac, hessp):
        self.fun = fun
        if jac:
            self.jac = None
        if hessp:
            self.hessp = None
        self.saved_gradient = False

    def __call__(self, v=None, **kwargs):

        if v is None:
            fg = self.fun(v=None, **kwargs)
        else:
            fg = self.fun(v=v, **kwargs)
            self.hessp = fg[2]
            self.saved_gradient = True
        self.jac = fg[1]
        return fg[0]

    def derivative(self, **kwargs):
        if self.saved_gradient:
            self.saved_gradient = False
            return self.jac
        else:
            self(**kwargs)
            return self.jac

    def hessian_product(self, v, **kwargs):
        self(v=v, **kwargs)
        return self.hessp


def Estimate_Hessian(fun=None, hess_rank=1, nbr_hv=3, prior_mean=None, prior_var=None, noise_est=None, jac=None, hessp=None, **kwargs):
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
                memoizeJac = True
            else:
                raise ValueError('fun must be callable for boolean jac')
        else:
            jac = None

    if not callable(hessp):
        if bool(hessp):
            if callable(fun):
                memoizeHessp = True
            else:
                raise ValueError('fun must be callable for boolean hessp')
        else:
            hessp = None

    if jac is None:
        raise ValueError('Jacobian is required for Hessian estimation')
    if hessp is None:
        raise ValueError(
            'Hessian-vector product handle is required for Hessian estimation')

    if memoizeHessp or memoizeJac:
        fun = MemoizeJacHessp(fun, jac=memoizeJac, hessp=memoizeHessp)
        if memoizeJac:
            jac = fun.derivative

        if memoizeHessp:
            hessp = fun.hessian_product

    fprime = jac
    fhess_p = hessp

    if prior_mean is None or prior_var is None or noise_est is None:
        # Estimate parameters
        prior_inv_mean_est, prior_var_est, noise_var_est, N = Estimate_Scalar_Parameters(
            jac=fprime, hessp=fhess_p, **kwargs)

    M = nbr_hv

    if prior_mean is None:
        prior_scalar_mean = 1/prior_inv_mean_est
    else:
        prior_scalar_mean = np.float(prior_mean)

    if prior_var is None:
        prior_var = prior_var_est
        W = prior_var*np.eye(N)
    elif np.ndim(prior_var) == 0:
        if prior_var > 0:
            W = prior_var*np.eye(N)
        else:
            raise ValueError(
                'scalar prior_var must be positive')
    elif np.ndim(prior_var) == 1 and prior_var.shape[0] == N:
        W = np.diag(np.abs(prior_var))
    elif Valid_Cov(prior_var, N):
        W = np.copy(prior_var)
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
    g = -fprime(**kwargs)

    # First direction along gradient
    si = A0_inv.dot(g)

    for i in (np.arange(M)+1):

        # Call Hessian-vector product first to memoize gradient
        yi = fhess_p(si, **kwargs)
        g = -fprime(**kwargs)
        # yi = fhess_p(si, **kwargs)

        S[:, i-1] = si
        Y[:, i-1] = yi

        WS[:, i-1] = W.dot(si)
        STWS[i-1, i-1] = si.dot(WS[:, i-1])

        if i != 1:
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
        return U.dot((V.dot(v))) + prior_scalar_mean*v

    def mvt(v):
        return V.T.dot((U.T.dot(v))) + prior_scalar_mean*v

    Hessian_op = splinalg.LinearOperator((N, N), matvec=mv, rmatvec=mvt)

    U_svd, S_svd, _ = splinalg.svds(Hessian_op, k=hess_rank)

    return U_svd, S_svd


def Estimate_Hessian_Scalar(fun=None, hess_rank=1, nbr_hv=3, prior_mean=None, prior_var=None, noise_est=None, jac=None, hessp=None, **kwargs):
    """ 
    Reduced memory version of 'Estimate_Hessian' which relies on scalar values instead of matrices to find the 'hess_rank' leading eigen-values and eigen-vectors from 'nbr_hv' noisy Hessian-vector products.
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
                memoizeJac = True
            else:
                raise ValueError('fun must be callable for boolean jac')
        else:
            jac = None

    if not callable(hessp):
        if bool(hessp):
            if callable(fun):
                memoizeHessp = True
            else:
                raise ValueError('fun must be callable for boolean hessp')
        else:
            hessp = None

    if jac is None:
        raise ValueError('Jacobian is required for Hessian estimation')
    if hessp is None:
        raise ValueError(
            'Hessian-vector product handle is required for Hessian estimation')

    if memoizeHessp or memoizeJac:
        fun = MemoizeJacHessp(fun, jac=memoizeJac, hessp=memoizeHessp)
        if memoizeJac:
            jac = fun.derivative

        if memoizeHessp:
            hessp = fun.hessian_product

    fprime = jac
    fhess_p = hessp

    if prior_mean is None or prior_var is None or noise_est is None:
        # Estimate parameters
        prior_inv_mean_est, prior_var_est, noise_var_est, N = Estimate_Scalar_Parameters(
            jac=fprime, hessp=fhess_p, **kwargs)

    M = nbr_hv

    if prior_mean is None:
        prior_scalar_mean = 1/prior_inv_mean_est
    else:
        prior_scalar_mean = np.float(prior_mean)

    if prior_var is None:
        prior_var = prior_var_est
        W = prior_var  # *np.eye(N)
    elif np.ndim(prior_var) == 0:
        if prior_var > 0:
            W = prior_var  # *np.eye(N)
        else:
            raise ValueError(
                'scalar prior_var must be positive')
    else:
        raise ValueError(
            'prior_var must be a positive scalar')

    if noise_est is None:
        noise_est = noise_var_est
        L = noise_est  # *np.eye(N)
    elif np.ndim(noise_est) == 0:
        if noise_est > 0:
            L = noise_est  # *np.eye(N)
        else:
            raise ValueError(
                'noise_est must be positive')
    else:
        raise ValueError(
            'noise_est must be a positive scalar or a positive definite matrix of same dimension as x0')

    ############################
    # Start of algorithm
    ############################
    S = np.zeros((N, M))
    Y = np.zeros((N, M))

    STS = np.zeros((M, M))
    STLS_diag = np.zeros(M)

    A0 = prior_scalar_mean  # *np.eye(N)
    A0_inv = 1.0/prior_scalar_mean  # *np.eye(N)

    L = noise_var_est  # *np.eye(N)

    # [Dl, Vl] = slinalg.eigh(W, L)
    # Generaized eigen-decomposition of scalar matrices
    Dl = W/L
    Vl = 1/np.sqrt(L)
    g = -fprime(**kwargs)

    # First direction along gradient
    si = A0_inv*g

    for i in (np.arange(M)+1):

        # Call Hessian-vector product first to memoize gradient
        yi = fhess_p(si, **kwargs)
        g = -fprime(**kwargs)


        S[:, i-1] = si
        Y[:, i-1] = yi

        STS[i-1, i-1] = si.dot(S[:, i-1])


        if i != 1:
            STS_irow = si.T.dot(S[:, :i])

            STS[i-1, :i] = STS_irow
            STS[:i, i-1] = STS_irow.T


        STLS_diag[i-1] = L*STS[i-1, i-1]  # si.T.dot(Lsi)

        [Dr, Vr] = slinalg.eigh(W*STS[:i, :i], np.diag(STLS_diag[:i]))

        Delta = (Y[:, :i]-A0*(S[:, :i]))

        Hadamard = 1.0/(Dl*Dr[np.newaxis, :]+1)

        X = Vl*((Hadamard*(Vl*(Delta.dot(Vr)))).dot(Vr.T))

        # Apply estimated inverse to gradient for new search direction
        si = 1/prior_scalar_mean * \
            (g-W**2*X.dot(np.linalg.solve(np.eye(i) +
                                          A0_inv*W**2*S[:, :i].T.dot(X), S[:, :i].T.dot(1/prior_scalar_mean*g))))

    # Use stored vectors to build low-rank estimate and

    def mv(v):
        return W**2*X.dot((S.T.dot(v))) + prior_scalar_mean*v

    def mvt(v):
        return W**2*S.dot((X.T.dot(v))) + prior_scalar_mean*v

    Hessian_op = splinalg.LinearOperator((N, N), matvec=mv, rmatvec=mvt)

    U_svd, S_svd, _ = splinalg.svds(Hessian_op, k=hess_rank)

    return U_svd, S_svd
    

