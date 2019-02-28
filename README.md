# Probabilistic Hessian Estimation

This is a scipy-inspired implementation of the Hessian-inference algorithm presented in the paper [Active Probabilistic Inference on Matrices for Pre-Conditioning in Stochastic Optimization][article]




## Installation

Install via

    pip install git+https://github.com/fderoos/probabilistic_hessian.git



## Usage

The ``Probabilistic_Hessian`` module contains the function ``Estimate_Hessian`` which requires handles to the gradient and a Hessian-vector product similar to `scipy.optimize.minimize`. See below for two examples.

```python
# (1) With handles
import numpy as np
from probabilistic_hessian import Estimate_Hessian
from scipy.optimize import rosen_der,rosen_hess_prod

x0=np.random.randn(5)

def rosen_grad(x):
    g = rosen_der(x)
    return g + 0.1 * np.random.randn(*g.shape)
    
def rosen_hessp(p,x):
    return rosen_hess_prod(x,p)

U,S = Estimate_Hesssian(jac=rosen_grad,hessp=rosen_hessp,x=x0)
```

```python
# (2) Collectively returned by function
import numpy as np
from probabilistic_hessian import Estimate_Hessian
from scipy.optimize import rosen,rosen_der,rosen_hess_prod

x0=np.random.randn(5)

def rosen_complete(v=None,**kwargs):
    f = rosen(**kwargs)
    g = rosen_der(**kwargs)
    g+= 0.1*np.random.randn(*g.shape)
    if v is None:
        return f, g
    else:
        hv = rosen_hess_prod(p=v,**kwargs)
        return f, g, hv


U,S = Estimate_Hesssian(fun=rosen_complete,jac=True,hessp=True,x=x0)
```



## Feedback

If you have any questions or suggestions regarding this implementation, please open an issue in [fderoos/probabilistic_hessian][repo] or contact by email (mail to fderoos@tue.mpg.de).

## Citation

If you use the algorithm for your research, please cite the [article][article].


[article]: https://arxiv.org/abs/1902.07557 "Preprint"
[repo]: https://github.com/fderoos/probabilistic_hessian