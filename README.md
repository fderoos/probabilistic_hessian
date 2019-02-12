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

def rosen_der_noise(x):
    return rosen_der(x)+np.random.randn(*x.shape)

U,S = Estimate_Hesssian(x0,jac=rosen_der_noise,hessp=rosen_hess_prod)
```

```python
# (2) Returned by function
import numpy as np
from probabilistic_hessian import Estimate_Hessian
from scipy.optimize import rosen,rosen_der,rosen_hess_prod

x0=np.random.randn(5)

def rosen_complete(x,v=None):
    f = rosen(x)
    g = rosen_der(x)+np.random.randn(*x.shape)
    if v is None:
        return f, g
    else:
        hv = rosen_hess_prod(x,v)
        return f, g, hv


U,S = Estimate_Hesssian(x0,fun=rosen_complete,jac=True,hessp=True)
```



## Feedback

If you have any questions or suggestions regarding this implementation, please open an issue in [fderoos/probabilistic_hessian][repo] or contact by email (mail to fderoos@tue.mpg.de).

## Citation

If you use the algorithm for your research, please cite the [article][article].


[article]: https://github.com/fderoos/probabilistic_hessian "Available after publication"
[repo]: https://github.com/fderoos/probabilistic_hessian