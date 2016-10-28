import numpy as N
import pdb

def ApproximateJacobian(f, x, dx=1e-6):
    """Return an approximation of the Jacobian Df(x) as a numpy matrix"""
    try:
        n = len(x)
    except TypeError:
        n = 1
    fx = f(x)
    Df_x = N.matrix(N.zeros((n,n)))
    for i in range(n):
        v = N.matrix(N.zeros((n,1)))
        v[i,0] = dx
        Df_x[:,i] = (f(x + v) - fx)/dx
    return Df_x

class Polynomial(object):
    """Callable polynomial object.

    Example usage: to construct the polynomial p(x) = x^2 + 2x + 3,
    and evaluate p(5):

    p = Polynomial([1, 2, 3])
    p(5)"""

    def __init__(self, coeffs):
        self._coeffs = coeffs

    def __repr__(self):
        return "Polynomial(%s)" % (", ".join([str(x) for x in self._coeffs]))

    def f(self,x):
        ans = self._coeffs[0]
        for c in self._coeffs[1:]:
            ans = x*ans + c
        return ans

    def __call__(self, x):
        return self.f(x)
        
class PolynomialJac(Polynomial):
    """Returns analytic Jacobian function for the given polynomial, which is an instance
    of the polynomial class. """
    def f(self,x):
        d = len(self._coeffs)
        if d == 1:
            df = 0.
        else:
            df = self._coeffs[d-2]
            for j in range(d-2):
                df = df + (d-1-j)*self._coeffs[j]*x
        return df

    
class Sinusoid1D(object):
    """Callable sinusoid object with amplitude a, frequency f, and phase p.

    Example usage: to construct the sinusoid p(x) = a*sin(f*x+p)
    and evaluate p(5):

    s = Sinusoid1D(a,f,p)
    s(5)"""
    def __init__(self, a=1., f=1., p=0.):
        if a < 0 or f < 0:
            raise RuntimeError('Amplitude and frequency must be positive!')
        self._a = a
        self._f = f
        self._p = p
        
    def f(self,x):
        ans = self._a * N.sin(x*self._f + self._p)
        return ans
    
    def __call__(self, x):
        return self.f(x)
        
class Sinusoid1DJac(Sinusoid1D):
    """Derived class of Sinusoid1D; computes the derivative of a 1-D sinusoid 
    given the inputs that define the original sinusoid. """    
    def f(self,x):
        ans = self._a * self._f * N.sin(x*self._f + self._p + N.pi/2.)
        return ans
        
class LinearMap(object):
    """General linear function mapping Rn -> Rn, specified by a transformation 
    matrix A(nxn).

    Example usage: l = LinearMap(A)"""

    def __init__(self, A):
        if A.shape[0] != A.shape[1]:
            raise RuntimeError('Make sure that A is nxn!')
        self._A = A

    def f(self,x):
        ans = self._A*x
        return ans

    def __call__(self, x):
        x = N.asmatrix(x)
        if x.shape[0] != self._A.shape[1]:
            raise RuntimeError('Make sure x is column of length n!')
        return self.f(x)
        
class LinearMapJac(LinearMap):
    def f(self,x):
        return self._A
        
class Vortex2D(object):
#    def __init__(self):
    def f(self,x):
        f0 = x[1]
        f1 = -x[0]
        return N.concatenate((f0,f1),axis=0)

    def __call__(self, x):
        x = N.asmatrix(x)
        if len(x) != 2:
            raise RuntimeError('Make sure x is 2x1 column vector!')
        return self.f(x)
        
class Vortex2DJac(Vortex2D):
    def __call__(self):
        ans = N.matlib.zeros((2,2))
        ans[1,0] = 1
        ans[0,1] = -1
        return ans