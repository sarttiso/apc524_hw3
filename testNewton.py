#!/usr/bin/env python

import newton
import unittest
import functions as F
import numpy as N

class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = F.Polynomial([3,6])
        solver = newton.Newton(f, tol=1.e-10, maxiter=4, dx=1.e-5)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)
        
    def testLinearAnalyticDf(self):
        f = F.Polynomial([3,6])
        df = F.Polynomial([3])
        solver = newton.Newton(f, tol=1.e-10, maxiter=4, dx=1.e-5, Df=df)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)
        
    def testQuadratic(self):
        f = F.Polynomial([2,0,0])
        solver = newton.Newton(f, tol=1.e-10, maxiter=30, dx=1.e-5)
        x = solver.solve(5.0)
        self.assertAlmostEqual(x, 0.0,4)
        
    def testQuadraticAnalyticDf(self):
        f = F.Polynomial([2,0,0])
        df = F.Polynomial([0,2,0])
        solver = newton.Newton(f, tol=1.e-10, maxiter=30, Df=df)
        x = solver.solve(5.0)
        self.assertAlmostEqual(x, 0.0,4)
        
    def testSinusoid1D(self):
        f = F.Sinusoid1D()
        solver = newton.Newton(f, tol=1.e-10, maxiter=20, dx=1.e-5)
        x = solver.solve(1.0)
        self.assertAlmostEqual(x, 0.0,4)
        
    def testSinusoid1DAnalyticDf(self):
        f = F.Sinusoid1D()
        df = F.Sinusoid1DJac()
        solver = newton.Newton(f, tol=1.e-10, maxiter=20, dx=1.e-5, Df=df)
        x = solver.solve(1.0)
        self.assertAlmostEqual(x, 0.0,4)
        
    def testLinearMap(self):
        A = N.matrix("1. 2.; 3. 4.")
        x0 = N.matrix("1.;2.")
        f = F.LinearMap(A)
        solver = newton.Newton(f, tol=1.e-10, maxiter=3)
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, N.matlib.zeros((2,1)))
        
    def testVortex2D(self):
        x0 = N.matrix('1;1')
        f = F.Vortex2D()
        solver = newton.Newton(f,tol=1.e-10, maxiter=50)
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x,N.matlib.zeros((2,1)))
        
    def testVortex2DAnalyticDf(self):
        x0 = N.matrix('1;1')
        f = F.Vortex2D()
        df = F.Vortex2DJac()
        solver = newton.Newton(f,tol=1.e-10, maxiter=50, Df=df)
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x,N.matlib.zeros((2,1)))
        
    def testUsageOfAnalyticDF(self):
        f = F.Polynomial([1.,0.,0.])
        # incorrect jacobian
        df = F.Polynomial([0.])
        solver = newton.Newton(f,tol=1.e-10, maxiter=50, Df=df)
        try: x = solver.solve(1.0)
        except RuntimeError:
            pass

    def testRadiusLinear(self):
        f = F.Polynomial([1,0])
        # outside of search radius of root
        x0 = 5.0
        solver = newton.Newton(f, tol=1.e-10, maxiter=30, dx=1.e-5, r=4.)
        try: x = solver.solve(x0)
        except RuntimeError:
            pass
        
    def testRadiusQuadratic(self):
        f = F.Polynomial([1,0,0])
        # outside of search radius of root
        x0 = 5.0
        solver = newton.Newton(f, tol=1.e-10, maxiter=30, dx=1.e-5, r=4.)
        try: x = solver.solve(x0)
        except RuntimeError:
            pass

if __name__ == "__main__":
    unittest.main()
