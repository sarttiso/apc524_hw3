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
#        
    def testLinearAnalyticDf(self):
        f = F.Polynomial([3,6])
        df = lambda x : 3.0
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
#        
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
        
    def testBivariateQuadratic(self):
        x0 = N.matrix("1.;2.")
        def f(x):
            f0 = x[0]*x[0]
            f1 = x[1]*x[1]
            return N.concatenate((f0,f1),axis=0)
        solver = newton.Newton(f, tol=1.e-15, maxiter=60)
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, N.matlib.zeros((2,1)))
        
#    def testBivariateQuadraticAnalyticDf(self):
#        x0 = N.matrix("1.;2.")
#        def f(x):
#            f0 = x[0]*x[0]
#            f1 = x[1]*x[1]
#            return N.concatenate((f0,f1),axis=0)
#        def df(x):
#            return N.matrix([[2.0*x[0,0],0],[0,2.0*x[1,0]]])
#        solver = newton.Newton(f, tol=1.e-10, maxiter=19, Df=df)
#        x = solver.solve(x0)
#        N.testing.assert_array_almost_equal(x, N.matlib.zeros((2,1)))
#    
    def testRadius(self):
        f = F.Polynomial([1,1,0,0])
        x0 = 5.0
        solver = newton.Newton(f, tol=1.e-10, maxiter=30, dx=1.e-5, r=5.)
        x = solver.solve(x0)
        self.assertAlmostEqual(x, 0.0, 4)

if __name__ == "__main__":
    unittest.main()
