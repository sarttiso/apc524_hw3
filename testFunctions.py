#!/usr/bin/env python

#%%
import functions as F
import numpy as N
import unittest

class TestFunctions(unittest.TestCase):
    def testApproxJacobian1(self):
        slope = 3.0
        def f(x):
            return slope * x + 5.0
        x0 = 2.0
        dx = 1.e-3
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (1,1))
        self.assertAlmostEqual(Df_x, slope)

    def testApproxJacobian2(self):
        A = N.matrix("1. 2.; 3. 4.")
        def f(x):
            return A * x
        x0 = N.matrix("5; 6")
        dx = 1.e-6
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (2,2))
        N.testing.assert_array_almost_equal(Df_x, A)

    def testPolynomial(self):
        # p(x) = x^2 + 2x + 3
        x1 = 1
        x2 = 2
        x3 = 3
        p = F.Polynomial([x1, x2, x3])
        for x in N.linspace(-2,2,11):
            self.assertEqual(p(x), x1*x**2 + x2*x + x3)
            
    def testPolynomialJac(self):
        x1 = 1
        x2 = 2
        x3 = 3
        j = F.PolynomialJac([x1,x2,x3])
        for x in N.linspace(-2,2,11):
            self.assertEqual(j(x), 2*x1*x + x2)
            
    def testSinusoid(self):
        a = 1
        f = 3
        p = 0
        s = F.Sinusoid1D(a,f,p)
        for x in N.linspace(-2,2,11):
            self.assertEqual(s(x), a*N.sin(f*x + p))
            
    def testSinusoid1DJac(self):
        a = 1
        f = 3
        p = 0
        j = F.Sinusoid1DJac(a,f,p)
        for x in N.linspace(-2,2,11):
            self.assertEqual(j(x), f*a*N.sin(f*x + p + N.pi/2))
            
    def testLinearMap(self):
        A = N.matrix('1,1;1,1')
        l = F.LinearMap(A)
        x = N.matrix('1;1')
        N.testing.assert_array_almost_equal(l(x),N.matrix('2;2'))
        
    def testLinearMapJac(self):
        A = N.matrix('1,1;1,1')
        l = F.LinearMapJac(A)
        x = N.matrix('1;1')
        N.testing.assert_array_almost_equal(l(x),A)
        
    def testVortex2D(self):
        f = F.Vortex2D()
        x = N.matrix('1;1')
        N.testing.assert_array_equal(f(x),N.matrix('1;-1'))
        
    def testVortex2DJac(self):
        f = F.Vortex2DJac()
        x = N.matrix('1;1')
        N.testing.assert_array_equal(f(x),N.matrix('0,1;-1,0'))
            
    def testCompareJacobians(self):
        x = 1.
        # Linear case
        linearPoly = F.Polynomial([1.,1.])
        dfLinearApp = F.ApproximateJacobian(linearPoly,x)
        dfLinearAna = F.PolynomialJac([1.,1.])
        N.testing.assert_array_almost_equal(dfLinearAna(x),dfLinearApp)
        # Quadratic case
        f = F.Polynomial([1.,0.,0.])
        dfQuadraticAna = F.Polynomial([2.,0.])
        dfQuadraticApp = F.ApproximateJacobian(f,x)
        N.testing.assert_array_almost_equal(dfQuadraticAna(x),dfQuadraticApp)
        # Linear map
        x0 = N.matrix("1.;1.")
        A = N.matrix('1,-1;1,1')
        l = F.LinearMap(A)
        dlAna = F.LinearMapJac(A)
        dlApp = F.ApproximateJacobian(l,x0)
        N.testing.assert_array_almost_equal(dlAna(x0),dlApp)
    
            

if __name__ == '__main__':
    unittest.main()



