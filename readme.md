functions.py
  Each of the function classes in this file has a corresponding derived class that implements its Jacobian. I have provided functions in addition to those given, namely three classes:
  PolynomialJac: a derived class of the polynomial Jacobian, which returns the one-dimensional derivative corresponding to the same set of coeffs given to instantiate a polynomial.
  Sinusoid1D & Sinusoid1DJac: classes the define sinusoids in 1-dimension with given amplitudes, frequencies, and phases. The jacobian class returns objects with the appropriate derivatives given the same input to instantiate a Sinusoid.
  LinearMap & LinearMapJac: classes that define linear maps by a given transformation matrix A. Maps Rn -> Rn.
  Vortex2D & Vortex2DJac: derived classes of LinearMap that generate a clockwise rotational vector field in 2 dimensions.

newton.py
  Class that implements general dimensional Newton root search algorithm. Implements gradient search within a given optional radius r of a beginning point ( .solve(x0) ) on a given function f using either a numerical approximation of the Jacobian of that function or a given analytic expression Df. The numerical approximation of the Jacobian also takes a parameter dx that defines the fineness of the derivative approximation for each component of the Jacboian. The search terminates in maxiter iterations if a root is not found.
  Member functions are solve and step. Calling solve(x0) at query point x0 begins the gradient descent at that point, and each iteration calls the step function which computes the Jacobian and updates the location x for the next step. When the functional value at a given x is within a the specified tolerance tol of zero (measured by the Euclidean norm), the search terminates and returns that location.
  All data are expected to be matrices as implemented by numpy.

testFunctions.py
  Each function type, both the function itself and its Jacobian, has a test in this file. There is also the given test of the numerical approximation of the Jacobians, and I also include a test that compares analytic and numerical Jacobians to ensure accuracy.

testNewton.py
  Each function type is tested with the Newton descent method for root-finding, both for the case where the Jacobian is approximated numerically and also for the case where the Jacobian is given analytically.
  I also test that the analytic Jacobian is used when provided (testUsageOfAnalyticDf)
  I also test that the radius criterion is abided by when a search radius is specified.
