import numpy as np

class regressor:
    """
    Regressor Class that compute the best fit parameters given a function, 
    x, and y. 
    The class use the Levenberg-Marquardt algorthm to solve the task.
    source: 
      - https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
    """
    def __init__(self, llambda = 1e-3, lambda_multiplier=10,
                 delta_factor = 1e-4, min_delta = 1e-4):
        """
        __init__ of regressor class. 

        Parameters
        ----------
        llambda : float
            Parameters of the algorithm of Levenberg Marquardt. Default: 1e-4.
        lambda_multiplier : float
            Multiplier for damping the llambda factor. Default: 10.
        delta_factor : float
            Perturbation respect to a parameter of fit to compute the Jacobian.
        min_delta : float
            Minimum value of delta for update the Jacobian.
        """
        self.delta_factor = delta_factor
        self.min_delta = min_delta
        self.llambda = llambda
        self.lambda_multiplier = lambda_multiplier
        self._rmse = None
        self._best_params = None
        self._reason = None


    @property
    def best_params(self):
        """Get the results of LM algorithm"""
        return self._best_params
    @property
    def rmse(self):
        """Get the root mean square error of LM algorithm"""
        return self._rmse
    @property
    def reason(self):
        """Get the reason of stop iteration of the LM algorithm"""
        return self._reason

    def compute_J(self, x, func, params):
        """
        Compute the Jacobian respect to the parameters of a given function.

        Parameters
        ----------
        x : numpy nd array
            Free variable of the function func.
        func : function object
            Function of which we want to calculate the Jacobian.
        params : list
            List of parameters of func exept x.

        Return
        ------
        J : numpy nd array
            The Jacobian of the function evaluated in x.
        """
        y0 = func(x, *params) # Initial value of func
        J = np.empty(shape=(len(params), *y0.shape)) # empty jacobian
        for i, param in enumerate(params): # for each parameter
            params_star = params[:] # save a new set of parameters
            delta = param * self.delta_factor # perturb one parameter
            if abs(delta) < self.min_delta: # set a minimum of perturbation
                delta = self.min_delta
            params_star[i] += delta # update the parameter
            y1 = func(x, *params_star) # Compute the y perturbated
            derivative = (y1-y0)/delta # compute the derivative
            J[i] = derivative # Insert the derivative in the Jacobian
        return J

    def fit(self, x, y, func, params, Jacobian = None, kmax = 500, eps = 1e-3, 
            method = 'lm'):
        """
        Implementation of the Levenberg-Marquardt algorithm.

        Parameters
        ----------
        x : numpy nd array
            Free variable of the function func.
        y : numpy nd array
            Values to fit.
        func : function object
            Function of which we want to calculate the Jacobian.
        params : list
            List of parameters of func exept x.
        Jacobian : function object
            Jacobian of the function func, if None the Jacobian is computed
            in place. Default: None.
        kmax : int
            Maximum number of iteration of the algorithm.
        eps = 1e-3
        """
        # Equation to solve (for delta):
        # (JtJ + lambda * I * diag(JtJ)) * delta = Jt * error
        # equivalent: MAT * delta = b

        if method == 'lm': # if the method is the Levenberg-Marquardt
            # If lazy user don't compute the Jacobian just use numerical 
            # derivatives
            if Jacobian == None: Jacobian = self.compute_J
    
            def fix_results(rmse, params, reason): # Simplify the results writing
                self._rmse = rmse
                self._best_params = params
                self._reason = reason
    
            # Function that compute the error: y - f(x)
            err_func = lambda x, y, func, params:  y - func(x,*params)
            llambda = self.llambda # local lambda (will be modified)
            rmse = None
    
            k = 0
            while k < kmax: 
                k+=1
                J = Jacobian(x, func, params) # Compute the Jacobian
                JtJ = np.inner(J, J) # dot product is just np.inner
                A = np.eye(len(params))*np.diag(JtJ) # diag(JtJ) * I
                error = err_func(x, y, func, params) # Compute the error
                Jerror = np.inner(J, error) # Jt * error
                rmse = np.linalg.norm(error) # The rmse is just the norm of error
                
                if rmse < eps: # If rmse is low enough stop the algorithm 
                    reason = "Converged to min epsilon"
                    fix_results(rmse, params, reason)
                    break
                
                reason = "" 
                error_star = error[:] # Initialize the perturbated error
                rmse_star = rmse + 1 # Initialize the rmse_star (arbitrary high)
                
                # This loop is for dumping the lambda value to reach a faster and 
                # stable convergence
                while rmse_star > rmse: # While the rmse don't improve
                    try: # Compute the delta solving the system 
                        delta = np.linalg.solve(JtJ + llambda * A, Jerror)
                    except np.linalg.LinAlgError: # if system cannot be solved
                        print("Error: Singular Matrix")
                        break
                
                    # Update params and calculate new error
                    params_star = params[:] + delta[:]
                    error_star = err_func(x, y, func, params_star)
                    rmse_star = np.linalg.norm(error_star)
        
                    # Update params if rmse is improved.
                    if rmse_star < rmse:
                        params = params_star
                        # Rescale the lambda value to the previous iteration
                        # because it was the good one!
                        llambda /= self.lambda_multiplier
                        break
    
                    # if the rmse is not improved try to increase lambda
                    # This help fast convergence
                    llambda *= self.lambda_multiplier
        
                    if llambda > 1e9: # return if lambda explodes
                        reason = "Lambda to large."
                        fix_results(rmse, params, reason)
                        break
    
                if abs(rmse - rmse_star) < 1e-18: # return if diff(rmse) too small
                    reason = "Change in error too small"
                    fix_results(rmse, params, reason)
                    break
            fix_results(rmse, params, "kmax iteration reached") # Return for max iteration
        else: raise Exception(f'Unknown method {method}')
