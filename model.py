from sympy import symbols, exp, Function, S, Matrix, Array, diff

class RBC:
    def __init__(self):
        # Parameters in symbol
        self.alpha, self.beta, self.rho, self.delta, self.sigma = symbols('alpha beta rho delta sigma')
        
        # parameters needs to estimate
        self.theta = Matrix([self.alpha, self.beta, self.rho])
        
        # Define symbols for state and control variables as functions of time
        k, z, c, q = symbols('k z c q', cls=Function)
        self.k, self.z, self.c, self.q = k, z, c, q
        t = symbols('t')
        self.t = t
        self.k_t = self.k(t)
        self.z_t = self.z(t)
        self.c_t = self.c(t)
        self.q_t = self.q(t)

        # State-space representation: x is state variable, y is control variable
        # State transfer:  xp = h(x) + eta @ epsilon, where epsilon~N(0,Sigma)
        # Policy function: yp = g(x), where x_hat = x - x_bar
        self.x = Matrix([self.k_t, self.z_t])
        self.y = Matrix([self.c_t, self.q_t])
        self.xp = Matrix([self.k(self.t + 1), self.z(self.t + 1)])
        self.yp = Matrix([self.c(self.t + 1), self.q(self.t + 1)])
        # Loadings of the exo-shocks eta and VCV matrix Sigma for epsilon
        self.eta = [0, 1]
        self.Sigma = self.sigma**2

        # length of state and control variables
        self.n_x, self.n_y, self.n_p = len(self.x), len(self.y), len(self.theta)
        # Define the steady state variable for x and y
        steady_state_expressions = self.steady_state()
        self.x_bar = Matrix([steady_state_expressions[self.k_t], steady_state_expressions[self.z_t]])
        self.y_bar = Matrix([steady_state_expressions[self.c_t], steady_state_expressions[self.q_t]])

        self.e = 1e-5 # measurement_error_variance

        # Define the system of model equations in operator H
        self.H = Matrix([
            1 / self.c_t - self.beta * (self.alpha * exp(self.z_t.subs({t: t + 1})) * self.k_t.subs({t: t + 1})**(self.alpha - 1) + 1 - self.delta) / self.c_t.subs({t: t + 1}),
            self.c_t + self.k_t.subs({t: t + 1}) - (1 - self.delta) * self.k_t - self.q_t,
            self.q_t - exp(self.z_t) * self.k_t**self.alpha,
            self.z_t.subs({t: t + 1}) - self.rho * self.z_t
        ])

        # variable to indicate whether derivatives have been computed
        self.derivatives_computed = False

    def set_parameter_values(self, alpha_value, beta_value, rho_value, delta_value=0.025, sigma_value=0.01):
        # This method allows you to set numerical values for your parameters
    
        self.numerical_params = {
            self.alpha: alpha_value,
            self.beta: beta_value,
            self.rho: rho_value,
            self.delta: delta_value,
            self.sigma: sigma_value
        }
        # Replace symbols in x_bar and y_bar with their numerical values
        self.x_bar_num0 = self.x_bar.subs(self.numerical_params)
        self.y_bar_num0 = self.y_bar.subs(self.numerical_params)

        # Replace S(0) with 0.0 in the numerical matrices
        self.x_bar_num = self.x_bar_num0.applyfunc(lambda expr: 0.0 if expr == S.Zero else float(expr))

        self.y_bar_num = self.y_bar_num0.applyfunc(lambda expr: 0.0 if expr == S.Zero else float(expr))

        self.numerical_steady_state = {
            self.k(self.t): self.x_bar_num[0],
            self.z(self.t): self.x_bar_num[1],
            self.c(self.t): self.y_bar_num[0],
            self.q(self.t): self.y_bar_num[1],
            self.k(self.t+1): self.x_bar_num[0],
            self.z(self.t+1): self.x_bar_num[1],
            self.c(self.t+1): self.y_bar_num[0],
            self.q(self.t+1): self.y_bar_num[1]
        }

    def steady_state(self):

        # Solve steady state
        k_bar = ((1/self.beta - 1 + self.delta)/(self.alpha)) ** (1/(self.alpha - 1))
        z_bar = S.Zero
        c_bar = k_bar**self.alpha - self.delta * k_bar
        q_bar = k_bar**self.alpha
        
        return {self.k_t: k_bar, self.z_t: z_bar, self.c_t: c_bar, self.q_t: q_bar}
    
    def compute_derivatives(self):
        if not self.derivatives_computed:
            # Compute the derivatives of the steady states with respect to the parameters
            self.dx_bar_dtheta = self.x_bar.jacobian(self.theta)
            self.dy_bar_dtheta = self.y_bar.jacobian(self.theta)
        
            # Compute the Jacobian matrices
            self.H_x = self.H.jacobian(self.x)
            self.H_y = self.H.jacobian(self.y)
            self.H_xp = self.H.jacobian(self.xp)
            self.H_yp = self.H.jacobian(self.yp)
        
            # Compute the Hessian matrices by taking the Jacobian of the Jacobian matrices with respect to the state and control variables
            self.H_x_x = self.compute_hessian_tensor(self.H_x, self.x)
            self.H_x_xp = self.compute_hessian_tensor(self.H_x, self.xp)
            self.H_x_y = self.compute_hessian_tensor(self.H_x, self.y)
            self.H_x_yp = self.compute_hessian_tensor(self.H_x, self.yp)

            self.H_xp_x = self.compute_hessian_tensor(self.H_xp, self.x)
            self.H_xp_xp = self.compute_hessian_tensor(self.H_xp, self.xp)
            self.H_xp_y = self.compute_hessian_tensor(self.H_xp, self.y)
            self.H_xp_yp = self.compute_hessian_tensor(self.H_xp, self.yp)

            self.H_y_x = self.compute_hessian_tensor(self.H_y, self.x)
            self.H_y_xp = self.compute_hessian_tensor(self.H_y, self.xp)
            self.H_y_y = self.compute_hessian_tensor(self.H_y, self.y)
            self.H_y_yp = self.compute_hessian_tensor(self.H_y, self.yp)

            self.H_yp_x = self.compute_hessian_tensor(self.H_yp, self.x)
            self.H_yp_xp = self.compute_hessian_tensor(self.H_yp, self.xp)
            self.H_yp_y = self.compute_hessian_tensor(self.H_yp, self.y)
            self.H_yp_yp = self.compute_hessian_tensor(self.H_yp, self.yp)

            # Compute the partial derivative of the Jacobian with respect to parameters
            self.pH_x_ptheta = self.compute_hessian_tensor(self.H_x, self.theta)
            self.pH_y_ptheta = self.compute_hessian_tensor(self.H_y, self.theta)
            self.pH_xp_ptheta = self.compute_hessian_tensor(self.H_xp, self.theta)
            self.pH_yp_ptheta = self.compute_hessian_tensor(self.H_yp, self.theta)
            self.derivatives_computed = True
    
    def compute_hessian_tensor(self, matrix, vars):
        # This method is used to elementwisely compute the Hessian matrix 
        n_rows, n_cols = matrix.shape
        n_var = len(vars)
    
        # Compute the Hessian tensor as a nested list comprehension
        hessian_tensor = [[[diff(matrix[i, j], vars[k]) for k in range(n_var)]
                            for j in range(n_cols)]
                            for i in range(n_rows)]
        # Convert the nested list to a sympy Array
        hessian_tensor_array = Array(hessian_tensor)
    
        return hessian_tensor_array




