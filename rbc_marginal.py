import pandas as pd
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
import tensorflow as tf
import tensorflow_probability as tfp
from src.model import RBC
from src.generate_perturbation import solve_first_order
from src.generate_perturbation_derivative import solve_first_order_p 
tfd = tfp.distributions

class MarginalLikelihood:
    def __init__(self):
        self.m = RBC()
        self.m.compute_derivatives()

    # Define the function to initialize priors for the parameters
    def initialize_priors(self):
        # Priors for the parameters
        alpha_prior = tfd.TruncatedNormal(loc=0.3, scale=0.025, low=0.2, high=0.5)
        mean_gamma = 0.25
        std_gamma = 0.1
        beta_draw_prior = tfd.Gamma(concentration=(mean_gamma / std_gamma) ** 2, rate=mean_gamma / (std_gamma ** 2))
        mean_beta = 0.5
        std_beta = 0.2
        nu = mean_beta * (1 - mean_beta) / std_beta**2 - 1
        rho_prior = tfd.Beta(mean_beta * nu, (1-mean_beta) * nu)
        
        # No need to cast here since we're returning the distributions themselves
        return alpha_prior, beta_draw_prior, rho_prior  

    def kalman_filter(self, hx, gx, P0, data):
        num_timesteps = data.shape[0]
        # call the RBC class for sigma and eta
        Sigma = np.sqrt(float(self.m.Sigma.subs(self.m.numerical_params)))
        eta = np.array(self.m.eta, dtype=np.float64)
        
        # state transfer matrix
        transition_cov = tf.constant(eta*Sigma, dtype=tf.float64)  
        # obseravation matrix
        measurement_error_std = np.sqrt(self.m.e)
        observation_cov = tf.constant([measurement_error_std, measurement_error_std], dtype=tf.float64)  

        A = tf.constant([[1,0],[-1,1]], dtype=tf.float64)
        obs_matrix = tf.matmul(A, gx)
        
        # Initialize the state
        initial_loc = tf.constant([0, 0], dtype=tf.float64)
        # use Cholesky decomposition to get lower triangular matrix
        P0_chol = tf.linalg.cholesky(P0)
        initial_state_prior = tfd.MultivariateNormalTriL(
            loc=initial_loc,
            scale_tril=P0_chol)

        # construct linear gaussian state space model
        lgssm = tfd.LinearGaussianStateSpaceModel(
            num_timesteps=num_timesteps, 
            transition_matrix=tf.linalg.LinearOperatorFullMatrix(hx),
            transition_noise=tfd.MultivariateNormalDiag(scale_diag=transition_cov),
            observation_matrix=tf.linalg.LinearOperatorFullMatrix(obs_matrix),
            observation_noise=tfd.MultivariateNormalDiag(scale_diag=observation_cov),
            initial_state_prior=initial_state_prior
        )

        # compute log likelihood of observation

        log_likelihood = lgssm.log_prob(data)
        return log_likelihood

    # Function to compute the gradient of the log-likelihood
    @tf.custom_gradient
    def marginal_likelihood_with_grad(self, params, data):
        # transform tf tensor to numpy array since all computation in numpy in perturbation
        numpy_params = [p.numpy() if tf.is_tensor(p) else p for p in params]
        numpy_params = [p.item() if isinstance(p, np.ndarray) else p for p in numpy_params]
        self.m.set_parameter_values(*numpy_params)
        try:
            # Implement the state-space model, including perturbation methods and the Kalman filter
            gx, hx = solve_first_order(self.m)
            grad_gx, grad_hx = solve_first_order_p(self.m)
            # Find the initial covariance P0 for Kalman filter by solving the discrete Lyapunov equation
            Sigma = float(self.m.Sigma.subs(self.m.numerical_params))
            q = np.array([[0,0],[0,Sigma]])
            P0 = solve_discrete_lyapunov(hx, q)
            grad_P0 = np.zeros_like(grad_hx)
            for i in range(self.m.n_p):
                q_grad = grad_hx[i] @ P0 @ (hx.T) + hx @ P0 @ (grad_hx[i].T)
                grad_P0[i] = solve_discrete_lyapunov(hx, q_grad)
        except Exception as e:
            # Return a placeholder for log-likelihood and a dummy gradient function
            def grad(_):
                # Return arbitrary small negative gradients to avoid sampler getting stuck
                arbitrary_grads = [-1e-3 * tf.ones_like(p, dtype=tf.float64) for p in params]
                return arbitrary_grads, None
            # Return a large negative value for the log-likelihood to indicate failure
            return tf.constant(tf.float64.min), grad
        
        # Convert Numpy arrays to TensorFlow tensors
        gx = tf.convert_to_tensor(gx, dtype=tf.float64)
        hx = tf.convert_to_tensor(hx, dtype=tf.float64)
        P0 = tf.convert_to_tensor(P0, dtype=tf.float64)
        grad_gx = tf.convert_to_tensor(grad_gx, dtype=tf.float64)
        grad_hx = tf.convert_to_tensor(grad_hx, dtype=tf.float64)
        grad_P0 = tf.convert_to_tensor(grad_P0, dtype=tf.float64)


        with tf.GradientTape() as tape:
            for p in params:
                tape.watch(p)
            tape.watch([gx, hx, P0])
            # Calculate log_likelihood using Kalman filter
            log_likelihood = self.kalman_filter(hx, gx, P0, data)

        # Compute gradients of log_likelihood w.r.t. gx and hx
        grad_log_likelihood_gx, grad_log_likelihood_hx, grad_log_likelihood_P0 = tape.gradient(log_likelihood, [gx, hx, P0])
        
        def grad(dlog_likelihood):
            # Compute the gradients using chain rule
            grad_params_gx = tf.reduce_sum(grad_log_likelihood_gx[tf.newaxis, ...] * grad_gx, axis=[1, 2])
            grad_params_hx = tf.reduce_sum(grad_log_likelihood_hx[tf.newaxis, ...] * grad_hx, axis=[1, 2])
            grad_params_P0 = tf.reduce_sum(grad_log_likelihood_P0[tf.newaxis, ...] * grad_P0, axis=[1, 2])
           
            # Sum the gradients from gx and hx
            grad_params = (grad_params_gx + grad_params_hx + grad_params_P0)

            # chain rule for target function
            grad_params *= dlog_likelihood
            
            return tf.unstack(grad_params), None

        # Return log-likelihood and the gradient function
        return log_likelihood, grad
    
    # another function to compute the marginal likelihood which compute the gradient by central difference instead of chain rule
    @tf.custom_gradient
    def marginal_likelihood_with_numerical_grad(self, params, data):
        # Compute the log-likelihood using the model
        log_likelihood = self.marginal_likelihood(params, data)
        
        def grad_fn(dy):
            # dy is the upstream gradient
            epsilon = 1e-8
            grads = []

            for i in range(len(params)):
                # Increment the i-th parameter by epsilon
                params_eps_plus = params[:]
                params_eps_plus[i] = tf.add(params[i], epsilon)

                # Decrement the i-th parameter by epsilon
                params_eps_minus = params[:]
                params_eps_minus[i] = tf.subtract(params[i], epsilon)

                # Compute the log-likelihood at params_eps_plus
                log_likelihood_plus = self.marginal_likelihood(params_eps_plus, data)

                # Compute the log-likelihood at params_eps_minus
                log_likelihood_minus = self.marginal_likelihood(params_eps_minus, data)

                # Compute the numerical gradient
                grad_i = tf.subtract(log_likelihood_plus, log_likelihood_minus) / (2.0 * epsilon)

                # Multiply by the upstream gradient dy
                grad_i = tf.multiply(grad_i, dy)
                
                grads.append(grad_i)

            # Return a list of gradients for the parameters
            return grads, None

        # Return the value and the custom gradient function
        return log_likelihood, grad_fn
    
    # function to compute the marginal likelihood without gradient for RWMH
    def marginal_likelihood(self, params, data):
        # transform tf tensor to numpy array since all computation in numpy in perturbation
        numpy_params = [p.numpy() if tf.is_tensor(p) else p for p in params]
        numpy_params = [p.item() if isinstance(p, np.ndarray) else p for p in numpy_params]
        self.m.set_parameter_values(*numpy_params)
        # try to compute the perturbation solution, for those param leads to a failed computation, return -inf loglikelihood
        try:
            gx, hx = solve_first_order(self.m)
            Sigma = float(self.m.Sigma.subs(self.m.numerical_params))
            q = np.array([[0,0],[0,Sigma]])
            P0 = solve_discrete_lyapunov(hx, q)
            
        except Exception as e:
            # Return a large negative value for the log-likelihood to indicate failure
            return tf.constant(tf.float64.min)
        
        # Convert Numpy arrays to TensorFlow tensors
        gx = tf.convert_to_tensor(gx, dtype=tf.float64)
        hx = tf.convert_to_tensor(hx, dtype=tf.float64)
        P0 = tf.convert_to_tensor(P0, dtype=tf.float64)

        # Calculate log_likelihood using Kalman filter
        log_likelihood = self.kalman_filter(hx, gx, P0, data)
        return log_likelihood


    # Define the function to run NUTS sampling
    def run_nuts(self, data, num_chains=1, num_results=6500, num_burnin_steps=650):

        alpha_prior, beta_draw_prior, rho_prior = self.initialize_priors()
        # Sample initial values for the parameters for each chain
        initial_states = [
        tf.constant(0.4, dtype=tf.float64, shape=[num_chains]),
        tf.constant(0.4, dtype=tf.float64, shape=[num_chains]),
        tf.constant(0.8, dtype=tf.float64, shape=[num_chains])]

        # Define the target log-probability function
        def target_log_prob_fn(alpha, beta_draw, rho):
            # Ensure that all computations are done in float32 for prior function requirement
            alpha = tf.cast(alpha, dtype=tf.float32)
            beta_draw = tf.cast(beta_draw, dtype=tf.float32)
            rho = tf.cast(rho, dtype=tf.float32)
            # Convert βdraw to β
            beta = 1 / (1 + beta_draw / 100)

            # Compute the log prior probabilities for alpha and rho directly
            log_prior_alpha = tf.cast(alpha_prior.log_prob(alpha), tf.float64)
            log_prior_rho = tf.cast(rho_prior.log_prob(rho), tf.float64)
            # Compute the log prior for beta_draw through its transformation
            log_prior_beta_draw = tf.cast(beta_draw_prior.log_prob(beta_draw), tf.float64)

            # Compute the log prior probabilities
            log_prior_prob = log_prior_alpha + log_prior_beta_draw + log_prior_rho

            # Ensure that all computations are done in float64 for gradient computation
            alpha = tf.cast(alpha, dtype=tf.float64)
            beta = tf.cast(beta, dtype=tf.float64)
            rho = tf.cast(rho, dtype=tf.float64)
            # Function to compute log likelihood for each chain
            def compute_log_likelihood(chain_states):
                alpha, beta, rho = chain_states
                return self.marginal_likelihood_with_numerical_grad([alpha, beta, rho], data)

            # Compute the log likelihood for each chain
            log_likelihoods = tf.map_fn(
                compute_log_likelihood,
                elems=(alpha, beta, rho),
                dtype=tf.float64
            )

            # Return the total log posterior (log prior + log likelihood)
            return (log_prior_prob + log_likelihoods) 

        # Initialize the NUTS sampler
        nuts = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=target_log_prob_fn,
            step_size=1e-5,
            max_tree_depth=10
        )

        # Adaptive NUTS sampler
        adaptive_nuts = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=nuts,
            num_adaptation_steps=int(num_burnin_steps * 0.8),
            target_accept_prob=0.65
        )

        # Run the chains (with burn-in)
        states, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_states,
            kernel=adaptive_nuts,
            trace_fn=lambda _, pkr: {
                'is_accepted': pkr.inner_results.is_accepted,
                'step_size': pkr.inner_results.step_size,
                'leapfrogs_taken': pkr.inner_results.leapfrogs_taken
            },
            parallel_iterations=num_chains
        )

        return states, kernel_results


    # Define the function to run RWMH sampling with multiple chains
    def run_rwmh(self, data, num_chains=1, num_results=99000, num_burnin_steps=11000):

        alpha_prior, beta_draw_prior, rho_prior = self.initialize_priors()
        # Sample initial values for the parameters for each chain
        initial_states = [
        tf.constant(0.4, dtype=tf.float64, shape=[num_chains]),
        tf.constant(0.4, dtype=tf.float64, shape=[num_chains]),
        tf.constant(0.8, dtype=tf.float64, shape=[num_chains])]
        
        # Define the target log-probability function
        def target_log_prob_fn(alpha, beta_draw, rho):
            # Ensure that all computations are done in float32 for prior function requirement
            alpha = tf.cast(alpha, dtype=tf.float32)
            beta_draw = tf.cast(beta_draw, dtype=tf.float32)
            rho = tf.cast(rho, dtype=tf.float32)

            # Convert βdraw to β
            beta = 1 / (1 + beta_draw / 100)

            # Compute the log prior probabilities for alpha and rho directly
            log_prior_alpha = tf.cast(alpha_prior.log_prob(alpha), tf.float64)
            log_prior_rho = tf.cast(rho_prior.log_prob(rho), tf.float64)
            # Compute the log prior for beta_draw through its transformation
            log_prior_beta_draw = tf.cast(beta_draw_prior.log_prob(beta_draw), tf.float64)

            # Compute the log prior probabilities
            log_prior_prob = log_prior_alpha + log_prior_beta_draw + log_prior_rho

            # Ensure that all computations are done in float64 for gradient computation
            alpha = tf.cast(alpha, dtype=tf.float64)
            beta = tf.cast(beta, dtype=tf.float64)
            rho = tf.cast(rho, dtype=tf.float64)
            # Function to compute log likelihood for each chain
            def compute_log_likelihood(chain_states):
                alpha, beta, rho = chain_states
                # If the parameters have no economic meaning, return 0 probability
                if not (0.2 < alpha < 0.5) or not (0 < beta < 1) or not (0 < rho < 1):
                    return -tf.float64.max              
                return self.marginal_likelihood([alpha, beta, rho], data)

            # Compute the log likelihood for each chain
            log_likelihoods = tf.map_fn(
                compute_log_likelihood,
                elems=(alpha, beta, rho),
                dtype=tf.float64
            )

            # Return the total log posterior (log prior + log likelihood)
            return log_prior_prob + log_likelihoods


        # Initialize the RWMH sampler
        rwmh = tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=target_log_prob_fn
        )
        # Run the chains (with burn-in)
        states, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_states,
            kernel=rwmh,
            trace_fn=lambda _, pkr: pkr.is_accepted,
            parallel_iterations=num_chains
        )

        return states, kernel_results
    

def run_rwmh(data, num_chains, num_results, num_burnin_steps):
    rbc_marginal_instance = MarginalLikelihood()  
    return rbc_marginal_instance.run_rwmh(data, num_chains, num_results, num_burnin_steps)

def run_nuts(data, num_chains, num_results, num_burnin_steps):
    rbc_marginal_instance = MarginalLikelihood()  
    return rbc_marginal_instance.run_nuts(data, num_chains, num_results, num_burnin_steps)

