import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from src.model import RBC
from src.generate_perturbation import solve_first_order
from src.generate_perturbation_derivative import solve_first_order_p 

tfd = tfp.distributions


class JointLikelihood:
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
    
    # Function to compute the gradient of the log-likelihood
    @tf.custom_gradient
    def joint_likelihood_with_gradient(self, params, z):
        # transform tf tensor to numpy array since all computation in numpy in perturbation
        numpy_params = [p.numpy() if tf.is_tensor(p) else p for p in params[:3]]
        eps = tf.stack(params[3:])
        self.m.set_parameter_values(*numpy_params)

        # try to compute the perturbation solution, for those param leads to a failed computation, return -inf loglikelihood
        try:
            gx, hx = solve_first_order(self.m)
            grad_gx, grad_hx = solve_first_order_p(self.m)
        except Exception as e:
            # Return a placeholder for log-likelihood and a dummy gradient function
            def grad(_):
                # Return arbitrary small negative gradients to avoid sampler getting stuck
                arbitrary_grads = [-1e-2 * tf.ones_like(p, dtype=tf.float64) for p in params]
                return arbitrary_grads, None
            # Return a large negative value for the log-likelihood to indicate failure
            return tf.constant(tf.float64.min), grad
        
        # Convert Numpy arrays to TensorFlow tensors
        gx = tf.convert_to_tensor(gx, dtype=tf.float64)
        hx = tf.convert_to_tensor(hx, dtype=tf.float64)
        grad_gx = tf.convert_to_tensor(grad_gx, dtype=tf.float64)
        grad_hx = tf.convert_to_tensor(grad_hx, dtype=tf.float64)

        # state transfer variance
        Sigma = np.sqrt(float(self.m.Sigma.subs(self.m.numerical_params)))
        eta = tf.constant(np.array(self.m.eta, dtype=np.float64))
        transition_cov = tf.constant(Sigma, dtype=tf.float64) # eta @ Sigma @ eta'
        # obseravation variance
        measurement_error_variance = np.sqrt(self.m.e)
        observation_cov = tf.constant([measurement_error_variance, measurement_error_variance], dtype=tf.float64)  # Omega
        A = tf.constant([[1,0],[-1,1]], dtype=tf.float64)
        
        # initialize the x0 = 0
        x = tf.constant([[0], [0]], dtype=tf.float64)
        # define the two pdf for z and e: N(0, Omega), N(0, eta*Sigma)
        z_pdf = tfd.MultivariateNormalDiag(loc=[0., 0.], scale_diag=observation_cov)
        e_pdf = tfd.Normal(loc=0., scale=transition_cov)
        

        with tf.GradientTape() as tape:
            for p in params:
                tape.watch(p)
            tape.watch(eps)
            tape.watch([gx, hx])
            # define the observation matrix
            obs_matrix = tf.matmul(A,gx)
            # initialize the log_likelihood
            log_likelihood = tf.constant(0, dtype=tf.float64)
            # compute the log-posterior over all time
            for t in range(len(z)):
                x_next = tf.matmul(hx, x) + tf.reshape(eta * eps[t], [2,1])
                # add the log_likelihood at this step
                log_likelihood += z_pdf.log_prob(z[t]-tf.squeeze(tf.matmul(obs_matrix,x_next)))
                log_likelihood += e_pdf.log_prob(eps[t])
                # update the state x
                x = x_next

        # Compute gradients of log_likelihood w.r.t. gx, hx
        grad_log_likelihood_gx, grad_log_likelihood_hx, grad_eps0 = tape.gradient(log_likelihood, [gx, hx, eps])
            

        def grad(dlog_likelihood):
            # Compute the gradients using chain rule
            grad_params_gx = tf.reduce_sum(grad_log_likelihood_gx[tf.newaxis, ...] * grad_gx, axis=[1, 2])
            grad_params_hx = tf.reduce_sum(grad_log_likelihood_hx[tf.newaxis, ...] * grad_hx, axis=[1, 2])
           
            # Sum the gradients from gx and hx
            grad_params = grad_params_gx + grad_params_hx
            # chain rule for target function
            grad_params *= dlog_likelihood

            # Chain rule for eps
            grad_eps = grad_eps0 * dlog_likelihood
            
            return tf.unstack(grad_params) + tf.unstack(grad_eps), None

        # Return log-likelihood and the gradient function
        return log_likelihood, grad

    # define joint likelihood fucntion to compute ln p(z^T|e^T,theta) + ln p(x^T|theta)
    def joint_likelihood(self, params, eps, z):
        # transform tf tensor to numpy array since all computation in numpy in perturbation
        numpy_params = [p.numpy() if tf.is_tensor(p) else p for p in params]
        self.m.set_parameter_values(*numpy_params)

        # try to compute the perturbation solution, for those param leads to a failed computation, return -inf loglikelihood
        try:
            gx, hx = solve_first_order(self.m)
        except Exception as e:
            # Return a large negative value for the log-likelihood to indicate failure
            return tf.constant(tf.float64.min)
        
        # state transfer variance
        Sigma = np.sqrt(float(self.m.Sigma.subs(self.m.numerical_params)))
        eta = tf.constant(np.array(self.m.eta, dtype=np.float64))
        transition_std = tf.constant(Sigma, dtype=tf.float64) # eta @ Sigma @ eta'
        # obseravation variance
        measurement_error_std = np.sqrt(self.m.e)

        observation_cov = tf.constant([measurement_error_std, measurement_error_std], dtype=tf.float64)  # Omega

        A = tf.constant([[1,0],[-1,1]], dtype=tf.float64)
        obs_matrix = tf.matmul(A,gx)
        # initialize the x0 = 0
        x = tf.constant([[0], [0]], dtype=tf.float64)
        # define the two pdf for z and e: N(0, Omega), N(0, eta*Sigma)
        z_pdf = tfd.MultivariateNormalDiag(loc=[0., 0.], scale_diag=observation_cov)
        e_pdf = tfd.Normal(loc=0., scale=transition_std)
        # initialize the log_likelihood
        log_likelihood = tf.constant(0, dtype=tf.float64)
  
        # compute the log-posterior over all time
        for t in range(len(z)):
            x_next = tf.matmul(hx, x) + tf.reshape(eta * eps[t], [2,1])
            # add the log_likelihood at this step
            log_likelihood += z_pdf.log_prob(z[t]-tf.squeeze(tf.matmul(obs_matrix,x_next)))
            log_likelihood += e_pdf.log_prob(eps[t])
            # update the state x
            x = x_next  
        
        return log_likelihood


    # Define the function to run NUTS sampling
    def run_nuts(self, eps0, z, num_results=6500, num_burnin_steps=650):

        alpha_prior, beta_draw_prior, rho_prior = self.initialize_priors()
        # Sample initial values for the parameters for each chain
        initial_states = [
            tf.constant(0.4, dtype=tf.float64),  # alpha
            tf.constant(0.4, dtype=tf.float64),  # beta_draw
            tf.constant(0.8, dtype=tf.float64),  # rho
        ] + tf.unstack(eps0)  # add eps0 elementwisely to the initial state list

         # Define the target log-probability function
        def target_log_prob_fn(*state_vector):
            # Ensure that all computations are done in float32 for prior function requirement
            alpha = tf.cast(state_vector[0], dtype=tf.float32)
            beta_draw = tf.cast(state_vector[1], dtype=tf.float32)
            rho = tf.cast(state_vector[2], dtype=tf.float32)

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

            # Compute the log likelihood
            log_likelihood = self.joint_likelihood_with_gradient([alpha,beta,rho]+list(state_vector[3:]), z) 

            # Return the total log posterior (log prior + log likelihood)
            return log_prior_prob + log_likelihood
         

        # Initialize the NUTS sampler
        nuts = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=target_log_prob_fn,
            step_size=1e-8,
            max_tree_depth=5
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
        )

        return states, kernel_results


    # Define the function to run RWMH sampling with multiple chains
    def run_rwmh(self, eps0, z, num_results=99000, num_burnin_steps=11000):

        alpha_prior, beta_draw_prior, rho_prior = self.initialize_priors()
        # Sample initial values for the parameters for each chain
        initial_states = [
            tf.constant(0.4, dtype=tf.float64),  # alpha
            tf.constant(0.4, dtype=tf.float64),  # beta_draw
            tf.constant(0.8, dtype=tf.float64),  # rho
        ] + tf.unstack(eps0)  # add eps0 elementwisely to the initial state list
        
        # Define the target log-probability function
        def target_log_prob_fn(*state_vector):
            # Ensure that all computations are done in float32 for prior function requirement
            alpha = tf.cast(state_vector[0], dtype=tf.float32)
            beta_draw = tf.cast(state_vector[1], dtype=tf.float32)
            rho = tf.cast(state_vector[2], dtype=tf.float32)
            eps = tf.stack(state_vector[3:])

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

            # If the parameters have no economic meaning, return 0 probability
            if not (0.2 < alpha < 0.5) or not (0 < beta < 1) or not (0 < rho < 1):
                return -tf.float64.max
            # Compute the log likelihood
            log_likelihood = self.joint_likelihood([alpha, beta, rho], eps, z) 

            # Return the total log posterior (log prior + log likelihood)
            return log_prior_prob + log_likelihood


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
        )

        return states, kernel_results


def run_rwmh(data, num_chains, num_results, num_burnin_steps):
    rbc_joint_instance = JointLikelihood()  
    return rbc_joint_instance.run_rwmh(data, num_chains, num_results, num_burnin_steps)

def run_nuts(data, num_chains, num_results, num_burnin_steps):
    rbc_joint_instance = JointLikelihood()  
    return rbc_joint_instance.run_nuts(data, num_chains, num_results, num_burnin_steps)