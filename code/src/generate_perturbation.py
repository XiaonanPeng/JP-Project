from sympy import lambdify
import numpy as np
from scipy.linalg import ordqz, solve, lstsq

def evaluate_first_order_conditions(m):
    """
    This function computes the Jacobian matrices of the first-order conditions
    of the RBC model with respect to current and future state and control variables.
    It returns the matrices in numerical form.

    Parameters:
    m: Model instance with necessary properties and methods.

    Returns:
    H_x, H_y, H_xp, H_yp: Jacobians of operator H to x,y,xp,yp respectively, evaluated at the steady state.
    """
    # Substitute the steady state numerical values and param numerical values to Jacobians
    H_x = np.array(m.H_x.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    H_y = np.array(m.H_y.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    H_xp = np.array(m.H_xp.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    H_yp = np.array(m.H_yp.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)

    return H_x, H_y, H_xp, H_yp


def solve_first_order(m):
    """
    Solves for the first-order approximation of the policy functions.

    Parameters:
    m: Model instance with necessary properties and methods.
    ϵ_BK: Tolerance for the Blanchard-Kahn condition.

    Returns:
    g_x, h_x: Policy function coefficients for state and control variables.
    """
    
    # Evaluate first-order conditions to get Jacobian matrices
    H_x, H_y, H_xp, H_yp = evaluate_first_order_conditions(m)
    n_x = m.n_x
    
    # Construct matrix A,B for QZ decomposition
    A = np.hstack((H_xp, H_yp))
    B = np.hstack((H_x, H_y))

    # Perform QZ decomposition with sorting criterion
    crit = 1.0 - 1e-8
    def sort_criterion(alpha, beta):
        return abs(alpha)**2 > crit * abs(beta)**2
    
    try: 
        S, T, _, _, Q, Z = ordqz(A, B, sort=sort_criterion)

        # Compute the generalized eigenvalues
        gen_eigval = np.abs(np.diag(S) / np.diag(T))

        # Check Blanchard-Kahn condition
        if not np.all(gen_eigval[n_x:] < 1):
            raise ValueError("Blanchard-Kahn condition is not satisfied.")

        # Extract policy function coefficients
        Z11 = Z[:n_x, :n_x]
        Z12 = Z[:n_x, n_x:]
        Z21 = Z[n_x:, :n_x]
        Z22 = Z[n_x:, n_x:]
        S11 = S[:n_x, :n_x]
        T11 = T[:n_x, :n_x]

        # Compute policy functions g_x and h_x
        x1, _, _, _ = lstsq(Z22.T.conj(), Z12.T.conj())
        g_x = (-x1).real
        tmp, _, _, _ = lstsq(S11, T11 @ (Z11.T.conj() + Z21.T.conj() @ g_x))
        x2, _, _, _ = lstsq(Z11.T.conj() + Z21.T.conj() @ g_x, tmp)
        h_x = (-x2).real
    except Exception as e:
        # Catch any exception that occurred and print/log the parameters
        print("An error occurred:", e)
        print("Current parameter values:\n", m.numerical_params)
        raise 

    return g_x, h_x

"""
from src.model import RBC
m = RBC() 
m.set_parameter_values(
    alpha_value=0.4,
    beta_value=0.9887188673019409,
    rho_value=0.12258812040090561,
    delta_value=0.025,
    sigma_value=0.1
)
m.compute_derivatives()
g_x, h_x = solve_first_order(m)
print(g_x)
print(h_x)
"""

