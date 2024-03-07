import numpy as np
import scipy.linalg
from scipy.linalg import lstsq, kron, solve
from src.generate_perturbation import evaluate_first_order_conditions, solve_first_order

def generalized_sylvester_solver(A, C, D, E):
    """
    Solve the generalized Sylvester equation E + AX + CXD = 0 for X.
    
    Parameters:
    A -- np.array, coefficient matrix
    C -- np.array, coefficient matrix
    D -- np.array, coefficient matrix
    E -- np.array, right-hand side matrix
    
    Returns:
    X -- np.array, the solution matrix to the generalized Sylvester equation
    """
    # Convert the problem to the standard Sylvester equation form
    identity_D = np.eye(D.shape[0])

    # Use Kronecker products to form the coefficients of the vec(X)
    AD_kron = kron(identity_D, A)
    CTD_kron = kron(D.T, C)

    # Form the right-hand side of the equation
    vec_E = E.flatten()
    # Add a regularization term to improve the condition number
    reg_lambda = 4 * 1e-1
    reg_matrix = reg_lambda * np.eye(AD_kron.shape[0])
    combined_matrix = AD_kron + CTD_kron + reg_matrix

    # Solve the linear equation
    vec_X = solve(combined_matrix, -vec_E)
    cond_number = np.linalg.cond(combined_matrix)
    print('Condition number: ', cond_number)

    # Reshape the vectorized solution back to a matrix
    X = vec_X.reshape(E.shape)

    return X

def generalized_sylvester_solver2(A, C, D, E):
    """
    This is a more numerically stable function to solve generalized Sylvester equation
                            E + AX + CXD = 0 for X
    """
    # Perform the QZ decomposition on A, C
    AA, CC, Q1, Z = scipy.linalg.qz(A, C, output="complex")
    
    # Perform the Schur decomposition on D
    DD, Q2 = scipy.linalg.schur(D, output="complex")
    
    # Calculate `F` using transformations from QZ and Schur decompositions on `E`
    F = Q1.conj().T @ E @ Q2
    
    # Iteratively solve the upper triangular form Sylvester equation of Y: F + AA*Y + CC*Y*DD = 0
    Y = np.zeros_like(E, dtype=complex)
    for k in range(E.shape[1]):
        rhs = F[:,k] + CC @ Y[:,0:k] @ DD[0:k,k] 
        x, residuals, rank, s = lstsq(AA + DD[k,k]*CC, rhs)
        Y[:,k] = -x
        
    # Transform Y back into the original matrix X
    X = (Z @ Y @ Q2.conj().T).real

    return X

def compute_jacobian_derivatives(m):
    # Define the parameters which need to be estimated
    n_p = m.n_p

    # Compute derivative of steady states : Right!!!
    dx_bar_dtheta = np.array(m.dx_bar_dtheta.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    dy_bar_dtheta = np.array(m.dy_bar_dtheta.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)

    # Compute the Hessian matrices: Right!!!
    H_x_x = np.array(m.H_x_x.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    H_x_y = np.array(m.H_x_y.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    H_x_xp = np.array(m.H_x_xp.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    H_x_yp = np.array(m.H_x_yp.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    H_y_x = np.array(m.H_y_x.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    H_y_y = np.array(m.H_y_y.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    H_y_xp = np.array(m.H_y_xp.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    H_y_yp = np.array(m.H_y_yp.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    H_xp_x = np.array(m.H_xp_x.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    H_xp_y = np.array(m.H_xp_y.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    H_xp_xp = np.array(m.H_xp_xp.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    H_xp_yp = np.array(m.H_xp_yp.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    H_yp_x = np.array(m.H_yp_x.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    H_yp_y = np.array(m.H_yp_y.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    H_yp_xp = np.array(m.H_yp_xp.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    H_yp_yp = np.array(m.H_yp_yp.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    
    # compute the partial derivatives of Jacobian with respect to parameters :Right!!!
    pH_x_ptheta = np.array(m.pH_x_ptheta.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    pH_y_ptheta = np.array(m.pH_y_ptheta.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    pH_xp_ptheta = np.array(m.pH_xp_ptheta.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)
    pH_yp_ptheta = np.array(m.pH_yp_ptheta.subs(m.numerical_params).subs(m.numerical_steady_state), dtype = np.float64)

    # Initialize the Jacobian derivative matrices
    dH_x_dtheta = []
    dH_y_dtheta = []
    dH_xp_dtheta = []
    dH_yp_dtheta = []

    # For each theta_i, compute the derivative of Jacobian H_{x,y,xp,yp} w.r.t theta_i using chain rule
    for i in range(n_p):
        dH_x_dtheta_param = np.einsum('ijk,k -> ij' ,(H_x_yp + H_x_y), dy_bar_dtheta[:,i]) \
                            + np.einsum('ijk,k -> ij' ,(H_x_xp + H_x_x), dx_bar_dtheta[:,i]) \
                            + pH_x_ptheta[:,:,i]
        dH_x_dtheta.append(dH_x_dtheta_param)

        dH_y_dtheta_param = np.einsum('ijk,k -> ij' ,(H_y_yp + H_y_y), dy_bar_dtheta[:,i]) \
                            + np.einsum('ijk,k -> ij' ,(H_y_xp + H_y_x), dx_bar_dtheta[:,i]) \
                            + pH_y_ptheta[:,:,i]
        dH_y_dtheta.append(dH_y_dtheta_param)

        dH_xp_dtheta_param = np.einsum('ijk,k -> ij' ,(H_xp_yp + H_xp_y), dy_bar_dtheta[:,i]) \
                            + np.einsum('ijk,k -> ij' ,(H_xp_xp + H_xp_x), dx_bar_dtheta[:,i]) \
                            + pH_xp_ptheta[:,:,i]
        dH_xp_dtheta.append(dH_xp_dtheta_param)

        dH_yp_dtheta_param = np.einsum('ijk,k -> ij' ,(H_yp_yp + H_yp_y), dy_bar_dtheta[:,i]) \
                            + np.einsum('ijk,k -> ij' ,(H_yp_xp + H_yp_x), dx_bar_dtheta[:,i]) \
                            + pH_yp_ptheta[:,:,i]
        dH_yp_dtheta.append(dH_yp_dtheta_param)

    return np.array(dH_x_dtheta), np.array(dH_y_dtheta), np.array(dH_xp_dtheta), np.array(dH_yp_dtheta)

def solve_first_order_p(m):
    # Compute derivatives of H with respect to theta at the DSS for each theta_i.
    g_x, h_x = solve_first_order(m)

    # read the required sizes 
    n_x, n_y, n_p = m.n_x, m.n_y, m.n_p
    
    # Prepare matrices to store derivatives of g_x and h_x with respect to each theta_i.
    dgx_dtheta = np.zeros((n_p, n_y, n_x))
    dhx_dtheta = np.zeros((n_p, n_x, n_x))
    # compute the Jacobian and its derivative
    _, H_y, H_xp, H_yp = evaluate_first_order_conditions(m)
    dHx_dtheta, dHy_dtheta, dHxp_dtheta, dHyp_dtheta =compute_jacobian_derivatives(m)
    # Loop over each element of theta and solve the Sylvester equation.
    for i in range(n_p):
        dHyp_dtheta_i, dHy_dtheta_i, dHxp_dtheta_i, dHx_dtheta_i = dHyp_dtheta[i], dHy_dtheta[i], dHxp_dtheta[i], dHx_dtheta[i]

        # Stack the derivatives for the Sylvester equation.
        E = np.hstack([dHyp_dtheta_i, dHy_dtheta_i, dHxp_dtheta_i, dHx_dtheta_i]) @ np.vstack([g_x @ h_x, g_x, h_x, np.eye(n_x)])
        A = np.hstack([H_y, H_xp + H_yp @ g_x])
        C = np.hstack([H_yp, np.zeros((n_x+n_y, n_x))])
        D = h_x
        
        # Solve the Sylvester equation.
        solution = generalized_sylvester_solver2(A, C, D, E)
        
        # Store the solutions.
        dgx_dtheta[i,:,:] = solution[:n_y]
        dhx_dtheta[i,:,:] = solution[n_y:]
    
    return dgx_dtheta, dhx_dtheta

