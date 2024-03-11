import numpy as np
import pytest
from src.model import RBC
from src.generate_perturbation import evaluate_first_order_conditions
from src.generate_perturbation_derivative import generalized_sylvester_solver, compute_jacobian_derivatives, solve_first_order_p 

@pytest.fixture
def rbc_model():
    rbc = RBC()  
    rbc.set_parameter_values(
        alpha_value=0.5,
        beta_value=0.95,
        rho_value=0.2,
        delta_value=0.02,
        sigma_value=0.01
    )
    rbc.compute_derivatives()
    return rbc

def test_generalized_sylvester_solver():
    # Generate matrices
    A = np.array([[1 ,-1 ,1],[1 ,1 ,-1],[1, 1, 1]])
    C = np.array([[2 ,-1 ,1],[2 ,1 ,-1],[1, 3, 10]])
    D = np.array([[8 ,1 ,6],[3 ,5 ,7],[4 ,9 ,2]])
    E = -np.eye(3)

    # Solve the equation using our function
    X = generalized_sylvester_solver(A, C, D, E)

    # Assert: verify whether E + AX + CXD = 0
    result = E + np.dot(A, X) + np.dot(C, np.dot(X, D))
    abs_diff = np.abs(result)
    mae = np.mean(abs_diff)
    np.testing.assert_allclose(mae, 0, atol=2)

def test_compute_jacobian_derivatives(rbc_model):
    # Compute the Jacobian matrices and its derivatives
    dH_x_dtheta, dH_y_dtheta, dH_xp_dtheta, dH_yp_dtheta = compute_jacobian_derivatives(rbc_model)
    
    # check the output shape
    assert dH_x_dtheta.shape == (rbc_model.n_p, rbc_model.n_x + rbc_model.n_y, rbc_model.n_x) 
    assert dH_y_dtheta.shape == (rbc_model.n_p, rbc_model.n_x + rbc_model.n_y, rbc_model.n_y)
    assert dH_xp_dtheta.shape == (rbc_model.n_p, rbc_model.n_x + rbc_model.n_y, rbc_model.n_x) 
    assert dH_yp_dtheta.shape == (rbc_model.n_p, rbc_model.n_x + rbc_model.n_y, rbc_model.n_y) 

def test_solve_first_order_p(rbc_model):

    # Call the function to test
    dgx_dtheta, dhx_dtheta = solve_first_order_p(rbc_model)

    # Check if the function returns numpy arrays
    assert isinstance(dgx_dtheta, np.ndarray), "dgx_dtheta should be a numpy array"
    assert isinstance(dhx_dtheta, np.ndarray), "dhx_dtheta should be a numpy array"

    dgx_dtheta_expected = np.array([
        [[-1.24652645e-01,  5.59621114e+00],
        [-1.66533601e-16,  6.68912336e+01]],

        [[-1.69467425e+00, -8.34362122e-01],
        [-1.10803324e+00,  1.05019939e+02]],

        [[-3.16037503e-17,  2.94500827e-01],
        [ 7.26298256e-33,  1.11022302e-16]]
    ])

    dhx_dtheta_expected = np.array([
        [[ 1.24652645e-01,  6.12950224e+01],
        [-1.77049387e-21,  2.81160248e-14]],

        [[ 5.86641013e-01,  1.05854302e+02],
        [ 6.93685138e-17,  4.59922268e-14]],

        [[ 3.16037503e-17, -2.94500827e-01],
        [ 1.06741052e-32,  1.00000000e+00]]
    ])

    np.testing.assert_allclose(dgx_dtheta, dgx_dtheta_expected, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(dhx_dtheta, dhx_dtheta_expected, rtol=1e-3, atol=1e-5)