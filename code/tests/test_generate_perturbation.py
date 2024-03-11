import pytest
import numpy as np
from src.model import RBC
from src.generate_perturbation import solve_first_order, evaluate_first_order_conditions

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

def test_evaluate_first_order_conditions(rbc_model):
      
    # Compute Jacobian matrix
    H_x, H_y, H_xp, H_yp = evaluate_first_order_conditions(rbc_model)
    
    # Check if Jacobian is numpy array
    assert isinstance(H_x, np.ndarray), "H_x should be a numpy array"
    assert isinstance(H_y, np.ndarray), "H_y should be a numpy array"
    assert isinstance(H_xp, np.ndarray), "H_xp should be a numpy array"
    assert isinstance(H_yp, np.ndarray), "H_yp should be a numpy array"

    # check the shape of Jacobians
    expected_shape_x = (rbc_model.n_x+rbc_model.n_y, rbc_model.n_x)
    expected_shape_y = (rbc_model.n_x+rbc_model.n_y, rbc_model.n_y) 
    assert H_x.shape == expected_shape_x, f"H_x should have shape {expected_shape_x}, but has {H_x.shape}"
    assert H_y.shape == expected_shape_y, f"H_y should have shape {expected_shape_y}, but has {H_y.shape}"
    assert H_xp.shape == expected_shape_x, f"H_xp should have shape {expected_shape_x}, but has {H_xp.shape}"
    assert H_yp.shape == expected_shape_y, f"H_yp should have shape {expected_shape_y}, but has {H_yp.shape}"
    # Check the values of Jacobian matrices
    np.testing.assert_allclose(H_x, np.array([[0.0, 0.0],
                                              [-0.98, 0.0],
                                              [-0.07263157894736837, -6.884057971014498],
                                              [0.0, -0.2]]), rtol=1e-4, atol=1e-4)
    
    np.testing.assert_allclose(H_y, np.array([[-0.0283775705621991, 0.0],
                                              [1.0, -1.0],
                                              [0.0, 1.0],
                                              [0.0, 0.0]]), rtol=1e-4, atol=1e-4)
    
    np.testing.assert_allclose(H_xp, np.array([[0.00012263591151906127, -0.011623494029190608],
                                               [1.0, 0.0],
                                               [0.0, 0.0],
                                               [0.0, 1.0]]), rtol=1e-4, atol=1e-4)
    
    np.testing.assert_allclose(H_yp, np.array([[0.028377570562199098, 0.0],
                                               [0.0, 0.0],
                                               [0.0, 0.0],
                                               [0.0, 0.0]]), rtol=1e-4, atol=1e-4)


def test_solve_first_order(rbc_model):

    # return the coefficients of linear model
    g_x, h_x = solve_first_order(rbc_model)

    # check if g_x and h_x is numpy array
    assert isinstance(g_x, np.ndarray), "g_x should be a numpy array"
    assert isinstance(h_x, np.ndarray), "h_x should be a numpy array"
    
    np.testing.assert_allclose(g_x, np.array([[0.09579643, 0.67468697],
                                             [0.07263158, 6.88405797]]), rtol=1e-4, atol=1e-4)
    
    np.testing.assert_allclose(h_x, np.array([[ 9.56835149e-01, 6.20937101e+00],
                                              [-4.52305977e-18, 2.00000000e-01]]), rtol=1e-4, atol=1e-4)
    