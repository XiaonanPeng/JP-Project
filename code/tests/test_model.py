import pytest
from sympy import Function, Matrix, S

# Import the RBC class
from src.model import RBC

@pytest.fixture
def rbc_model():
    # This fixture creates an RBC model instance for use in tests
    model = RBC()
    model.set_parameter_values(0.33, 0.96, 0.95, 0.02, 0.01)
    return model

def test_parameter_setting(rbc_model):
    # Test if the parameters are set correctly
    assert rbc_model.alpha.subs(rbc_model.numerical_params) == 0.33
    assert rbc_model.beta.subs(rbc_model.numerical_params) == 0.96
    assert rbc_model.rho.subs(rbc_model.numerical_params) == 0.95
    assert rbc_model.delta.subs(rbc_model.numerical_params) == 0.02
    assert rbc_model.sigma.subs(rbc_model.numerical_params) == 0.01

def test_steady_state_numerical(rbc_model):
    # Expected steady state values calculated manually
    expected_k_bar = ((1 / 0.96 - 1 + 0.02) / 0.33) ** (1 / (0.33 - 1))
    expected_c_bar = expected_k_bar ** 0.33 - 0.02 * expected_k_bar
    expected_q_bar = expected_k_bar ** 0.33
    expected_z_bar = 0.0  # z_bar should be zero in the steady state

    # Extract the actual numerical steady state values from the model
    k_t_num = rbc_model.x_bar_num[0]
    z_t_num = rbc_model.x_bar_num[1]
    c_t_num = rbc_model.y_bar_num[0]
    q_t_num = rbc_model.y_bar_num[1]

    # Assert that the computed steady state values match the expected values
    assert abs(k_t_num - expected_k_bar) < 1e-4, "k_t numerical value does not match expected value"
    assert z_t_num == expected_z_bar, "z_t numerical value should be zero"
    assert abs(c_t_num - expected_c_bar) < 1e-4, "c_t numerical value does not match expected value"
    assert abs(q_t_num - expected_q_bar) < 1e-4, "q_t numerical value does not match expected value"

def test_state_space_representation_symbols(rbc_model):
    # Test if the state space representation is correctly defined in symbols
    assert isinstance(rbc_model.x, Matrix)
    assert isinstance(rbc_model.y, Matrix)
    assert all(isinstance(elem, (Function, S.One)) for elem in rbc_model.x)
    assert all(isinstance(elem, (Function, S.One)) for elem in rbc_model.y)

def test_model_equations_at_steady_state(rbc_model):

    # Substitute steady state values into the model equations
    H_with_params = [eq.subs(rbc_model.numerical_params) for eq in rbc_model.H]

    # Create a dictionary with the steady state values for k, z, c, q
    steady_state_vars = {
        rbc_model.k_t: rbc_model.x_bar_num[0],
        rbc_model.z_t: rbc_model.x_bar_num[1],
        rbc_model.c_t: rbc_model.y_bar_num[0],
        rbc_model.q_t: rbc_model.y_bar_num[1],
        rbc_model.k(rbc_model.t+1): rbc_model.x_bar_num[0],
        rbc_model.z(rbc_model.t+1): rbc_model.x_bar_num[1],
        rbc_model.c(rbc_model.t+1): rbc_model.y_bar_num[0],
        rbc_model.q(rbc_model.t+1): rbc_model.y_bar_num[1]
    }

    # Now substitute the steady state values into the equations
    H_at_steady_state = [eq.subs(steady_state_vars) for eq in H_with_params]

    # Check if each equation is numerically close to 0
    for i, eq in enumerate(H_at_steady_state, start=1):
        assert abs(eq) < 1e-4, f"Equation H[{i}] at steady state is not close to 0. Value: {eq}"

def test_model_derivatives(rbc_model):
    rbc_model.compute_derivatives()
    assert rbc_model.dx_bar_dtheta.shape == (rbc_model.n_x, rbc_model.n_p), "dx_bar_dtheta does not match the expected shape"
    assert rbc_model.H_x.shape == (rbc_model.n_x + rbc_model.n_y, rbc_model.n_x), "H_x does not match the expected shape"
    assert rbc_model.H_x_x.shape == (rbc_model.n_x + rbc_model.n_y, rbc_model.n_x, rbc_model.n_x), "H_x_x does not match the expected shape"
    assert rbc_model.pH_x_ptheta.shape == (rbc_model.n_x + rbc_model.n_y, rbc_model.n_x, rbc_model.n_p), "pH_x_dptheta does not match the expected shape"
    



    