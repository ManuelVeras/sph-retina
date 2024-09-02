import torch

def degrees_to_radians(degrees):
    """Convert degrees to radians."""
    return degrees * torch.pi / 180

def spherical_to_cartesian(theta, phi):
    """
    Convert spherical coordinates to Cartesian coordinates.
    Args:
    - theta (torch.Tensor): Polar angle (measured from z-axis)
    - phi (torch.Tensor): Azimuthal angle (measured from x-axis)
    
    Returns:
    - (torch.Tensor, torch.Tensor, torch.Tensor): Cartesian coordinates (Y1, Y2, Y3)
    """
    Y1 = torch.sin(theta) * torch.cos(phi)
    Y2 = torch.sin(theta) * torch.sin(phi)
    Y3 = torch.cos(theta)
    return Y1, Y2, Y3

def calculate_area(theta1, theta2, phi1, phi2):
    """
    Calculate the area of the spherical quadrangle.
    """
    # Ensure inputs are tensors
    if not isinstance(theta1, torch.Tensor):
        theta1 = torch.tensor(theta1, dtype=torch.float32)
    if not isinstance(theta2, torch.Tensor):
        theta2 = torch.tensor(theta2, dtype=torch.float32)
    if not isinstance(phi1, torch.Tensor):
        phi1 = torch.tensor(phi1, dtype=torch.float32)
    if not isinstance(phi2, torch.Tensor):
        phi2 = torch.tensor(phi2, dtype=torch.float32)
    
    return (phi2 - phi1) * (torch.cos(theta1) - torch.cos(theta2))

def calculate_sigma_11_analytical(theta1, theta2, phi1, phi2):
    """
    Calculate sigma_11 using analytical integration for the given bounds.
    """
    # Convert input angles from degrees to radians
    theta1 = degrees_to_radians(torch.tensor(theta1, dtype=torch.float32))  # Convert latitude to colatitude
    theta2 = degrees_to_radians(torch.tensor(theta2, dtype=torch.float32))  # Convert latitude to colatitude
    phi1 = degrees_to_radians(torch.tensor(phi1, dtype=torch.float32))
    phi2 = degrees_to_radians(torch.tensor(phi2, dtype=torch.float32))
    
    # Compute the area of the spherical quadrangle
    A = calculate_area(theta1, theta2, phi1, phi2)
    
    # Analytical integration to compute E[Y1^2]
    E_Y1_squared = (1/A)*(0.5)*(phi2-phi1+torch.sin(2*phi2)/2-torch.sin(2*phi1)/2) * (1/3) * (-torch.cos(theta2)+torch.cos(theta2)**3 + torch.cos(theta1) - torch.cos(theta1)**3)
    
    # Analytical integration to compute (E[Y1])^2
    E_Y1 = (1 / A) * (torch.sin(phi2) - torch.sin(phi1)) *(0.5)* (theta2 - theta1 - (torch.sin(2*theta2)-torch.sin(2*theta1))/2) 
    E_Y1_squared_term = E_Y1**2
    
    # Variance: sigma_11 = E[Y1^2] - (E[Y1])^2
    sigma_11 = E_Y1_squared - E_Y1_squared_term
    
    # Debug prints
    print(f"theta1: {theta1}, theta2: {theta2}, phi1: {phi1}, phi2: {phi2}")
    print(f"Area (A): {A}")
    print(f"E[Y1^2]: {E_Y1_squared}")
    print(f"(E[Y1])^2: {E_Y1_squared_term}")
    print(f"sigma_11: {sigma_11}")
    
    return sigma_11

def compute_sigma_11_numerical(theta1, theta2, phi1, phi2, num_samples=10000000):
    """
    Compute sigma_11 using Monte Carlo approximation.
    
    Args:
    - theta1, theta2, phi1, phi2 (float): Angle bounds in degrees.
    - num_samples (int): Number of samples for Monte Carlo approximation.
    
    Returns:
    - sigma_11 (float): Estimated variance.
    """
    # Convert input angles from degrees to radians
    theta1 = degrees_to_radians(theta1)
    theta2 = degrees_to_radians(theta2)
    phi1 = degrees_to_radians(phi1)
    phi2 = degrees_to_radians(phi2)
    
    # Generate random samples in the specified ranges
    theta_samples = torch.rand(num_samples) * (theta2 - theta1) + theta1
    phi_samples = torch.rand(num_samples) * (phi2 - phi1) + phi1
    
    # Convert to Cartesian coordinates
    Y1, _, _ = spherical_to_cartesian(theta_samples, phi_samples)
    
    # Compute mean and variance
    mean_Y1 = torch.mean(Y1)
    sigma_11 = torch.mean(Y1**2) - mean_Y1**2
    return sigma_11.item()

# Define spherical quadrangle bounds in degrees
theta1 = 10.0       # Start of theta range
theta2 = 50.0      # End of theta range
phi1 = 0.0         # Start of phi range
phi2 = 45.0        # End of phi range

# Compute sigma_11 analytically
sigma_11_analytical = calculate_sigma_11_analytical(theta1, theta2, phi1, phi2)
print(f"sigma_11 (analytical): {sigma_11_analytical}")

# Compute sigma_11 numerically using Monte Carlo method
sigma_11_numerical = compute_sigma_11_numerical(theta1, theta2, phi1, phi2)
print(f"sigma_11 (numerical, Monte Carlo): {sigma_11_numerical}")

# Compare analytical and numerical results
comparison = torch.isclose(torch.tensor(sigma_11_analytical), torch.tensor(sigma_11_numerical), atol=1e-4)
print(f"Analytical and numerical results are close: {comparison.item()}")

# Compute sigma_11 with a shift in longitude to test invariance
phi_shift = 90.0  # Shift longitude by 90 degrees

phi1 = (phi1 + phi_shift) % (2 * torch.pi)
phi2 = (phi2 + phi_shift) % (2 * torch.pi)
sigma_11_shifted_analytical = calculate_sigma_11_analytical(theta1, theta2, phi1, phi2)
print(f"sigma_11 (analytical, shifted): {sigma_11_shifted_analytical}")

# Check invariance
invariance_check = torch.isclose(torch.tensor(sigma_11_analytical), torch.tensor(sigma_11_shifted_analytical), atol=1e-4)
print(f"Variance invariant to longitude shift: {invariance_check.item()}")

# Debug prints for comparison
print(f"sigma_11_analytical: {sigma_11_analytical}")
print(f"sigma_11_shifted_analytical: {sigma_11_shifted_analytical}")
