import torch
from kent_formator import deg2kent_single_torch

def spherical_quadrangle_covariance(lon_center, lat_center, h_fov, v_fov):
    """
    Computes the covariance matrix of a uniform distribution over a spherical quadrangle
    defined by the center coordinates and field of view (FOV) parameters.
    
    Parameters:
    lat_center (float): Latitude of the center of the spherical quadrangle (in degrees).
    lon_center (float): Longitude of the center of the spherical quadrangle (in degrees).
    h_fov (float): Horizontal field of view (in degrees).
    v_fov (float): Vertical field of view (in degrees).
    
    Returns:
    torch.Tensor: 3x3 covariance matrix.
    """
    # Convert degrees to radians
    lat_center = torch.deg2rad(torch.tensor(lat_center, dtype=torch.float32, requires_grad=True))
    lon_center = torch.deg2rad(torch.tensor(lon_center, dtype=torch.float32, requires_grad=True))
    h_fov = torch.deg2rad(torch.tensor(h_fov, dtype=torch.float32))
    v_fov = torch.deg2rad(torch.tensor(v_fov, dtype=torch.float32))
    
    # Convert latitude to theta (colatitude) and longitude to phi (azimuth)
    theta_center = torch.pi / 2 - lat_center  # Convert latitude to colatitude
    phi_center = lon_center % (2 * torch.pi)  # Normalize longitude to [0, 2*pi]
    #('phi_center', phi_center)
    
    # Calculate the bounds of the quadrangle
    theta1 = torch.clamp(theta_center - v_fov / 2, 0, torch.pi)  # Latitude bounds converted to colatitude
    theta2 = torch.clamp(theta_center + v_fov / 2, 0, torch.pi)
    phi1 = (phi_center - h_fov / 2) % (2 * torch.pi)  # Azimuth bounds, wrapped around
    phi2 = (phi_center + h_fov / 2) % (2 * torch.pi)
    
    
    
    # If phi1 > phi2, it indicates wrapping around the longitude, need to handle that
    if phi1 > phi2:
        phi1, phi2 = phi2, phi1  # Swap to handle wrapping correctly

    # Define the area of the spherical quadrangle for normalization
    def spherical_area(phi1, phi2, theta1, theta2):
        area = (phi2 - phi1) * (torch.cos(theta1) - torch.cos(theta2))
        return area

    A = spherical_area(phi1, phi2, theta1, theta2)
    #print(A)
    
    # Define the integrals for the mean components
    def mean_integral_phi_cos(theta1, theta2):
        return torch.sin(theta2) - torch.sin(theta1)

    def mean_integral_theta_sin_cos(phi1, phi2):
        return (phi2 - phi1) / 2

    # Calculate mean components
    mu1 = (1 / A) * mean_integral_theta_sin_cos(phi1, phi2) * mean_integral_phi_cos(theta1, theta2)
    mu2 = (1 / A) * mean_integral_theta_sin_cos(phi1, phi2) * mean_integral_phi_cos(theta1, theta2)
    mu3 = (1 / A) * (phi2 - phi1) * (torch.cos(theta1) - torch.cos(theta2))

    # Mean vector
    mean_vector = torch.tensor([mu1, mu2, mu3])

    # Define integrals for the covariance components
    def covariance_integral_phi_cos(theta1, theta2):
        return (torch.sin(theta2) * (2 + torch.cos(theta2)**2) / 3) - \
               (torch.sin(theta1) * (2 + torch.cos(theta1)**2) / 3)

    # Calculate covariance components
    sigma11 = (1 / A) * mean_integral_theta_sin_cos(phi1, phi2) * covariance_integral_phi_cos(theta1, theta2) - mu1**2
    sigma12 = (1 / A) * (0.5 * (torch.sin(phi2)**2 - torch.sin(phi1)**2)) * covariance_integral_phi_cos(theta1, theta2) - mu1 * mu2
    sigma13 = (1 / A) * (0.5 * (torch.sin(phi2) - torch.sin(phi1))) * (mean_integral_phi_cos(theta1, theta2))**2 - mu1 * mu3
    sigma22 = sigma11  # Due to symmetry in the uniform distribution
    sigma23 = sigma13
    sigma33 = (1 / A) * (phi2 - phi1) * covariance_integral_phi_cos(theta1, theta2) - mu3**2

    # Covariance matrix
    covariance_matrix = torch.tensor([
        [sigma11, sigma12, sigma13],
        [sigma12, sigma22, sigma23],
        [sigma13, sigma23, sigma33]
    ])

    return covariance_matrix


def first_term_covariance(phi1, phi2, theta1, theta2):
    
    theta1 = torch.deg2rad(torch.tensor(theta1, dtype=torch.float32))
    theta2 = torch.deg2rad(torch.tensor(theta2, dtype=torch.float32))
    phi1 = torch.deg2rad(torch.tensor(phi1, dtype=torch.float32))
    phi2 = torch.deg2rad(torch.tensor(phi2, dtype=torch.float32))
    
    def spherical_area(phi1, phi2, theta1, theta2):
        area = (phi2 - phi1) * (torch.cos(theta1) - torch.cos(theta2))
        return area

    A = spherical_area(phi1, phi2, theta1, theta2)
    mu1 = (torch.sin(phi2) - torch.sin(phi1)) / A * 0.5 * (theta2 - theta1 - (torch.sin(2 * theta2) - torch.sin(2 * theta1)) / 2)
    
    print('area', A)
    print('mu1', mu1)
    
    return ((phi2 - phi1) / 2 + (torch.sin(2 * phi2) - torch.sin(2 * phi1)) / 4) * (1 / A) * \
              (-torch.cos(theta2) * (2 + torch.cos(theta2)**2) / 3 + torch.cos(theta1) * (2 + torch.cos(theta1)**2) / 3) - mu1**2

# Test the function with example values
lat_center = 30.0  # Latitude center in degrees
lon_center = 306.0  # Longitude center in degrees
h_fov = 30.0       # Horizontal FOV in degrees
v_fov = 30.0        # Vertical FOV in degrees

'''cov_matrix = spherical_quadrangle_covariance(lon_center, lat_center, h_fov, v_fov)
print(first_term_covariance(0+30, 20+30, 10, 30))
print(first_term_covariance(0, 20, 10, 30))
#print(first_term)
cov_matrix = deg2kent_single(torch.tensor([350,0, 20,20]), 480, 960)'''

print(deg2kent_single_torch(torch.tensor([350,0, 20,20]), 480, 960))