import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import multivariate_normal 

def generate_kent_samples(mean_dir, kappa, beta, size):
    # Calculating orthogonal minor and third axes
    temp = np.cross(mean_dir, np.array([1, 0, 0]))  
    gamma_2 = np.cross(mean_dir, temp) / np.linalg.norm(np.cross(mean_dir, temp))
    gamma_3 = np.cross(mean_dir, gamma_2)

    # Constructing the covariance matrix
    cov_11 = 1 / kappa
    cov_22 = (1 / kappa) * (1 + beta / 2)
    cov_33 = (1 / kappa) * (1 - beta / 2)
    cov_12 = cov_13 = cov_23 = 0  
    covariance_matrix = np.array([
        [cov_11, cov_12, cov_13],
        [cov_12, cov_22, cov_23],
        [cov_13, cov_23, cov_33]
    ])

    # Generate samples
    mean = mean_dir * kappa 
    samples = multivariate_normal.rvs(mean=mean, cov=covariance_matrix, size=size)

    # Project onto the unit sphere
    samples = samples / np.linalg.norm(samples, axis=1)[:, np.newaxis]   
    return samples

# Parameters for Kent distribution 
mean_direction = np.array([0.5, 0.5, 0.707]) 
kappa = 5      
beta = 2
num_samples = 1000

# Generate grid
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi, 100)
theta, phi = np.meshgrid(theta, phi)
x = np.cos(theta) * np.sin(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(phi)

# Generate samples
samples = generate_kent_samples(mean_direction, kappa, beta, num_samples)
x_samples, y_samples, z_samples = samples.T  

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere
ax.plot_surface(x, y, z, rstride=5, cstride=5, color='lightgray', alpha=0.5)

# Plot the samples
ax.scatter(x_samples, y_samples, z_samples, s=10, color='red')  

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Kent Distribution Samples')
plt.show()
