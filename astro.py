from astropy.modeling.models import FisherKappaDistribution
import numpy as np
import matplotlib.pyplot as plt

# Set up distribution parameters
kappa = 20
mean_lat, mean_long = 30, -50

# Generate samples
distribution = FisherKappaDistribution(kappa=kappa, mu=mean_long, scale=mean_lat)
lats, longs = distribution.rvs(size=1000)

# Plot the samples
plt.scatter(longs, lats, s=10)  
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'Kent Distribution Samples (Astropy) - kappa={kappa}')
plt.show()