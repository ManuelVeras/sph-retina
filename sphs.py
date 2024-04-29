from sphstat.distributions import kent
import numpy as np
from sphstat.plotting import plotdata
from sphstat.utils import readsample, convert, polartocart, carttopolar

mu0 = np.array([0., 1., 0.])
sample = kent(30, 50, 50, np.array([1., 1., 1.]), mu0)
samplerad = carttopolar(sample)
print(samplerad['tetas'])
plotdata(samplerad, proj='mollweide')