#http://arxiv.org/abs/1506.08105

from numpy import *
from numpy.random import randn
from numpy.random import uniform
from scipy.special import jv as I_
from scipy.special import gamma as G_

from pdb import set_trace as pause
from matplotlib import pyplot as plt

def projectEquirectangular2Sphere(u, w, h):
   phi = u[:,1] * (pi/float(h))
   theta = u[:,0] * (2.*pi/float(w))
   sinphi = sin(phi)
   return vstack([sinphi * cos(theta), sinphi * sin(theta), cos(phi)]).T

def projectSphere2Equirectangular(x, w, h):
   phi = squeeze(asarray(arccos(clip(x[:,2],-1,1))))
   theta = squeeze(asarray(arctan2(x[:,1],x[:,0])))
   theta[theta < 0] += 2*pi 
   return vstack([theta * float(w)/(2.*pi), phi * float(h)/pi])

def angle2Gamma(alpha, eta, psi):
	gamma_1 = asarray([cos(alpha), 
					   sin(alpha)*cos(eta), 
					   sin(alpha)*sin(eta)])
	
	gamma_2 = asarray([-cos(psi)*sin(alpha), 
					   cos(psi)*cos(alpha)*cos(eta) - sin(psi)*sin(eta), 
					   cos(psi)*cos(alpha)*sin(eta) + sin(psi)*cos(eta)])
	
	gamma_3 = asarray([sin(psi)*sin(alpha), 
					   -sin(psi)*cos(alpha)*cos(eta) - cos(psi)*sin(eta), 
					   -sin(psi)*cos(alpha)*sin(eta) + cos(psi)*cos(eta)])
	
	return asarray([gamma_1, gamma_2, gamma_3])
'''
kappa is the concentration
beta is the ovalness
gamma_1 is the unit mean axis
gamma_2 is the unit major axis
gamma_3 is the unit minor axis
'''
def FB5(Theta, x):
	def __c(kappa, beta, terms = 10):
		su = 0
		for j in range(terms):
			su += G_(j+.5)/G_(j+1)*beta**(2*j)*(2/kappa)**(2*j+.5)*I_(2*j+.5, kappa)
		return 2*pi*su
	
	kappa, beta, Q = Theta
	gamma_1, gamma_2, gamma_3 = Q
	assert beta >= 0 and beta < kappa/2
	assert isclose(dot(gamma_1, gamma_2),0) and isclose(dot(gamma_2, gamma_3),0)
	# Eq. (2)
	return __c(kappa, beta)**(-1) * exp(kappa * dot(gamma_1, x) + beta * (dot(gamma_2,x)**2 - dot(gamma_3,x)**2))


'''
kappa is the concentration
gamma_1 is the unit mean vector
'''
def vMF(Theta, x):
	def __c(kappa): 
		#return 2/kappa * sinh(kappa)
		return (4*pi*sinh(kappa))/kappa
	kappa, gamma_1 = Theta
	# Eq. (?) # first equation 
	return __c(kappa)**(-1) * exp(kappa * dot(gamma_1, x)) 


if __name__ == '__main__':

	h, w = 32, 64
	v, u = mgrid[0:h:1, 0:w:1]
	X = projectEquirectangular2Sphere(vstack((u.reshape(-1),v.reshape(-1))).T, w, h)

	'''
	kappa = 3
	mu = randn(3); mu /= linalg.norm(mu)
	Theta = (kappa, mu)

	P = asarray([vMF(Theta, x) for x in X])
	'''

	kappa = uniform()
	beta = uniform() * .5*kappa
	alpha, eta, psi = [uniform() for _ in range(3)]
	Q = angle2Gamma(alpha, eta, psi)
	Theta = (kappa, beta, Q)

	P = asarray([FB5(Theta, x) for x in X])

	plt.imshow(P.reshape((h,w))); plt.show()

	#pause()




