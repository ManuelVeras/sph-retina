from kent_distribution import *
from numpy.random import seed, uniform, randint	
import warnings
import pdb
import sys
import json 

from matplotlib import pyplot as plt

import pdb
from skimage.io import imread

class Rotation:
    @staticmethod
    def Rx(alpha):
        return asarray([[1, 0, 0], [0, cos(alpha), -sin(alpha)], [0, sin(alpha), cos(alpha)]])

    @staticmethod
    def Ry(beta):
        return asarray([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])

    @staticmethod
    def Rz(gamma):
        return asarray([[cos(gamma), -sin(gamma), 0], [sin(gamma), cos(gamma), 0], [0, 0, 1]])

def projectEquirectangular2Sphere(u, w, h):
   #NOTE: phi and theta differ from usual definition
   theta = u[:,1] * (pi/float(h))
   phi = u[:,0] * (2.*pi/float(w))
   return vstack([cos(theta), sin(theta)*cos(phi), sin(theta)*sin(phi)]).T

def projectSphere2Equirectangular(x, w, h):
   #NOTE: phi and theta differ from usual definition
   theta = squeeze(asarray(arccos(clip(x[:,0],-1,1))))
   phi = squeeze(asarray(arctan2(x[:,2],x[:,1])))
   phi[phi < 0] += 2*pi 
   return vstack([phi * float(w)/(2*pi), theta * float(h)/(pi)])

def createSphere(I):
	h, w = I.shape #960, 1920
	#pdb.set_trace()
	v, u = mgrid[0:h:1, 0:w:1]
	#print(u.max(), v.max()) # u in [0,w), v in [0,h)]
	X = projectEquirectangular2Sphere(vstack((u.reshape(-1),v.reshape(-1))).T, w, h)
	return X, I.reshape(-1)


def plotSphere(X, c):
	x, y, z = X.T
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.set_box_aspect([1,1,1])
	ax.scatter(x, y, z, c=c)	
	plt.show()

def selectAnnotation(annotations, class_name=None):
	idx = 0
	rnd = randint(0,len(annotations['boxes']))
	for ann in annotations['boxes']:
		if class_name and ann[6] == class_name: return ann
		elif class_name is None:
			if idx == rnd: return ann
			else: idx+=1 

def sampleFromAnnotation_deg(annotation, shape):
	h, w = shape
	phi_deg, theta_deg, fov_h, fov_v = annotation
	#x, y, _, _, fov_h, fov_v, label = annotation

	# Transform absolute coordinates (data_x, data_y) of an equirectangular image
	# to angular coordinates (phi00, theta00).
	# phi00: Longitude, ranges from -π to π.
	# theta00: Latitude, ranges from -π/2 to π/2.	
	
	#phi00 = (x - w / 2.) * ((2. * pi) / w)
	#theta00 = (y - h / 2.) * (pi / h)
	
	#phi_deg = rad2deg(phi00)+180
	#theta_deg = rad2deg(theta00)+90

	phi00 = deg2rad(phi_deg-180)
	theta00 = deg2rad(theta_deg-90)

	a_lat = deg2rad(fov_v)
	a_long = deg2rad(fov_h)
	
	r = 11
	d_lat = r / (2 * tan(a_lat / 2))
	d_long = r / (2 * tan(a_long / 2))
	
	p = []
	for i in range(-(r - 1) // 2, (r + 1) // 2):
		for j in range(-(r - 1) // 2, (r + 1) // 2):
			p += [asarray([i * d_lat / d_long, j, d_lat])]

	R = dot(Rotation.Ry(phi00), Rotation.Rx(theta00))
	p = asarray([dot(R, (p[ij] / norm(p[ij]))) for ij in range(r * r)])

	phi = asarray([arctan2(p[ij][0], p[ij][2]) for ij in range(r * r)])
	theta = asarray([arcsin(p[ij][1]) for ij in range(r * r)])
	u = (phi / (2 * pi) + 1. / 2.) * w
	v = h - (-theta / pi + 1. / 2.) * h
	return  projectEquirectangular2Sphere(vstack((u,v)).T, w, h)



def sampleFromAnnotation(annotation, shape):
	h, w = shape
	#phi_deg, theta_deg, _, _, fov_h, fov_v, label = annotation
	x, y, _, _, fov_h, fov_v, label = annotation

	# Transform absolute coordinates (data_x, data_y) of an equirectangular image
	# to angular coordinates (phi00, theta00).
	# phi00: Longitude, ranges from -π to π.
	# theta00: Latitude, ranges from -π/2 to π/2.	
	
	phi00 = (x - w / 2.) * ((2. * pi) / w)
	theta00 = (y - h / 2.) * (pi / h)

	#phi00 = deg2rad(phi_deg-180)
	#theta00 = deg2rad(theta_deg-90)

	a_lat = deg2rad(fov_v)
	a_long = deg2rad(fov_h)
	
	r = 11
	d_lat = r / (2 * tan(a_lat / 2))
	d_long = r / (2 * tan(a_long / 2))
	
	p = []
	for i in range(-(r - 1) // 2, (r + 1) // 2):
		for j in range(-(r - 1) // 2, (r + 1) // 2):
			p += [asarray([i * d_lat / d_long, j, d_lat])]

	R = dot(Rotation.Ry(phi00), Rotation.Rx(theta00))
	p = asarray([dot(R, (p[ij] / norm(p[ij]))) for ij in range(r * r)])

	phi = asarray([arctan2(p[ij][0], p[ij][2]) for ij in range(r * r)])
	theta = asarray([arcsin(p[ij][1]) for ij in range(r * r)])
	u = (phi / (2 * pi) + 1. / 2.) * w
	v = h - (-theta / pi + 1. / 2.) * h
	return projectEquirectangular2Sphere(vstack((u,v)).T, w, h)



def tlts_kent_me(xs, xbar):
	"""Generates and returns a KentDistribution based on a moment estimation."""
	lenxs = len(xs)
	#pause()
	xbar = average(xs, 0) # average direction of samples from origin
	S = average(xs.reshape((lenxs, 3, 1))*xs.reshape((lenxs, 1, 3)), 0) # dispersion (or covariance) matrix around origin
	gamma1 = xbar/norm(xbar) # has unit length and is in the same direction and parallel to xbar
	theta, phi = KentDistribution.gamma1_to_spherical_coordinates(gamma1)

	H = KentDistribution.create_matrix_H(theta, phi)
	Ht = KentDistribution.create_matrix_Ht(theta, phi)
	B = MMul(Ht, MMul(S, H))
	eigvals, eigvects = eig(B[1:,1:])
	eigvals = real(eigvals)
	if eigvals[0] < eigvals[1]:
		eigvals[0], eigvals[1] = eigvals[1], eigvals[0]
		eigvects = eigvects[:,::-1]
	K = diag([1.0, 1.0, 1.0])
	K[1:,1:] = eigvects
  
	G = MMul(H, K)
	Gt = transpose(G)
	T = MMul(Gt, MMul(S, G))
  
	r1 = norm(xbar)
	t22, t33 = T[1, 1], T[2, 2]
	r2 = t22 - t33
  
	# kappa and beta can be estimated but may not lie outside their permitted ranges
	min_kappa = 1E-6
	kappa = max( min_kappa, 1.0/(2.0-2.0*r1-r2) + 1.0/(2.0-2.0*r1+r2)  )
	beta  = 0.5*(1.0/(2.0-2.0*r1-r2) - 1.0/(2.0-2.0*r1+r2))

	print(f'kappa, beta  = {kappa}, {beta}')
  
	return kent4(G, kappa, beta)

def deg2kent(annotation, h = 960, w = 1920):
	Xs = sampleFromAnnotation_deg(annotation, (h,w))
	k = kent_me(Xs)
	return [k.theta, k.phi, k.psi, k.kappa, k.beta]

if __name__ == '__main__':

	I = imread('7fB0x.jpg', as_gray=True)
	X, C = createSphere(I)
	h, w = 	I.shape
	with open('7fB0x.json') as file: A = json.load(file)
	
	annotation = selectAnnotation(A, 'clock')
	#x, y, _, _, data_fov_h, data_fov_v, label = annotation
	pdb.set_trace()
	
	#phi, theta = 2*pi*data_x/I.shape[1], pi*data_y/I.shape[0]
	#beta = deg2rad(data_fov_v)/deg2rad(data_fov_h)
	
	#beta = deg2rad(tan(data_fov_v / 2))/deg2rad(tan(data_fov_h / 2))
	#psi = 0
	#xbar = asarray([cos(theta), sin(theta)*cos(phi), sin(theta)*sin(phi)])

	#o ultimo parametro de Q ta dando diferente, pelo jeito
 
	annotation = [0.0000, 0.0000, 0, 0, 31.8198, 15.9099, 0]

	Xs = sampleFromAnnotation(annotation, I.shape)

	k = kent_me(Xs) #WARNING: 
	#k = tlts_kent_me(Xs, xbar)
	#k = kent_mle(Xs, warning=sys.stdout)
	#P = k.pdf(X, normalize=False)
	print(k) # theta, phi, psi, kappa, beta	


	phi00 = (x - w / 2.) * ((2. * pi) / w)
	theta00 = (y - h / 2.) * (pi / h)

	print(phi00, theta00)

	annotation = [273.09375, 87.65625, 8., 8.]

	#Xs = sampleFromAnnotation_deg(annotation, I.shape)

	#k = kent_me(Xs) #WARNING: 
	#k = tlts_kent_me(Xs, xbar)
	#k = kent_mle(Xs, warning=sys.stdout)
	#P = k.pdf(X, normalize=False)
	#print(k) # theta, phi, psi, kappa, beta

	#phi_deg, theta_deg, _, _, fov_h, fov_v, label = annotation

	##phi00 = deg2rad(phi_deg-180)
	#theta00 = deg2rad(theta_deg-90)

	#print(phi00, theta00)

	kent_list = deg2kent(annotation)