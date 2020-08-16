import numpy as np
from scipy.interpolate import RectBivariateSpline
from physics import Physics

class GrowthInterpolation:

	def __init__(self, file_root, physics=None, smoothing=0.0, degree=3, SET_GLOBAL_EVOL=True):
		if physics is None:
			self.__phys=Physics()
		else:
			self.__phys=physics

		self.__f_data = np.loadtxt(file_root + '_devolution.dat')
		self.__delta_data = np.loadtxt(file_root + '_evolution.dat')
		a_data = np.loadtxt(file_root + '_a_vals.dat')
		self.__a_vals = a_data[:,0]
		assert(a_data[-1,0]==1.0)
		self.__k_vals = np.loadtxt(file_root + '_matterpower_out.dat')[:,0]*self.__phys.h

		if SET_GLOBAL_EVOL:
			physics.read_in_H(self.__a_vals, a_data[:,1]/a_data[-1,1])


		normalized_dataset=self.__delta_data/np.meshgrid(self.__a_vals, self.__delta_data[:,-1])[1]

		G_interpolation=RectBivariateSpline(np.log10(self.__k_vals), np.log10(self.__a_vals), normalized_dataset, bbox=[min(np.log10(self.__k_vals)), max(np.log10(self.__k_vals)), min(np.log10(self.__a_vals)), max(np.log10(self.__a_vals))], kx=degree, ky=degree, s=smoothing)
		self.__G_func= lambda a, k: np.squeeze(G_interpolation.ev(np.log10(k), np.log10(a)))

		f_interpolation = RectBivariateSpline(np.log10(self.__k_vals), np.log10(self.__a_vals), self.__f_data, bbox=[min(np.log10(self.__k_vals)), max(np.log10(self.__k_vals)), min(np.log10(self.__a_vals)), max(np.log10(self.__a_vals))], kx=degree, ky=degree, s=smoothing)
		self.__f_func = lambda a, k: np.squeeze(f_interpolation.ev(np.log10(k), np.log10(a)))

		self.__dG_dt=lambda a, k: self.__phys.da_over_a((1-a)/a)*np.squeeze(f_interpolation.ev(np.log10(a), np.log10(k)))*self.__G_func(a,k)

	def get_growth(self):
		return self.__G_func

	def get_dG_dt(self):
		return self.__dG_dt

	def get_dlogG_dlogA(self):
		return self.__f_func


