from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from physics import Physics


class Noise_Models:

	@staticmethod
	def noise_function(ell, delta, beam_FWHM, cut_off=1e6):
		noise=delta**2*np.exp(ell*(ell+1)*beam_FWHM**2/(8*np.log(2)))
		noise[np.where(ell*(ell+1)*noise>cut_off)[0]]=None
		return noise
	@staticmethod
	def s4_noise(ell):
		return Noise_Models.noise_function(ell, 1.0/60/180*np.pi, 3.0/60/180*np.pi)
	@staticmethod
	def planck_noise(ell):
		return Noise_Models.noise_function(ell, 7.1/60/180*np.pi, 37/60/180*np.pi)
	@staticmethod
	def spt_noise(ell):
		return Noise_Models.noise_function(ell, 1.1/60/180*np.pi, 2.5/60/180*np.pi)
	@staticmethod
	def act_noise(ell):
		return Noise_Models.noise_function(ell, 1.4/60/180*np.pi, 8.9/60/180*np.pi)

def cosmic_variance(ell, Cl, noise_model=None):
	if noise_model is None:
		return np.sqrt(2/(2*ell+1))*np.fabs(Cl)
	else:
		return np.sqrt(2/(2*ell+1)*(Cl+noise_model(ell))**2)

"""
ells=np.logspace(0, 3, 1000)

phys=Physics()

plt.loglog(ells, ells*(ells+1)*Noise_Models.s4_noise(ells)/(2*np.pi)/phys.T0**2, label="CMB-S4")
plt.loglog(ells, ells*(ells+1)*Noise_Models.planck_noise(ells)/(2*np.pi)/phys.T0**2, label="Planck 143 Ghz")
plt.loglog(ells, ells*(ells+1)*Noise_Models.spt_noise(ells)/(2*np.pi)/phys.T0**2, label="SPT-3G")
plt.loglog(ells, ells*(ells+1)*Noise_Models.act_noise(ells)/(2*np.pi)/phys.T0**2, label="ACTPol")
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell(\ell+1)N_\ell^{TT}/2\pi$")
plt.legend()
plt.show()
"""