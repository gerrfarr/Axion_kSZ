from __future__ import division
from helpers import *
from numerics import interpolate,differentiate,num
from scipy.optimize import newton_krylov

class Physics:

	def __init__(self, GAUSS=True, PRINT_PARAMS=True, READ_H=True):

		##DEFINE CONSTANTS
		self.ZMAX=1100 	#redshift at suface of last scattering
		self.H0INV=3000	#h/H0INV=H0 in units where c=1 (also: Hubble radius)
		self.T0=2.73e6 	#CMB temperature in muK
		self.SIGMA_T=6.6e-25 	#cm^2 Thompson cross section
		self.Mpc_to_cm=3.086e24 	#cm/Mpc
		self.Mpc_to_m=3.08568e22 #m/Mpc
		self.E_to_invL=1/1.9746359e-7 #m^-1/eV
		self.RHO_C=1.86e-29 		#g/cm^-3/h^2 (rho_B=Omega_Bh2 RHO_C)
		self.m_planck_L=1.91183e57 #1/Mpc
		self.delta_crit=1.686
		self.m_sun=1.988e33 #g
		self.m_proton=1.6726219e-24 #g

		##DEFINE PARAMETERS

		#Realistic values LambdaCDM
		self.Omega0=0.314626	#matter density in units of the critical density
		self.OmegaR=9.28656e-5 	#radiation density in units of the critical density
		self.OmegaB=0.0491989	#baryonic matter density in units of the critical density
		self.h=0.6737 			#h parameter defiend by H0=100 h km/sec Mpc^-1
		self.n=0.9652 			#intial tilt of the power spectrum
		self.tau=0.0543 		#optical depth to reionization
		self.logAs_1010=3.043 		#normalization of the power spectrum
		
		self.OmegaLambda=1-self.Omega0-self.OmegaR #LAMBDA parameter
		self.OmegaK=0.0#1-self.OmegaLambda-self.Omega0 #curvature of the universe
		self.sig8=0.81 #sigma8 normalization of power spectrum
		
		self.z_r=None 			#redshift at reionization
		self.delta_w=10 		#width of the rescattering surface
		self.x_e=1.0 			#ionization fraction
		

		self.f_axion=0.0 					#fraction of dark matter in the form of axions
		self.f_CDM=1.0-self.f_axion 		#fraction of dark matter in the form of CDM

		self.m_axion=1e-16		#axion mass eV
		self.ma_L=self.m_axion*self.E_to_invL*self.Mpc_to_m
		self.w_a=0
		

		self.Omega_Bh2=self.OmegaB * self.h**2 	#baryon denisty in units of the critical density times h^2
		self.H0=self.h/self.H0INV 	#Hubble constant
		self.a0=1 					#scale factor today

		self.rho_crit=8.11299e-11*self.h**2 #eV^4
		self.rho_crit_L=self.rho_crit*self.E_to_invL**4*self.Mpc_to_m**4

		self.rho0=self.Omega0*self.RHO_C*self.h**2*self.Mpc_to_cm**3/self.m_sun

		self.__num=num

		self.__a_vals=np.logspace(-5, 0, 10000)
		self.__z_vals=np.flip((1-self.__a_vals)/self.__a_vals, 0)

		"""
		try:
			self.z_osc=newton_krylov(lambda z:3*self.H0*self.E(z)-self.ma_L, 1000)
		except ValueError:
			self.z_osc=np.nan
		self.a_osc=1/(1+self.z_osc)"""

		aa=(46.9*self.Omega0*self.h**2)**0.67*(1.0+(32.1*self.Omega0*self.h**2)**(-0.532))
		ab=(12.0*self.Omega0*self.h**2)**0.424*(1.0+(45.*self.Omega0*self.h**2)**(-0.582))
		al=aa**(-self.Omega_Bh2/self.h**2/self.Omega0)*ab**((self.Omega_Bh2/self.h**2/self.Omega0)**3)

		self.correctionQ=1/np.sqrt(al)*1.0104**2/np.sqrt(1.0-self.Omega_Bh2/self.h**2/self.Omega0)
		
		if PRINT_PARAMS:
			print("Omega0={}, OmegaLambda={}, Omega_Bh2={}, OmegaK={}".format(self.Omega0, self.OmegaLambda, self.Omega_Bh2, 1-self.Omega0-self.OmegaLambda-self.OmegaR))
			print("AxionFrac={}, m_axion={}".format(self.f_axion, self.m_axion))
			print("n={}, h={}".format(self.n, self.h))
			#print("z_r={}, tau_r={}, x_e={}, delta_w={}, w_r={}".format(self.z_r,self.tau,self.x_e,self.delta_w, self.w_r))

		self.__wz_interp=None
		self.__zw_interp=None
		self.__tz_interp=None
		self.__zt_interp=None
		self.__dz_interp=None

		self.__READ_H=READ_H
		self.__H_interp=None

	@staticmethod
	def create_parameter_set(axion_mass=1e-24, axion_frac=0.0, omega0h2=0.314626*0.6737**2, omegaBh2=0.0491989*0.6737**2, h=0.6737, ns=0.9652, logAs_1010=3.043, print=False, read_H=True):
		phys=Physics(True, False, read_H)
		phys.m_axion=axion_mass
		phys.f_axion=axion_frac
		phys.Omega0=omega0h2/h**2
		phys.OmegaB=omegaBh2/h**2
		phys.h=h
		phys.logAs_1010=logAs_1010
		phys.n=ns

		phys.OmegaLambda=1-phys.Omega0-phys.OmegaR
		phys.f_CDM=1.0-phys.f_axion

		phys.Omega_Bh2 = phys.OmegaB * phys.h ** 2
		phys.ma_L = phys.m_axion * phys.E_to_invL * phys.Mpc_to_m
		phys.rho0 = phys.Omega0 * phys.RHO_C * phys.h ** 2 * phys.Mpc_to_cm ** 3 / phys.m_sun

		aa = (46.9 * phys.Omega0 * phys.h ** 2) ** 0.67 * (1.0 + (32.1 * phys.Omega0 * phys.h ** 2) ** (-0.532))
		ab = (12.0 * phys.Omega0 * phys.h ** 2) ** 0.424 * (1.0 + (45. * phys.Omega0 * phys.h ** 2) ** (-0.582))
		al = aa ** (-phys.Omega_Bh2 / phys.h ** 2 / phys.Omega0) * ab ** ((phys.Omega_Bh2 / phys.h ** 2 / phys.Omega0) ** 3)

		phys.correctionQ = 1 / np.sqrt(al) * 1.0104 ** 2 / np.sqrt(1.0 - phys.Omega_Bh2 / phys.h ** 2 / phys.Omega0)

		if print:
			phys.print_params()

		return phys

	def __hash__(self):
		return hash((self.m_axion, self.f_axion, self.Omega0, self.OmegaB, self.h, self.logAs_1010, self.n))

	def print_params(self):
		print("Omega0={}, OmegaLambda={}, OmegaB={}".format(self.Omega0, self.OmegaLambda, self.OmegaB))
		print("AxionFrac={}, m_axion={}".format(self.f_axion, self.m_axion))
		print("n={}, h={}".format(self.n, self.h))

	def read_in_H(self, a_vals, H_vals_normalized):
		if np.any(H_vals_normalized[np.where(a_vals==1.0)[0]]!=1):
			raise ValueError("The data read in for H does not seem to be normalized!")
		self.__H_interp=interpolate(a_vals, H_vals_normalized)

	#factor in the friedman equation
	def E(self, z):
		if self.__READ_H:
			try:
				return self.__H_interp(1/(1+z))
			except TypeError:
				raise Exception("Interpolation of H is not defined. Read in H first using function read_in_H(a_vals, H_vals_normalized)")
		else:
			return np.sqrt(self.OmegaR*(1+z)**4+self.Omega0*(1+z)**3+self.OmegaLambda+(1-self.Omega0-self.OmegaLambda)*(1+z)**2)

	#comoving distance
	def w(self, z):
		a=self.a0/(1+z)
		return -1/self.H0*self.__num.integrate(lambda a: 1/(a**2*self.E((1-a)/a)), self.a0, a)

	#growth factor
	def D(self, z):
		integrand = lambda x: (1+x)/self.E(x)**3

		return 5*self.Omega0*self.E(z)/2*self.__num.integrate(lambda x: integrand(x/(1-x))*1/(1-x)**2, z/(1+z), 1)

	#time as a function of redshift
	def t(self, z):
		a=self.a0/(1+z)
		integrand = lambda x: 1/(x*self.E((1-x)/x))
		return 1/self.H0*self.__num.integrate(integrand, 0, a)

	def eta(self, z):
		a=self.a0/(1+z)
		return 1/self.H0*self.__num.integrate(lambda x: 1/(self.E((1-x)/x)*x**2), 0, a)

	#scale factor as a function of redshift
	def a(self, z):
		return self.a0/(1+z)

	#time derivative of the scale factor
	def da(self, z):
		return self.da_over_a(z)*self.a(z)

	def da_over_a(self, z):
		return self.H0*self.E(z)

	#second time derivative of the scale factor
	def dda(self, z):
		return self.dda_over_a(z)*self.a(z)

	def dda_over_a(self, z):
		return self.H0**2*(self.OmegaLambda-self.Omega0/2*(1+z)**3-self.OmegaR*(1+z)**4)

	#time derivative of the growth factor
	def dD(self, z, d_func=None):
		if d_func is None:
			if self.__dz_interp is not None:
				d_func=self.__dz_interp
			else:
				d_func=self.D

		dda_over_da=self.dda(z)/self.da(z)
		da_over_a=self.da(z)/self.a(z)
		return d_func(z)*(dda_over_da-da_over_a)+5*self.Omega0/2*da_over_a*((1+z)/self.E(z))**2

	#optical depth to reionization
	def tau_r(self, z_r):
		return (0.69*self.Omega_Bh2/self.h*self.x_e)*self.__num.integrate(lambda z:(1+z)**2/self.E(z), xmin=0, xmax=z_r)

	#visibility function
	def g(self, w):
		return (1-np.exp(-self.tau))/np.sqrt(2*np.pi*self.delta_w**2)*np.exp(-1/2*(w-self.w_r)**2/self.delta_w**2)

	def get_g_camb(self):
		import camb
		dw_dz=lambda z: 1/(self.a0*self.H0*self.E(z))
		z_vis=np.linspace(0, 20, 5000)
		pars = camb.set_params(H0=100*self.h, ombh2=self.Omega_Bh2, omch2=self.Omega0*self.h**2-self.Omega_Bh2, ns=self.n, tau=self.tau)
		data= camb.get_background(pars)
		back_ev = data.get_background_redshift_evolution(z_vis, ['x_e', 'visibility'], format='array')
		g_interpolation=glueAsymptode(interpolate(z_vis, back_ev[:,1]), min=0.1, minAsym=lambda z: interpolate(z_vis, back_ev[:,1])(0.1)*(1+z)**2)
		norm=self.__num.integrate(lambda z: g_interpolation(z)*dw_dz(z), min(z_vis), max(z_vis))
		g_func=lambda z: (1-np.exp(-self.tau))/norm*g_interpolation(z)

		return g_func


	def get_D_interpolation(self, z_vals=None, recompute=False, save=True):
		if recompute:
			if z_vals is None:
				z_vals=self.__z_vals
			d_vals=self.D(z_vals)
			if save:
				np.save("../InterpolationData/growthFactor", (z_vals,d_vals))	
		elif self.__dz_interp is None:
			z_vals,d_vals=np.load("../InterpolationData/growthFactor.npy")

		if self.__dz_interp is None or recompute:
			self.__dz_interp=interpolate(z_vals, d_vals)

		return self.__dz_interp

	def get_t_interpolation(self, z_vals=None, recompute=False, doInverse=False, save=True):
		if recompute:
			if z_vals is None:
				z_vals=self.__z_vals
			t_vals=self.t(z_vals)
			if save:
				np.save("../InterpolationData/time", (z_vals,t_vals))	
		elif self.__tz_interp is None:
			z_vals,t_vals=np.load("../InterpolationData/time.npy")

		if self.__tz_interp is None or recompute:
			self.__tz_interp=interpolate(z_vals, t_vals)
			self.__zt_interp=interpolate(np.flip(t_vals, 0), np.flip(z_vals,0))

		if doInverse:
			return self.__tz_interp,self.__zt_interp
		else:
			return self.__tz_interp

	def get_w_interpolation(self, z_vals=None, recompute=False, doInverse=False):
		if recompute:
			if z_vals is None:
				z_vals=self.__z_vals
			w_vals=self.w(z_vals)
			np.save("../InterpolationData/comovingDistance", (z_vals,w_vals))
		elif self.__wz_interp is None:
			z_vals,w_vals=np.load("../InterpolationData/comovingDistance.npy")

		if self.__wz_interp is None or recompute:
			self.__wz_interp=interpolate(z_vals, w_vals)
			self.__zw_interp=interpolate(w_vals, z_vals)


		if doInverse:
			return self.__wz_interp,self.__zw_interp
		else:
			return self.__wz_interp

	#primary mass power spectrum (Cold Dark Matter) (Bardeen, Bond, Kaiser, Szalay; 1986)
	def P_cdm(self, kp):
		return ((kp/self.H0)**self.n)*(Physics.trans(kp/(self.Omega0*self.h**2)*self.correctionQ))**2

	#mass power spectrum fuzzy dark matter (Hu, Barakana, Gruzinov; 2000)
	def P_fcdm(self, kp):
		kj_eq=9*(self.m_axion/1e-22)**0.5
		x=1.61*(self.m_axion/1e-22)**(1/18)*kp/kj_eq

		return (np.cos(x**3)/(1+x**8))**2*P_cdm(kp)

	#bbks transfer function as a function of q=k/Omega0*h^2
	@staticmethod
	def trans(q):
		return np.log(1+2.34*q)/(2.34*q)/((1+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**(1/4))