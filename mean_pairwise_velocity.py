from numerics import interpolate,differentiate,num
from scipy.interpolate import interp1d
from read_mode_evolution import GrowthInterpolation
from helpers import *
from physics import Physics


class mean_pairwise_velocity:

	TOP_HAT=0
	GAUSSIAN=1
	SHARP_K=2
	NO_FILTER=3

	def __init__(self, power_spectrum, mode_evolution, kmin=1.0e-3, kmax=1.0e4, mMin=1e12, mMax=1e15, physics=None, AXIONS=True, jenkins_mass=False, window_function=GAUSSIAN):
		"""

		:type power_spectrum: function(float)->float
		:type mode_evolution: GrowthInterpolation
		:type kmin: float
		:type kmax: float
		:type mMin: float
		:type mMax: float
		:type physics: Physics
		:type AXIONS: bool
		:type jenkins_mass: bool
		"""

		if physics is None:
			self.__phys=Physics(True,True)
		else:
			self.__phys=physics

		self.__mMin=mMin
		self.__mMax=mMax
		self.__kmin=kmin
		self.__kmax=kmax

		self.__jenkins_mass=jenkins_mass
		self.__AXIONS=AXIONS

		self.__power_spectrum=power_spectrum

		if window_function == mean_pairwise_velocity.TOP_HAT:
			self.__window_function=mean_pairwise_velocity.top_hat_window
			self.__radius_of_mass=self.radius_of_mass_top_hat
		elif window_function == mean_pairwise_velocity.GAUSSIAN:
			self.__window_function = mean_pairwise_velocity.gaussian_window
			self.__radius_of_mass = self.radius_of_mass_gaussian
		elif window_function == mean_pairwise_velocity.SHARP_K:
			self.__window_function = mean_pairwise_velocity.sharp_k_window
			self.__radius_of_mass=self.radius_of_mass_sharp_k
		elif window_function == mean_pairwise_velocity.NO_FILTER:
			self.__window_function = mean_pairwise_velocity.no_window
			self.__radius_of_mass=self.radius_of_mass_top_hat
		else:
			print("This doesn't work!")
			print(window_function)
			raise Exception("Function Comparision fails!")


		if AXIONS:
			axion_G,axion_dlogG_dlogA=mode_evolution.get_growth(),mode_evolution.get_dlogG_dlogA()
			self.__growth,self.__G0,self.__dlogG_dlogA=lambda K, A: axion_G(A, K),lambda K: axion_G(1, K),lambda K,A:axion_dlogG_dlogA(A, K)
		else:
			dz_interp=self.__phys.get_D_interpolation(recompute=True, save=False)
			D0=self.__phys.D(0)

			cdm_growth=lambda k, a: dz_interp((1-a)/a)
			cdm_dlogG_dlogA=lambda K, A: differentiate(lambda log_a: cdm_growth(K, np.exp(log_a)), h=0.01)(np.log(A))/cdm_growth(K, A)
			cdm_G0=lambda k: D0

			self.__growth,self.__G0,self.__dlogG_dlogA=cdm_growth,cdm_G0,cdm_dlogG_dlogA

	def compute(self, z, do_unbiased=False, high_res=False, high_res_multi=2, diagnostic=False):
		"""

		:type z: float
		:type do_unbiased: bool
		:type high_res: bool
		:type high_res_multi: int
		"""
		# do_unbiased:	whether to compute unbiased spectra as well
		# high_res 	:	compute correlations functions with hiher resolution
		# high_res_multi:multiplier by which to increase the number of intenrgation points for the computation of correlations functions

		a=1/(1+z)

		masses=np.logspace(np.log10(self.__mMin)-0.5,np.log10(self.__mMax)+0.5,300)
		print("Computing variance of the mass distribution...")
		
		sigma_sq=np.vectorize(lambda r: mean_pairwise_velocity.sigma_mass_distribution_sq(r, a, self.__power_spectrum, self.__growth, self.__G0, kmin_log=np.log(self.__kmin), kmax_log=np.log(self.__kmax), window_function=self.__window_function))(self.__radius_of_mass(masses))
		sigma_sq_0=np.vectorize(lambda r: mean_pairwise_velocity.sigma_mass_distribution_sq(r, 1.0, self.__power_spectrum, self.__growth, self.__G0, kmin_log=np.log(self.__kmin), kmax_log=np.log(self.__kmax), window_function=self.__window_function))(self.__radius_of_mass(masses))

		sigma8=np.sqrt(mean_pairwise_velocity.sigma_mass_distribution_sq(8/self.__phys.h, 1, self.__power_spectrum, self.__growth, self.__G0, kmin_log=np.log(self.__kmin), kmax_log=np.log(self.__kmax), window_function=self.__window_function))
		print(sigma8)

		sigma_sq_log_interp=interpolate(np.log(masses), np.log(sigma_sq))
		sigma_sq_interp=lambda m: np.exp(sigma_sq_log_interp(np.log(m)))

		sigma_sq_0_log_interp=interpolate(np.log(masses), np.log(sigma_sq_0))
		sigma_sq_0_interp=lambda m: np.exp(sigma_sq_0_log_interp(np.log(m)))

		if self.__jenkins_mass:
			mass_function=self.jenkins_mass_function
		else:
			mass_function=self.press_schechter_mass_function

		k_vals=np.logspace(np.log10(self.__kmin), np.log10(self.__kmax), 300)
		print("Computing halo bias moments...")
		halo_bias_moments=[np.zeros(len(k_vals)),np.zeros(len(k_vals))]

		halo_bias_moments[0][:]=np.array(list(map(lambda k: self.mass_averaged_halo_bias(k, 1, lambda m: self.halo_bias(m, sigma_sq_interp, sigma_sq_0_interp), lambda m: mass_function(m, sigma_sq_interp), mMin=self.__mMin, mMax=self.__mMax), k_vals)))
		halo_bias_moments[1][:]=np.array(list(map(lambda k: self.mass_averaged_halo_bias(k, 2, lambda m: self.halo_bias(m, sigma_sq_interp, sigma_sq_0_interp), lambda m: mass_function(m, sigma_sq_interp), mMin=self.__mMin, mMax=self.__mMax), k_vals)))

		# computation may fail when numerator is nuermcially zero; use last successful value as limit
		halo_bias_moments[0][np.isnan(halo_bias_moments[0])]=np.nanmin(halo_bias_moments[0])
		halo_bias_moments[1][np.isnan(halo_bias_moments[1])]=np.nanmin(halo_bias_moments[1])
		halo_bias_1_interp = interp1d(k_vals, halo_bias_moments[0], fill_value=(halo_bias_moments[0][0], halo_bias_moments[0][-1]), bounds_error=False)
		halo_bias_2_interp = interp1d(k_vals, halo_bias_moments[1], fill_value=(halo_bias_moments[1][0], halo_bias_moments[1][-1]), bounds_error=False)
		
		print("Computing two-point correlation functions...")
		r_vals=np.linspace(1e-3, 300, 550)
		correlation_func_q2_vals=[]
		correlation_func_q1_vals=[]
		if do_unbiased:
			correlation_func_q1_vals_unbiased=[]
			correlation_func_q2_vals_unbiased=[]

		if high_res:
			res_multiplier=high_res_multi
		else:
			res_multiplier=1.0

		if self.__AXIONS:
			# axion growth function
			f = lambda k: self.__dlogG_dlogA(k, a)

			correlation_func_vals=np.array(list(map(lambda r: mean_pairwise_velocity.correlation_func(r, a, self.__power_spectrum, self.__growth, self.__G0, halo_bias_1_interp, halo_bias_2_interp, self.__kmin, self.__kmax, f, N=1000*res_multiplier), r_vals)))
			correlation_func_q1_vals,correlation_func_q2_vals=correlation_func_vals[:,0],correlation_func_vals[:,1]

			if do_unbiased:
				correlation_func_vals_unbiased=np.array(list(map(lambda r: mean_pairwise_velocity.correlation_func(r, a, self.__power_spectrum, self.__growth, self.__G0, lambda k: 1, lambda k: 1, self.__kmin, self.__kmax, f, N=1000*res_multiplier), r_vals)))
				correlation_func_q1_vals_unbiased,correlation_func_q2_vals_unbiased=correlation_func_vals_unbiased[:,0],correlation_func_vals_unbiased[:,1]
			
		else:
			correlation_func_vals=np.array(list(map(lambda r: mean_pairwise_velocity.correlation_func(r, a, self.__power_spectrum, self.__growth, self.__G0, halo_bias_1_interp, halo_bias_2_interp, self.__kmin, self.__kmax, N=1000*res_multiplier), r_vals)))
			correlation_func_q1_vals,correlation_func_q2_vals=correlation_func_vals[:,0],correlation_func_vals[:,1]
		
			if do_unbiased:
				correlation_func_vals_unbiased=np.array(list(map(lambda r: mean_pairwise_velocity.correlation_func(r, a, self.__power_spectrum, self.__growth, self.__G0, lambda k: 1, lambda k: 1, self.__kmin, self.__kmax, N=1000*res_multiplier), r_vals)))
				correlation_func_q1_vals_unbiased,correlation_func_q2_vals_unbiased=correlation_func_vals_unbiased[:,0],correlation_func_vals_unbiased[:,1]

		print("Computing volume averaged correlation function ...")
		correlation_func_q1_interp=interp1d(r_vals, correlation_func_q1_vals)
		correlation_func_bar_vals=[]
		if do_unbiased:
			correlation_func_q1_interp_unbiased=interp1d(r_vals, correlation_func_q1_vals_unbiased)
			correlation_func_bar_vals_unbiased=[]

		correlation_func_bar_vals=np.vectorize(lambda r: mean_pairwise_velocity.correlation_func_bar(r, correlation_func_q1_interp, rmin=min(r_vals)))(r_vals)
		if do_unbiased:
			correlation_func_bar_vals_unbiased=np.vectorize(lambda r: mean_pairwise_velocity.correlation_func_bar(r, correlation_func_q1_interp_unbiased, rmin=min(r_vals)))(r_vals)
			
		if self.__AXIONS:
			v_vals=2/3*100*self.__phys.h*self.__phys.E(z)*a*r_vals*np.array(correlation_func_bar_vals)/(1+np.array(correlation_func_q2_vals))
			if do_unbiased:
				v_vals_unbiased=2/3*100*self.__phys.h*self.__phys.E(z)*a*r_vals*np.array(correlation_func_bar_vals_unbiased)/(1+np.array(correlation_func_q2_vals_unbiased))
		else:
			v_vals=2/3*self.__dlogG_dlogA(0.0, a)*100*self.__phys.h*self.__phys.E(z)*a*r_vals*np.array(correlation_func_bar_vals)/(1+np.array(correlation_func_q2_vals))
			if do_unbiased:
				v_vals_unbiased=2/3*self.__dlogG_dlogA(0.0, a)*100*self.__phys.h*self.__phys.E(z)*a*r_vals*np.array(correlation_func_bar_vals_unbiased)/(1+np.array(correlation_func_q2_vals_unbiased))

		if diagnostic:
			return r_vals, v_vals, correlation_func_q1_vals, correlation_func_q2_vals, correlation_func_bar_vals, k_vals, halo_bias_moments, masses, sigma_sq, sigma_sq_0, mass_function(masses, sigma_sq_interp)

		if do_unbiased:
			return r_vals, v_vals, v_vals_unbiased
		else:
			return r_vals, v_vals

	@staticmethod
	def correlation_func(r, a, P, D, D0, bias_1, bias_2, kmin, kmax, f=lambda k: 1, N=500):

		step_N=N
		
		# simpsons rule log
		width=(np.log(kmax)-np.log(kmin))/step_N # compute width of the intervals
		eval_points_1=np.log(kmin)+np.arange(1,step_N,2)*width
		eval_points_2=np.log(kmin)+np.arange(2,step_N,2)*width
		eval_points_real_1=np.exp(eval_points_1)
		eval_points_real_2=np.exp(eval_points_2)

		integrand_log_gen=lambda k: k**2*np.sin(k*r)*P(k)*D(k,a)**2/D0(k)**2
		xi_1_multiplier=lambda k: f(k)*bias_1(k)

		xi_1=(integrand_log_gen(kmin)*xi_1_multiplier(kmin)+integrand_log_gen(kmax)*xi_1_multiplier(kmax)+4*sum(integrand_log_gen(eval_points_real_1)*xi_1_multiplier(eval_points_real_1))+2*sum(integrand_log_gen(eval_points_real_2)*xi_1_multiplier(eval_points_real_2)))*width/3
		xi_2=(integrand_log_gen(kmin)*bias_2(kmin)+integrand_log_gen(kmax)*bias_2(kmax)+4*sum(integrand_log_gen(eval_points_real_1)*bias_2(eval_points_real_1))+2*sum(integrand_log_gen(eval_points_real_2)*bias_2(eval_points_real_2)))*width/3
		
		return 1/(2*np.pi**2*r)*np.array([xi_1, xi_2])

	@staticmethod
	def correlation_func_bar(r, correlation_func_f, rmin=1e-3,N=2000):
		return 3/r**3*num.integrateS(lambda r: r**2*correlation_func_f(r), rmin, r, N)
	
	@staticmethod
	def top_hat_window(x):
		return np.piecewise(x, [np.fabs(x)<1e-4, np.fabs(x)>=1e-4], [1, lambda x: 3*(np.sin(x)-x*np.cos(x))/x**3])

	@staticmethod
	def gaussian_window(x):
		return np.exp(-x**2/2)

	@staticmethod
	def sharp_k_window(x):
		return np.piecewise(x, [np.fabs(x)<=1, np.fabs(x)>1], [1, 0])

	@staticmethod
	def no_window(x):
		return 1

	@staticmethod
	def sigma_mass_distribution_sq(R, a, P, G, G0, N=2000, kmin_log=np.log(1e-4), kmax_log=np.log(1e4), window_function=None):
		if window_function is None:
			window_function=mean_pairwise_velocity.gaussian_window
		elif window_function==mean_pairwise_velocity.sharp_k_window:
			kmax_log=min([kmax_log, np.log(1/R)])
			if kmin_log>=kmax_log:
				return 0.0
		elif window_function==mean_pairwise_velocity.gaussian_window or window_function==mean_pairwise_velocity.top_hat_window:
			pass
		elif window_function==mean_pairwise_velocity.no_window:
			window_function=mean_pairwise_velocity.top_hat_window
		else:
			raise Exception("There is a problem with the window function!")

		integrand_log = lambda log_k: np.exp(log_k) ** 3 * P(np.exp(log_k)) * G(np.exp(log_k), a) ** 2 / G0(np.exp(log_k)) ** 2 * window_function(np.exp(log_k) * R) ** 2
		return 1/(2*np.pi**2)*num.integrateS(integrand_log, kmin_log, kmax_log, N)

	def radius_of_mass_top_hat(self, M):
		return (3*M/(4*np.pi*self.__phys.rho0))**(1/3)

	def radius_of_mass_gaussian(self, M):
		return (2*np.pi)**(-1/2)*(M/self.__phys.rho0)**(1/3)

	def radius_of_mass_sharp_k(self, M):
		return (9*np.pi/2)**(-1/3)*self.radius_of_mass_top_hat(M)

	def mass_of_radius_top_hat(self, R):
		return 4/3*np.pi*R**3*self.__phys.rho0

	def mass_of_radius_gaussian(self, R):
		return np.sqrt(2*np.pi)*(4*np.pi/3)**(-1/3)*self.mass_of_radius_top_hat(R)

	def mass_of_radius_sharp_k(self, R):
		return (9*np.pi/2)*self.mass_of_radius_top_hat(R)

	def halo_bias(self, M, sigma_sq, sigma_sq_0):
		return 1+(self.__phys.delta_crit**2-sigma_sq_0(M))/(np.sqrt(sigma_sq(M))*np.sqrt(sigma_sq_0(M))*self.__phys.delta_crit)

	def jenkins_mass_function(self, M, sigma_sq):
		sigma=lambda m:np.sqrt(sigma_sq(m))

		dlogSigma_dlogM=differentiate(lambda log_m: np.log(sigma(np.exp(log_m))), h=0.01)(np.log(M))

		f=0.315*np.exp(-np.fabs(np.log(1/sigma(M))+0.61)**3.8)

		return self.__phys.rho0/M**2*f*np.fabs(dlogSigma_dlogM)

	def press_schechter_mass_function(self, M, sigma_sq):
		sigma=lambda m:np.sqrt(sigma_sq(m))
		
		dlogSigma_dlogM=differentiate(lambda log_m: np.log(sigma(np.exp(log_m))), h=0.01)(np.log(M))

		return np.sqrt(2/np.pi)*(self.__phys.rho0*self.__phys.delta_crit/sigma(M)/M**2)*np.fabs(dlogSigma_dlogM)*np.exp(-self.__phys.delta_crit**2/(2*sigma(M)**2))

	def mass_averaged_halo_bias(self, k, q, halo_bias_function, mass_function, mMin=1e10, mMax=1e15, N=2000):

		integrand_mass=lambda log_m: np.exp(log_m)**2*mass_function(np.exp(log_m))*self.__window_function(k*self.__radius_of_mass(np.exp(log_m)))**2
		integrand_bias=lambda log_m: np.exp(log_m)**2*mass_function(np.exp(log_m))*halo_bias_function(np.exp(log_m))**q*self.__window_function(k*self.__radius_of_mass(np.exp(log_m)))**2

		if self.__window_function==mean_pairwise_velocity.sharp_k_window:
			mMax=min([mMax, self.mass_of_radius_sharp_k(1/k)])
			if mMax<=mMin:
				return 0.0
		elif self.__window_function==mean_pairwise_velocity.gaussian_window or self.__window_function==mean_pairwise_velocity.top_hat_window or self.__window_function==mean_pairwise_velocity.no_window:
			pass
		else:
			raise Exception("There is a problem with the window function!")

		average_mass=num.integrateS(integrand_mass, np.log(mMin), np.log(mMax), N)
		average_halo_bias=num.integrateS(integrand_bias, np.log(mMin), np.log(mMax), N)
		
		try:
			return average_halo_bias/average_mass
		except ZeroDivisionError:
			return np.nan
