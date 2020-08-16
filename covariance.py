from numerics import interpolate, differentiate, num
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
from helpers import *
import dill
from physics import Physics
from mean_pairwise_velocity import mean_pairwise_velocity


class covariance_matrix:

    def __init__(self, power_spectrum, mode_evolution, zmin, zmax, Nz, r_vals, deltaR, f_sky, sigma_v_vals, kmin=1.0e-3, kmax=1.0e4, mMin=1e12, mMax=1e15, physics=None, jenkins_mass=False, window_function=mean_pairwise_velocity.GAUSSIAN):

        if physics is None:
            self.__phys = Physics(True, True)
        else:
            self.__phys = physics

        self.__zmin = zmin
        self.__zmax = zmax
        self.__mMin = mMin
        self.__mMax = mMax
        self.__kmin = kmin
        self.__kmax = kmax
        self.__f_sky = f_sky
        self.__sigma_v_vals=sigma_v_vals
        self.__deltaR=deltaR
        self.vs=[]
        self.correlation_f=[]

        self.__r_vals = r_vals
        self.__z_vals = np.linspace(zmin, zmax, Nz + 1)

        self.__window_function=None
        self.__radius_of_mass=None

        if jenkins_mass:
            self.__mass_function = self.jenkins_mass_function
        else:
            self.__mass_function = self.press_schechter_mass_function

        self.__AXIONS = False

        self.__power_spectrum = power_spectrum

        if self.__AXIONS:
            axion_G, axion_dlogG_dlogA = mode_evolution.get_growth(), mode_evolution.get_dlogG_dlogA()
            self.__growth, self.__G0, self.__dlogG_dlogA = lambda K, A: axion_G(A, K), lambda K: axion_G(1, K), lambda K, A: axion_dlogG_dlogA(A, K)
        else:
            dz_interp = self.__phys.get_D_interpolation(recompute=True, save=False)
            D0 = self.__phys.D(0)

            cdm_growth = lambda k, a: dz_interp((1 - a) / a)
            cdm_dlogG_dlogA = lambda K, A: differentiate(lambda log_a: cdm_growth(K, np.exp(log_a)), h=0.01)(np.log(A)) / cdm_growth(K, A)
            cdm_G0 = lambda k: D0

            self.__growth, self.__G0, self.__dlogG_dlogA = cdm_growth, cdm_G0, cdm_dlogG_dlogA
            self.__f=lambda a: self.__dlogG_dlogA(0,a)

        R1, R2 = np.meshgrid(self.__r_vals, self.__r_vals)
        self.__r_pairs = [(R1.flatten()[i], R2.flatten()[i]) for i in range(0, len(self.__r_vals) ** 2)]

        output = []
        with MyProcessPool(min([6,Nz])) as pool:
            output = list(pool.imap(lambda i: self.evaluate_covariance(i, window_function), np.arange(0, Nz, 1)))

        self.__covariance = np.empty((Nz, len(r_vals), len(r_vals)))
        self.__inv_covariance_interpolations=np.empty((Nz), dtype='object')

        for i in range(Nz):
            self.__covariance[i, :, :] = np.reshape(output[i][0], (len(r_vals), len(r_vals)))
            self.__inv_covariance_interpolations[i] = RectBivariateSpline(r_vals, r_vals, np.linalg.inv(self.__covariance[i]), bbox=[min(r_vals), max(r_vals), min(r_vals), max(r_vals)])
            self.vs.append(output[i][1])
            self.correlation_f.append(output[i][2])

    def save_covariance(self, path):
        with open(path, "wb") as f:
            dill.dump(self, f)

    @staticmethod
    def load_covariance(path):
        cov=None
        with open(path, "rb") as f:
            cov=dill.load(f)
        return cov

    def evaluate_covariance(self, z_bin, window_function):

        if window_function == mean_pairwise_velocity.TOP_HAT:
            self.__window_function = covariance_matrix.top_hat_window
            self.__radius_of_mass = self.radius_of_mass_top_hat
        elif window_function == mean_pairwise_velocity.GAUSSIAN:
            self.__window_function = covariance_matrix.gaussian_window
            self.__radius_of_mass = self.radius_of_mass_gaussian
        elif window_function == mean_pairwise_velocity.SHARP_K:
            self.__window_function = covariance_matrix.sharp_k_window
            self.__radius_of_mass = self.radius_of_mass_sharp_k
        elif window_function == mean_pairwise_velocity.NO_FILTER:
            self.__window_function = covariance_matrix.no_window
            self.__radius_of_mass = self.radius_of_mass_top_hat
        else:
            print("This doesn't work!")
            print(window_function)
            raise Exception("Function Comparision fails!")

        zmin, zmax = self.__z_vals[z_bin], self.__z_vals[z_bin + 1]
        print(zmin, zmax)
        z_central = (zmax - zmin) / 2
        a_central=1/(1+z_central)
        sigma_sq_interp, sigma_sq_0_interp, halo_bias_1_interp, halo_bias_2_interp, correlation_func_q1_interp, correlation_func_q2_interp, v_interp = self.pre_compute(z_central)

        volume = self.survey_volume(zmin,zmax)
        number_density = self.number_density(sigma_sq_interp)
        covariances = []

        for r in self.__r_pairs:
            covariances.append(self.compute(z_bin, r[0], r[1], self.__deltaR, correlation_func_q2_interp, halo_bias_1_interp, number_density, volume))

        #self.vs = v_interp
        return covariances, v_interp, correlation_func_q2_interp

    def pre_compute(self, z, high_res=False, high_res_multi=2):
        a = 1 / (1 + z)

        masses = np.logspace(1, 19, 300)
        print("Computing variance of the mass distribution...")

        sigma_sq = np.vectorize(lambda r: self.sigma_mass_distribution_sq(r, a, self.__power_spectrum, self.__growth, self.__G0, kmin_log=np.log(self.__kmin), kmax_log=np.log(self.__kmax), window_function=self.__window_function))(self.__radius_of_mass(masses))
        sigma_sq_0 = np.vectorize(lambda r: self.sigma_mass_distribution_sq(r, 1.0, self.__power_spectrum, self.__growth, self.__G0, kmin_log=np.log(self.__kmin), kmax_log=np.log(self.__kmax), window_function=self.__window_function))(self.__radius_of_mass(masses))

        sigma8 = np.sqrt(self.sigma_mass_distribution_sq(8 / self.__phys.h, 1, self.__power_spectrum, self.__growth, self.__G0, kmin_log=np.log(self.__kmin), kmax_log=np.log(self.__kmax), window_function=self.__window_function))
        print(sigma8)

        sigma_sq_log_interp = interpolate(np.log(masses), np.log(sigma_sq))
        sigma_sq_interp = lambda m: np.exp(sigma_sq_log_interp(np.log(m)))

        sigma_sq_0_log_interp = interpolate(np.log(masses), np.log(sigma_sq_0))
        sigma_sq_0_interp = lambda m: np.exp(sigma_sq_0_log_interp(np.log(m)))

        k_vals = np.logspace(np.log10(self.__kmin), np.log10(self.__kmax), 200)

        """ plt.figure()
        m_vals=np.logspace(12, 15, 100)
        plt.loglog(m_vals, np.array(list(map(lambda m: self.__mass_function(m, sigma_sq_interp), m_vals))*m_vals))
        plt.loglog(m_vals, np.array(list(map(lambda m: self.jenkins_mass_function(m, sigma_sq_interp), m_vals))*m_vals))
        plt.show()
        exit()"""

        print("Computing halo bias moments...")
        halo_bias_moments = [np.zeros(len(k_vals)), np.zeros(len(k_vals))]

        halo_bias_moments[0] = np.array(list(map(lambda k: self.mass_averaged_halo_bias(k, 1, lambda m: self.halo_bias(m, sigma_sq_interp, sigma_sq_0_interp), lambda m: self.__mass_function(m, sigma_sq_interp), mMin=self.__mMin, mMax=self.__mMax), k_vals)))
        halo_bias_moments[1] = np.array(list(map(lambda k: self.mass_averaged_halo_bias(k, 2, lambda m: self.halo_bias(m, sigma_sq_interp, sigma_sq_0_interp), lambda m: self.__mass_function(m, sigma_sq_interp), mMin=self.__mMin, mMax=self.__mMax), k_vals)))

        # computation may fail when numerator is numercially zero; use last successful value as limit
        halo_bias_moments[0][np.isnan(halo_bias_moments[0])] = np.nanmin(halo_bias_moments[0])
        halo_bias_moments[1][np.isnan(halo_bias_moments[1])] = np.nanmin(halo_bias_moments[1])
        halo_bias_1_interp = interp1d(k_vals, halo_bias_moments[0], fill_value=(halo_bias_moments[0][0], halo_bias_moments[0][-1]), bounds_error=False)
        halo_bias_2_interp = interp1d(k_vals, halo_bias_moments[1], fill_value=(halo_bias_moments[1][0], halo_bias_moments[1][-1]), bounds_error=False)
        print(halo_bias_1_interp(self.__kmax), halo_bias_1_interp(self.__kmin))

        print("Computing two-point correlation functions...")
        r_vals = np.linspace(1e-3, 300, 400)
        correlation_func_q2_vals = []
        correlation_func_q1_vals = []

        if high_res:
            res_multiplier = high_res_multi
        else:
            res_multiplier = 1.0

        if self.__AXIONS:
            correlation_func_vals = np.array(list(map(lambda r: self.correlation_func(r, a, self.__power_spectrum, self.__growth, self.__G0, halo_bias_1_interp, halo_bias_2_interp, min(k_vals), max(k_vals),  self.__f, N=500 * res_multiplier), r_vals)))
            correlation_func_q1_vals, correlation_func_q2_vals = correlation_func_vals[:, 0], correlation_func_vals[:, 1]
        else:
            correlation_func_vals = np.array(list(map(lambda r: self.correlation_func(r, a, self.__power_spectrum, self.__growth, self.__G0, halo_bias_1_interp, halo_bias_2_interp, min(k_vals), max(k_vals), N=500 * res_multiplier), r_vals)))
            correlation_func_q1_vals, correlation_func_q2_vals = correlation_func_vals[:, 0], correlation_func_vals[:, 1]

        print("Computing volume averaged correlation function ...")
        correlation_func_q1_interp = interp1d(r_vals, correlation_func_q1_vals)
        correlation_func_q2_interp = interp1d(r_vals, correlation_func_q2_vals)
        correlation_func_bar_vals = []

        correlation_func_bar_vals = np.vectorize(lambda r: self.correlation_func_bar(r, correlation_func_q1_interp, rmin=min(r_vals)))(r_vals)

        if self.__AXIONS:
            v_vals = 2 / 3 * 100 * self.__phys.h * self.__phys.E(z) * a * r_vals * np.array(correlation_func_bar_vals) / (1 + np.array(correlation_func_q2_vals))
        else:
            v_vals = 2 / 3 * self.__f(a) * 100 * self.__phys.h * self.__phys.E(z) * a * r_vals * np.array(correlation_func_bar_vals) / (1 + np.array(correlation_func_q2_vals))

        v_interp = interp1d(r_vals, v_vals)

        return sigma_sq_interp, sigma_sq_0_interp, halo_bias_1_interp, halo_bias_2_interp, correlation_func_q1_interp, correlation_func_q2_interp, v_interp

    def compute(self, z_bin, r1, r2, deltaR, correlation_func_q2_interp, halo_bias_1_interp, number_density, volume, N=2000):
        if self.__AXIONS:
            raise Exception("Not implemented!")
        else:
            zmin, zmax = self.__z_vals[z_bin], self.__z_vals[z_bin + 1]
            z_central = (zmax - zmin) / 2
            a_central = 1 / (1 + z_central)

            integrand_log = lambda log_k: np.exp(log_k) * (self.__growth(0, a_central)**2/self.__G0(0)**2 * self.__power_spectrum(np.exp(log_k)) * halo_bias_1_interp(np.exp(log_k)) + 1/number_density)**2 * covariance_matrix.binned_window_function(np.exp(log_k), r1, r1+deltaR)*covariance_matrix.binned_window_function(np.exp(log_k), r2, r2+deltaR)
            #integrand_log = lambda log_k: np.exp(log_k) * (1/number_density)**2 * covariance_matrix.binned_window_function(np.exp(log_k), r1, r1+deltaR)*covariance_matrix.binned_window_function(np.exp(log_k), r2, r2+deltaR)

            integral = num.integrateS(integrand_log, np.log(self.__kmin), np.log(self.__kmax), N)

            if r1==r2:
                n_pairs=self.number_of_pairs(r1, deltaR, number_density, volume, correlation_func_q2_interp(r1))

                return 4 / (np.pi ** 2 * volume) * (100 * self.__phys.h * self.__phys.E(1/a_central-1) * a_central) ** 2 / (1 + correlation_func_q2_interp(r1)) / (1 + correlation_func_q2_interp(r2)) * self.__f(a_central) ** 2 * integral + 2*self.__sigma_v_vals[z_bin]**2/n_pairs
            else:
                return 4 / (np.pi**2 * volume) * (100 * self.__phys.h * self.__phys.E(1/a_central-1) * a_central)**2 / (1 + correlation_func_q2_interp(r1)) / (1 + correlation_func_q2_interp(r2)) * self.__f(a_central) ** 2 * integral

    def get_inverted_covariance_interpolation(self, z_bin, r1, r2):
        return np.squeeze(self.__inv_covariance_interpolations[z_bin].ev(r1, r2))

    def get_covariance(self, z_bin):
        return self.__covariance[z_bin]

    def get_inverted_covariance(self, z_bin):
        return np.linalg.inv(self.__covariance[z_bin])

    def survey_volume(self, z_1, z_2):
        w = lambda z0: 3000.0 / self.__phys.h * num.integrate(lambda z: 1 / (self.__phys.E(z)), 0, z0)
        return 4/3 * np.pi * self.__f_sky * (w(z_2) ** 3 - w(z_1) ** 3)

    def number_density(self, sigma_sq):
        mass_f = lambda m: self.__mass_function(m, sigma_sq)
        return num.integrate(mass_f, self.__mMin, self.__mMax)

    @staticmethod
    def number_of_pairs(r, deltaR, number_density, survey_volume, correlation):
        bin_volume=4/3*np.pi*((r+deltaR)**3-r**3)

        return number_density**2*survey_volume*bin_volume*(1+correlation)/2

    @staticmethod
    def binned_window_function(k, rmin, rmax):
        w_tilde=lambda x: (2*np.cos(x)+x*np.sin(x))/x**3

        return 3*(rmin**3*w_tilde(k*rmin)-rmax**3*w_tilde(k*rmax))/(rmax**3-rmin**3)

    @staticmethod
    def correlation_func(r, a, P, D, D0, bias_1, bias_2, kmin, kmax, f=lambda k: 1, N=500):

        step_N = int(np.clip((r / 50) + 1, 1, 7) ** 2 * N)

        # simpsons rule log
        width = (np.log(kmax) - np.log(kmin)) / step_N  # compute width of the intervals
        eval_points_1 = np.log(kmin) + np.arange(1, step_N, 2) * width
        eval_points_2 = np.log(kmin) + np.arange(2, step_N, 2) * width
        eval_points_real_1 = np.exp(eval_points_1)
        eval_points_real_2 = np.exp(eval_points_2)

        integrand_log_gen = lambda k: k ** 2 * np.sin(k * r) * P(k) * D(k, a) ** 2 / D0(k) ** 2
        # print(type(f), type(bias_1))
        xi_1_multiplier = lambda k: f(k) * bias_1(k)

        xi_1 = (integrand_log_gen(kmin) * xi_1_multiplier(kmin) + integrand_log_gen(kmax) * xi_1_multiplier(kmax) + 4 * sum(integrand_log_gen(eval_points_real_1) * xi_1_multiplier(eval_points_real_1)) + 2 * sum(integrand_log_gen(eval_points_real_2) * xi_1_multiplier(eval_points_real_2))) * width / 3
        xi_2 = (integrand_log_gen(kmin) * bias_2(kmin) + integrand_log_gen(kmax) * bias_2(kmax) + 4 * sum(integrand_log_gen(eval_points_real_1) * bias_2(eval_points_real_1)) + 2 * sum(integrand_log_gen(eval_points_real_2) * bias_2(eval_points_real_2))) * width / 3

        return 1 / (2 * np.pi ** 2 * r) * np.array([xi_1, xi_2])

    @staticmethod
    def correlation_func_bar(r, correlation_func_f, rmin=1e-3):
        return 3 / r ** 3 * num.integrate(lambda R: R ** 2 * correlation_func_f(R), rmin, r)

    @staticmethod
    def top_hat_window(x):
        return np.piecewise(x, [np.fabs(x) < 1e-4, np.fabs(x) >= 1e-4], [1, lambda x: 3 * (np.sin(x) - x * np.cos(x)) / x ** 3])

    @staticmethod
    def gaussian_window(x):
        return np.exp(-x ** 2 / 2)

    @staticmethod
    def sharp_k_window(x):
        return np.piecewise(x, [np.fabs(x) <= 1, np.fabs(x) > 1], [1, 0])

    @staticmethod
    def no_window(x):
        return 1

    def radius_of_mass_top_hat(self, M):
        return (3 * M / (4 * np.pi * self.__phys.rho0)) ** (1 / 3)

    def radius_of_mass_gaussian(self, M):
        return (2 * np.pi) ** (-1 / 2) * (M / self.__phys.rho0) ** (1 / 3)

    def radius_of_mass_sharp_k(self, M):
        return (9 * np.pi / 2) ** (-1 / 3) * self.radius_of_mass_top_hat(M)

    def mass_of_radius_top_hat(self, R):
        return 4 / 3 * np.pi * R ** 3 * self.__phys.rho0

    def mass_of_radius_gaussian(self, R):
        return np.sqrt(2 * np.pi) * (4 * np.pi / 3) ** (-1 / 3) * self.mass_of_radius_top_hat(R)

    def mass_of_radius_sharp_k(self, R):
        return (9 * np.pi / 2) * self.mass_of_radius_top_hat(R)

    @staticmethod
    def sigma_mass_distribution_sq(R, a, P, G, G0, N=2000, kmin_log=np.log(1e-4), kmax_log=np.log(1e4), window_function=None):
        if window_function is None:
            window_function = covariance_matrix.gaussian_window
        elif window_function == covariance_matrix.sharp_k_window:
            kmax_log = min([kmax_log, np.log(1 / R)])
            if kmin_log >= kmax_log:
                return 0.0
        elif window_function == covariance_matrix.gaussian_window or window_function == covariance_matrix.top_hat_window:
            pass
        elif window_function == covariance_matrix.no_window:
            window_function = covariance_matrix.top_hat_window
        else:
            raise Exception("There is a problem with the window function!")

        integrand_log = lambda log_k: np.exp(log_k) ** 3 * P(np.exp(log_k)) * G(np.exp(log_k), a) ** 2 / G0(np.exp(log_k)) ** 2 * window_function(np.exp(log_k) * R) ** 2

        return 1 / (2 * np.pi ** 2) * num.integrateS(integrand_log, kmin_log, kmax_log, N)

    def halo_bias(self, M, sigma_sq, sigma_sq_0):
        return 1 + (self.__phys.delta_crit ** 2 - sigma_sq_0(M)) / (np.sqrt(sigma_sq(M)) * np.sqrt(sigma_sq_0(M)) * self.__phys.delta_crit)

    def jenkins_mass_function(self, M, sigma_sq):
        sigma = lambda m: np.sqrt(sigma_sq(m))

        dlogSigma_dlogM = differentiate(lambda log_m: np.log(sigma(np.exp(log_m))), h=0.01)(np.log(M))

        f = 0.315 * np.exp(-np.fabs(np.log(1 / sigma(M)) + 0.61) ** 3.8)

        return self.__phys.rho0 / M ** 2 * f * np.fabs(dlogSigma_dlogM)

    def press_schechter_mass_function(self, M, sigma_sq):
        sigma = lambda m: np.sqrt(sigma_sq(m))

        dlogSigma_dlogM = differentiate(lambda log_m: np.log(sigma(np.exp(log_m))), h=0.01)(np.log(M))

        return np.sqrt(2 / np.pi) * (self.__phys.rho0 * self.__phys.delta_crit / sigma(M) / M ** 2) * np.fabs(dlogSigma_dlogM) * np.exp(-self.__phys.delta_crit ** 2 / (2 * sigma(M) ** 2))

    def mass_averaged_halo_bias(self, k, q, halo_bias_function, mass_function, mMin=1e10, mMax=1e15):

        integrand_mass = lambda log_m: np.exp(log_m) ** 2 * mass_function(np.exp(log_m)) * self.__window_function(k * self.__radius_of_mass(np.exp(log_m))) ** 2
        integrand_bias = lambda log_m: np.exp(log_m) ** 2 * mass_function(np.exp(log_m)) * halo_bias_function(np.exp(log_m)) ** q * self.__window_function(k * self.__radius_of_mass(np.exp(log_m))) ** 2

        if self.__window_function == covariance_matrix.sharp_k_window:
            mMax = min([mMax, self.mass_of_radius_sharp_k(1 / k)])
            if mMax <= mMin:
                return 0.0
        elif self.__window_function == covariance_matrix.gaussian_window or self.__window_function == covariance_matrix.top_hat_window or self.__window_function==covariance_matrix.no_window:
            pass
        else:
            raise Exception("There is a problem with the window function!")

        average_mass = num.integrate(integrand_mass, np.log(mMin), np.log(mMax))
        average_halo_bias = num.integrate(integrand_bias, np.log(mMin), np.log(mMax))

        try:
            return average_halo_bias / average_mass
        except Exception as ex:
            return np.nan

"""
import matplotlib.pyplot as plt

print("test")
filename = "/Users/gerrit/SynologyDrive/College/Research/kSZ/source_parameterConstraints/axion_frac=0.000_matterpower_out.dat"  # askopenfilename() # show an "Open" dialog box and return the path to the selected file
# print(filename)

phys = Physics(True, True)

P_CAMB = np.loadtxt(filename)
p_interp = interpolate(P_CAMB[:, 0] * phys.h, P_CAMB[:, 1] / phys.h ** 3, phys.P_cdm, phys.P_cdm)
r_vals = np.arange(20, 180, 2)
cov = covariance_matrix(p_interp, None, 0.1, 0.6, 5, r_vals, 2, 6000/129600*np.pi, 120, kmin=min(P_CAMB[:, 0] * phys.h), kmax=max(P_CAMB[:, 0] * phys.h), mMin=1e14, mMax=1e16, physics=phys)
plt.figure()
plt.imshow(cov.get_covariance(0), origin="lower", extent=(20, 200, 20, 200))
plt.colorbar()

plt.figure()
plt.plot(r_vals, np.sqrt(cov.get_covariance(0).diagonal()) / cov.vs(r_vals))

plt.figure()
plt.plot(r_vals, cov.vs(r_vals))
plt.show()

"""