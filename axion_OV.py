import numpy as np
from numpy import pi
from scipy.interpolate import pchip, RectBivariateSpline
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import spherical_jn as bessel_j
from scipy.optimize import newton_krylov 

from helpers import *
from numerics import *
from physics import Physics
from read_mode_evolution import GrowthInterpolation
import camb


def COMPUTE_AXION_KSZ(phys, modeEvolution, axion_power_spectrum_file, folder="", MODEL=["AXION_CAMB", "CDM_CAMB"]):
	###SET COMPUTATION OPTIONS
	SINGULARITY=True 		#use substitution to deal with singularity in S(k) integrand
	GAUSS=True 				#use Gauss-Legendre Quadrature for 2D integration (False: use Riemann integration)
	COMPUTE_VISH=True 		#compute the vishniac spectrum (False: load from file)
	READ_AXION_GROWTH_NEW=True #solve pertrubation equation numerically for axion time evolution
	COMPUTE_W=True 		#compute the comoving distacne as function of redshift (False: load from file)
	COMPUTE_D=True 		#compute the growth factor as function of redshift (False: load from file)
	COMPUTE_T=True			#compute the time as function of redshift (False: load from file)
	COMPUTE_PROJECTION=True #compute projection of vishniac spectrum on the line of sight

	PRINT_PARAMS=False

	PLOT_MASS_POWER=False 	#plot mass power spectrum
	PLOT_AXION_GROWTH=False 	#plot axion growth function
	PLOT_VISIBILITY_FUNCTION=False
	NUM_PLOT_VISH=3
	PLOT_APPROXIMATION=(COMPUTE_PROJECTION and False) #plot approximate projection (without integration) (only available when projection is being plotted)

	num=IntegrationNumerics(GAUSS, gaussPoints=300)


	aa=(46.9*phys.Omega0*phys.h**2)**0.67*(1.0+(32.1*phys.Omega0*phys.h**2)**(-0.532))
	ab=(12.0*phys.Omega0*phys.h**2)**0.424*(1.0+(45.*phys.Omega0*phys.h**2)**(-0.582))
	al=aa**(-phys.Omega_Bh2/phys.h**2/phys.Omega0)*ab**((phys.Omega_Bh2/phys.h**2/phys.Omega0)**3)
	correctionQ=1/np.sqrt(al)*1.0104**2/np.sqrt(1.0-phys.Omega_Bh2/phys.h**2/phys.Omega0)

	#primary mass power spectrum (Cold Dark Matter) (Bardeen, Bond, Kaiser, Szalay; 1986)
	def P_cdm(kp):
		return ((kp/phys.H0)**phys.n)*(trans(kp/(phys.Omega0*phys.h**2)*correctionQ))**2

	#mass power spectrum fuzzy dark matter (Hu, Barakana, Gruzinov; 2000)
	def P_fcdm(kp):
		kj_eq=9*(phys.m_axion/1e-22)**0.5
		x=1.61*(phys.m_axion/1e-22)**(1/18)*kp/kj_eq

		return (np.cos(x**3)/(1+x**8))**2*P_cdm(kp)

	#bbks transfer function as a function of q=k/Omega0*h^2
	def trans(q):
		return np.log(1+2.34*q)/(2.34*q)/((1+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**(1/4))

	def sigma8(P, kmin=None, kmax=None):
		R=8/phys.h
		integrand=lambda k: k**2*P(k)*(3*bessel_j(1,k*R)/(k*R))**2

		if kmin is not None and kmax is not None:
			return 1/(2*np.pi**2)*num.integrate(integrand, kmin, kmax)
		else:
			if kmin is None:
				xmin=0
			else:
				xmin=kmin/(1+kmin)

			if kmax is None:
				xmax=1
			else:
				xmax=kmax/(kmax+1)

			integrand_new=lambda x:integrand(x/(1-x))*1/(1-x)**2
			return 1/(2*np.pi**2)*num.integrate(integrand_new, xmin, xmax)

	#vishniac power spectrum
	def vish_Spec(k, P=None):
		def x_integrand(k, x, y, P=None):
			factorA=1+y**2-2*x*y
			factorB=(1-x**2)*(1-2*x*y)**2

			return P(k*y)*P(k*np.sqrt(factorA))*factorB/factorA**2
		


		integrand=lambda t,u: 10**u*2/phys.n*t**(2/phys.n-1)*x_integrand(k, 1.0-t**(2/phys.n), 10**u, P)
		integrand2=lambda x,u: 10**u*x_integrand(k, x, 10**u, P)

		lk=np.log10(k)
		if SINGULARITY:
			return k*np.log(10)*(num.integrate2D(integrand, 0, 2**(phys.n/2), min([-7-lk, -1]), 0)+num.integrate2D(integrand, 0, 2**(phys.n/2), 0, max([3-lk, 1])))
		else:
			return k*np.log(10)*(num.integrate2D(integrand2, -1, 1, min([-7-lk, -1]), 0)+num.integrate2D(integrand2, -1, 1, 0,max([3-lk, 1])))
	def S_integrand(k, X_T, U, z, P, G, dG):
		def x_integrand(k, x, y, z):
			factorA=(1-x**2)
			factorB=1+y**2-2*x*y
			sqrt_factorB=np.sqrt(factorB)
			a=1/(1+z)
			power=P(k*y)*P(k*sqrt_factorB)
			
			numerator=factorA*(dG(a,k*sqrt_factorB)*G(a, k*y)*y**2-dG(a,k*y)*G(a,k*sqrt_factorB)*factorB)**2
			#numerator=factorA*(y**2-factorB)**2
			denominator=factorB**2

			return power*numerator/denominator

		integrand=lambda t,u: 10**u*2/phys.n*t**(2/phys.n-1)*x_integrand(k, 1.0-t**(2/phys.n), 10**u, z)
		integrand2=lambda x,u: 10**u*x_integrand(k, x, 10**u, z)

		#return x_integrand(k, X_T, U, z)

		if SINGULARITY:
			return np.log(10)*integrand(X_T,U)
		else:
			return np.log(10)*integrand2(X_T,U)

	def vish_axions(k, z, P, G, dG):
		lk=np.log10(k)
		if SINGULARITY:
			return k*(num.integrate2D(lambda t, u: S_integrand(k, t, u, z, P, G, dG), 0, 2**(phys.n/2), min([-7-lk, -1]), 0)+num.integrate2D(lambda t, u: S_integrand(k, t, u, z, P, G, dG), 0, 2**(phys.n/2), 0, max([3-lk, 1])))
		else:
			return k*(num.integrate2D(lambda x, u: S_integrand(k, x, u, z, P, G, dG), -1, 1, min([-7-lk, -1]), 0)+num.integrate2D(lambda x, u: S_integrand(k, t, u, z, P, G, dG), -1, 1, 0,max([3-lk, 1])))

	def C_proj_axions(ell, g, S, w, z_of_w, zmin=0, zmax=20):
		#print("computing ell={}".format(ell))
		dw_dz=lambda z: 1/(phys.a0*phys.H0*phys.E(z))
		integrand_a=lambda a:g((phys.a0-a)/a)**2/w((phys.a0-a)/a)**2*S(a, (ell+1/2)/w((1-a)/a))*dw_dz((phys.a0-a)/a)
		integrand_loga=lambda loga:10**loga*np.log(10)*integrand_a(10**loga)
		#integrand_z=lambda z:g(z)**2/w(z)**2*a(z)**2*S(1/(1+z), (ell+1/2)/w(z))*dw_dz(z)

		return 1/(16*np.pi**2)*num.integrate(integrand_loga, np.log10(phys.a0/(1+zmax)), np.log10(phys.a0/(1+zmin)))



	#COBE normalization (Note: need to change to planck)
	def deltaH():
		return (1e-5)*(2.422-1.166*np.exp(Omega0)+0.800*np.exp(OmegaLambda)+3.780*Omega0-2.267*Omega0*np.exp(OmegaLambda)+0.487*Omega0**2+0.561*OmegaLambda+3.392*OmegaLambda*np.exp(Omega0)-8.568*Omega0*OmegaLambda+1.080*OmegaLambda**2)

	def C_proj(ell, g, D, dD, D0, S, w, z_of_w, zmin=0, zmax=20):
		#print("computing ell={}".format(ell))
		dw_dz=lambda z: 1/(phys.a0*phys.H0*phys.E(z))
		integrand_a=lambda a:g((phys.a0-a)/a)**2/w((phys.a0-a)/a)**2*(dD((phys.a0-a)/a)*D((phys.a0-a)/a)/D0**2)**2*S((ell+1/2)/w((phys.a0-a)/a))*dw_dz((phys.a0-a)/a)
		integrand_loga=lambda loga:10**loga*np.log(10)*integrand_a(10**loga)

		return 1/(16*np.pi**2)*num.integrate(integrand_loga, np.log10(phys.a0/(1+zmax)), np.log10(phys.a0/(1+zmin)))

	def rmsT(cls, min, max):
		return np.sqrt(num.integrate(cls, min, max)/(2*np.pi))



	axion_G_interp,axion_dG_interp=modeEvolution.get_growth()

	P_CDM=lambda k: phys.sig8**2/sigma8(P_cdm)*P_cdm(k)
	P_FCDM=lambda k: phys.sig8**2/sigma8(P_fcdm)*P_fcdm(k)

	P_AxionCAMB=np.loadtxt(axion_power_spectrum_file)
	P_CAMB=np.loadtxt("../CAMB-0.1.6.1/test_matterpower.dat")

	p_AxionCAMB_interp=interpolate(P_AxionCAMB[:,0]*phys.h, P_AxionCAMB[:,1], P_CDM, P_CDM)#
	p_CAMB_interp=interpolate(P_CAMB[:,0]*phys.h, P_CAMB[:,1], P_CDM, P_CDM)

	a_vals=np.logspace(-5, 0, 3000)
	z_vals=(1-a_vals)/a_vals
	dz_interp=phys.get_D_interpolation(recompute=COMPUTE_D)

	tz_interp,zt_interp=phys.get_t_interpolation(recompute=COMPUTE_T, doInverse=True)

	wz_interp,zw_interp=phys.get_w_interpolation(recompute=COMPUTE_W, doInverse=True)

	def get_MassPower(model):

		if model=="CDM":
			return lambda k: P_CDM(k)
		elif model=="FCDM":
			return P_FCDM
		elif model=="CDM_CAMB":
			return lambda k: p_CAMB_interp(k)
		elif model=="AXION_CAMB":
			return lambda k: p_AxionCAMB_interp(k)
		elif model=="AXION_CAMB_simple":
			return lambda k: p_AxionCAMB_interp(k)

	for m in MODEL:
		print("{}: sigma8={}".format(m, np.sqrt(sigma8(get_MassPower(m)))))

	if PLOT_MASS_POWER:
		plt.figure()
		k_vals=np.logspace(-5,4,1000)
		for m in MODEL:
			plt.loglog(k_vals/phys.h, get_MassPower(m)(k_vals), label=m)
		plt.legend()
		plt.xlabel(r"$k_{ph} h^{-1} \left[Mpc^{-1}\right]$")
		plt.ylabel(r"$P(k)$")
		plt.title("Mass Power Spectrum")
	"""
	plt.figure()
	k=1
	lk=np.log10(k)
	x_vals=np.linspace(-1, 1, 1000)
	y_vals=np.logspace(-6-np.log10(k), 3-np.log10(k), 1000)
	t_vals=np.linspace(0, 2**(phys.n/2), 1000)
	u_vals=np.linspace(-7-np.log10(k), 3-np.log10(k), 1000)
	print('plotting integrand')
	tGrid,uGrid=np.meshgrid(t_vals,u_vals)
	xGrid,yGrid=np.meshgrid(x_vals, y_vals)
	plt.imshow(np.log10(S_integrand(k, tGrid, uGrid, phys.z_r, get_MassPower(m), axion_G_interp, axion_dG_interp)), aspect='auto')
	plt.colorbar()
	plt.xlabel("t")
	plt.ylabel("u")

	plt.show()
	"""

	g_func_gauss=lambda z:phys.g(wz_interp(z))
	dw_dz=lambda z: 1/(phys.a0*phys.H0*phys.E(z))
	z_vis=np.linspace(0, 20, 5000)
	pars = camb.set_params(H0=100*phys.h, ombh2=phys.Omega_Bh2, omch2=phys.Omega0*phys.h**2-phys.Omega_Bh2, ns=phys.n, tau=phys.tau)
	data= camb.get_background(pars)
	back_ev = data.get_background_redshift_evolution(z_vis, ['x_e', 'visibility'], format='array')
	g_interpolation=glueAsymptode(interpolate(z_vis, back_ev[:,1]), min=0.1, minAsym=lambda z: interpolate(z_vis, back_ev[:,1])(0.1)*(1+z)**2)
	norm=num.integrate(lambda z: g_interpolation(z)*dw_dz(z), min(z_vis), max(z_vis))
	g_func=lambda z: (1-np.exp(-phys.tau))/norm*g_interpolation(z)

	if PLOT_VISIBILITY_FUNCTION:
		plt.figure()
		plt.plot(z_vis, g_func(z_vis))
		plt.xlabel(r"$g(z)$")
		plt.ylabel(r"$z$")


	a_vals=np.logspace(np.log10(1/(1+max(z_vis))), 0, 7)
	z_vals=(1-a_vals)/a_vals
	vish_interp={}
	k_vals=np.logspace(-3,6,300)*phys.h#physical k
	if not COMPUTE_VISH:
		for m in MODEL:
			if m=="AXION_CAMB":
				try:
					kVals,aVals,vish = np.load(folder+"vishniac_"+m+".npy")
					interpolation=RectBivariateSpline(np.log10(aVals), np.log10(kVals), np.log10(vish), bbox=[min(np.log10(aVals)), max(np.log10(aVals)), min(np.log10(kVals)), max(np.log10(kVals))])
					interp_func=lambda a, k:10**np.squeeze(interpolation.ev(np.log10(np.array([a])), np.log10(np.array([k]))))

					vish_interp[m]=interp_func
				except Exception as ex:
					print(ex)
					print("Vishniac spectrum for model "+m+" does not exist yet.")
			else:
				try:
					k_vals,vish = np.load(folder+"vishniac_"+m+".npy")
					vish_interp[m]=interpolate(k_vals,vish)
				except:
					print("Vishniac spectrum for model "+m+" does not exist yet.")
					print("Computing...")
					with MyProcessPool(4) as p:
						vish=list(p.imap(lambda k: vish_Spec(k, get_MassPower(m)), k_vals))
						np.save(folder+"vishniac_"+m, (k,vish))
						vish_interp[m]=interpolate(k,vish)
	else:
		#plt.figure()
		vish={}
		
		for m in MODEL:
			if m =="AXION_CAMB":
				vish_z=[]
				for i in range(len(z_vals)):
					z=z_vals[i]
					print("computing S(k,z) for z={}...".format(z))
					with MyProcessPool(4) as p:
						vish_z.append(list(p.imap(lambda k: vish_axions(k, z, get_MassPower(m), axion_G_interp, axion_dG_interp), k_vals)))
						#if i%(len(z_vals)//NUM_PLOT_VISH + 1)==0:
							#plt.loglog(k_vals/phys.h, (k_vals)**2*vish_z[-1], label=m+" at a={:.3f}".format(1/(1+z)))
				vish[m]=np.array(vish_z)
				try:	
					np.save(folder+"vishniac_"+m, (k_vals,a_vals,vish[m]))
				except Exception as ex:
					print(ex)
				interpolation=RectBivariateSpline(np.log10(a_vals), np.log10(k_vals), np.log10(vish[m]))
				interp_func=lambda a, k:10**np.squeeze(interpolation.ev(np.log10(np.array([a])), np.log10(np.array([k]))))
				vish_interp[m]=interp_func

			else:
				with MyProcessPool(4) as p:
					vish[m]=list(p.imap(lambda k: vish_Spec(k, get_MassPower(m)), k_vals))
					for i in range(len(z_vals)):
						if i%(len(z_vals)//NUM_PLOT_VISH + 1)==0:
							z=z_vals[i]
							d_factor=(phys.dD(z)*dz_interp(z)/phys.D(0)**2)
							#plt.loglog(k_vals/phys.h, (k_vals)**2*vish[m]*d_factor**2, label=m+" at a={:.3f}".format(1/(1+z)))
						else:
							continue
					np.save(folder+"vishniac_"+m, (k_vals,vish[m]))
					vish_interp[m]=interpolate(k_vals,vish[m])

		#plt.legend()
		#plt.xlabel(r"$k/h$ [Mpc$^{-1}$]")
		#plt.ylabel(r"$S(k) k^2$ [A.U.]")
		#plt.title("Vishniac Power Spectrum")
		#plt.savefig("../Axion_vishniac.png")


	proj_interp={}
	if COMPUTE_PROJECTION:
		#plt.figure()
		
		l_vals=np.logspace(0,5,500)
		pp_vals={}
		
		for m in MODEL:
			with MyProcessPool(4) as p:
				if m =="AXION_CAMB":
					pp_vals[m]=list(p.imap(lambda l:C_proj_axions(l, g_func, vish_interp[m], wz_interp, zw_interp), l_vals))
					#plt.loglog(l_vals, l_vals*(l_vals+1)*pp_vals[m], label=m)
					np.save(folder+"projectedPower_"+m, (l_vals,pp_vals[m]))
					proj_interp[m]=interpolate(l_vals,pp_vals[m])
				else:
					pp_vals[m]=list(p.imap(lambda l:C_proj(l, g_func, dz_interp, phys.dD, phys.D(0), vish_interp[m], wz_interp, zw_interp), l_vals))
					#					#plt.loglog(l_vals, l_vals*(l_vals+1)*pp_vals[m], label=m)
					np.save(folder+"projectedPower_"+m, (l_vals,pp_vals[m]))
					proj_interp[m]=interpolate(l_vals,pp_vals[m])

		#plt.legend()
		#plt.xlabel(r"$\ell$")
		#plt.ylabel(r"$\ell(\ell+1) C_\ell$")
		#plt.savefig("../Axion_kSZ.png")

	else:
		for m in MODEL:
			try:
				l_vals,pp_vals=np.load(folder+"projectedPower_"+m+".npy")
				proj_interp[m]=interpolate(l_vals, pp_vals)
			except:
				print("Projected power spectrum for model "+m+" does not exist yet.")
				print("Computing...")
				l_vals=np.logspace(1,4,100)
				with MyProcessPool(4) as p:
					if m =="AXION_CAMB":
						pp_vals=list(p.imap(lambda l:C_proj_axions(l, g_func, vish_interp[m], wz_interp, zw_interp), l_vals))
						np.save(folder+"projectedPower_"+m, (l_vals,pp_vals))
						proj_interp[m]=interpolate(l_vals,pp_vals)
					else:
						pp_vals=list(p.imap(lambda l:C_proj(l, g_func, dz_interp, phys.dD, phys.D(0), vish_interp[m], wz_interp, zw_interp), l_vals))
						np.save(folder+"projectedPower_"+m, (l_vals,pp_vals))
						proj_interp[m]=interpolate(l_vals,pp_vals)

	
	
	return proj_interp

#print(COMPUTE_AXION_KSZ())
#plt.show()
