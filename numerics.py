from gaussxw import gaussxw,gaussxwab
import numpy as np
from numpy import pi
from scipy.interpolate import pchip
from helpers import *
from num_solve import secant as solve


class IntegrationNumerics:

	def __init__(self, GAUSS=True, gaussPoints=300):
		self.__eval_points, self.__weights =gaussxw(gaussPoints)

		if GAUSS:
			self.integrate2D=self.integrate2DGauss
		else:
			self.integrate2D=self.integrate2DRiemann


	"""
	function to integrate a function within a list of limtis or just one well defined limit
	"""
	def integrate(self, func, xmin, xmax):
		if is_array(xmax)and is_array(xmin):
			return np.array(list(map(lambda lim: self.integrate(func, lim[0], lim[1]), zip(xmin, xmax))))
		elif is_array(xmax, np.ndarray):
			return np.array(list(map(lambda max: self.integrate(func, xmin, max), xmax)))
		elif is_array(xmin, np.ndarray):
			return np.array(list(map(lambda min: self.integrate(func, min, xmax), xmin)))
		else:
			return self.integrateSimple(func, xmin, xmax)
	
	def integrateSimple(self, func, xmin, xmax):
		x=(xmax-xmin)/2*self.__eval_points+(xmax+xmin)/2
		return (xmax-xmin)/2*sum(self.__weights*func(x))

	"""
	function to perform integration via Trapezoidal rule
	"""
	def integrateT(self, func, xmin, xmax, N=1000):
		width=(xmax-xmin)/N#compute width of the intervals
		"""
		Note:when func is given xmin+np.arange(...)*width this will be a vector where each entry is xmin + np.arange(...)[i]*width. 
		Since the function only relies on numpy it is able the process this and itself return a vector r where r[i]=func(xmin + np.arange(...)[i]*width).
		One can then sum over this vector.
		"""
		return (func(xmin)+func(xmax)+2*sum(func(xmin+np.arange(1,N)*width)))*(xmax-xmin)/(2*N)
		#		function values start and end
		#								2*sum(of intermediate function values)
		#																	multiply by columns width/2

	"""
	function to perform integration via Simpson's rule
	"""
	def integrateS(self, func, xmin, xmax, N=1000):
		width=(xmax-xmin)/N#compute width of the intervals
		"""
		Note:when func is given xmin+np.arange(...)*width this will be a vector where each entry is xmin + np.arange(...)[i]*width. 
		Since the function only relies on numpy it is able the process this and itself return a vector r where r[i]=func(xmin + np.arange(...)[i]*width).
		One can then sum over this vector.
		"""
		return (func(xmin)+func(xmax)+4*sum(func(xmin+np.arange(1,N,2)*width))+2*sum(func(xmin+np.arange(2,N,2)*width)))*width/3
		#		function values at integration limits
		#								4*sum of odd x values (width times odd number)
		#																		2*sum of even x values (width times even number)
		#																												multiply by width/3 


	"""
	function to integrate a function with respect to two variables x and y using Gauss-Legendre Quadrature
	"""
	def integrate2DGauss(self, func, xmin, xmax, ymin=None, ymax=None):
		#if there are no separate y limits given assume the same limits as for x
		if ymin is None:
			ymin = xmin
		if ymax is None:
			ymax = xmax
		
		x=(xmax-xmin)/2*self.__eval_points+(xmax+xmin)/2
		y=(ymax-ymin)/2*self.__eval_points+(ymax+ymin)/2

		xgrid, ygrid = np.meshgrid(x,y)
		weightgrid_x, weightgrid_y = np.meshgrid(self.__weights, self.__weights)

		return (xmax-xmin)*(ymax-ymin)/4*sum(sum(weightgrid_x*weightgrid_y*func(xgrid, ygrid)))
		
	"""
	function to integrate a function with respect to two variables x and y using Riemann integration
	"""
	def integrate2DRiemann(self, func, xmin, xmax, ymin=None, ymax=None, N=1000):
		if ymin is None:
			ymin = xmin
		if ymax is None:
			ymax = xmax

		x_vals = np.linspace(xmin, xmax, N)
		y_vals = np.linspace(ymin, ymax, N)

		spacing_x = (xmax-xmin)/N
		spacing_y = (ymax-ymin)/N

		x,y=np.meshgrid(x_vals, y_vals)

		return np.nansum(np.nansum(spacing_y*spacing_x*func(x,y)))

	def integrate3D(self, func, xmin, xmax, ymin=None, ymax=None, zmin=None, zmax=None, Nx=None, Ny=None, Nz=None):
		if ymin is None:
			ymin = xmin
		if ymax is None:
			ymax = xmax
		
		if zmin is None:
			zmin = xmin
		if zmax is None:
			zmax = xmax

		if Nx is None:
			x=(xmax-xmin)/2*self.__eval_points+(xmax+xmin)/2
			weightsx=self.__weights
		else:
			x,weightsx=gaussxwab(Nx,xmin, xmax)

		if Ny is None:
			y=(ymax-ymin)/2*self.__eval_points+(ymax+ymin)/2
			weightsy=self.__weights
		else:
			y,weightsy=gaussxwab(Ny,ymin, ymax)

		if Nz is None:
			z=(zmax-zmin)/2*self.__eval_points+(zmax+zmin)/2
			weightsz=self.__weights
		else:
			z,weightsz=gaussxwab(Nz,zmin, zmax)


		xgrid, ygrid, zgrid = np.meshgrid(x,y,z)
		weightgrid_x, weightgrid_y, weightgrid_z = np.meshgrid(self.__weights, self.__weights, self.__weights)
		weights=(weightgrid_x*weightgrid_y*weightgrid_z)

		return (xmax-xmin)*(ymax-ymin)*(zmax-zmin)/8*sum(sum(sum(weights*func(xgrid, ygrid, zgrid))))

	def integrateT2D(self, func, N, M, xmin, xmax, ymin, ymax):
		h=(xmax-xmin)/N
		k=(ymax-ymin)/M

		x_vals=np.linspace(xmin, xmax, N)
		y_vals=np.linspace(ymin, ymax, M)

		xGrid,yGrid=np.meshgrid(x_vals, y_vals)

		return h*k/4*(func(xmin, ymin)+func(xmin, ymax)+func(xmax, ymin)+func(xmax, ymax) + 2*(sum(func(xmax, y_vals[1:-1]))+sum(func(xmin, y_vals[1:-1]))+sum(func(x_vals[1:-1], ymin))+sum(func(x_vals[1:-1], ymax)))+4*sum(sum(func(xGrid[1:-1,1:-1],yGrid[1:-1,1:-1]))))

	def integrate2DRomberg(self, func, xmin, xmax, ymin, ymax, N0=500, M0=500, tol=1e-3, max_steps=20):
		R_last=np.array([self.integrateT2D(func, N0, M0, xmin, xmax, ymin, ymax)])
		for i in range(1, max_steps+1):
			rs=np.zeros(i+1)
			rs[0]=self.integrateT2D(func, N0*2**i, M0*2**i, xmin, xmax, ymin, ymax)
			for m in range(1, i+1):
				rs[m]=rs[m-1]+(rs[m-1]-R_last[m-1])/(4**m-1)
			error = np.fabs((R_last[-1]-rs[-1])/R_last[-1])
			print(error)
			if error <= tol:
				print(i)
				return rs[-1]
			else:

				R_last=rs

		raise Exception("Integral did not converge!")

num=IntegrationNumerics(True, gaussPoints=300)

def differentiate(func, h=0.01, log=False):
	try:
		if not log:
			return lambda x: (func(x+h)-func(x-h))/(2*h)
		else:
			return lambda x: (func(x*10**h)-func(x/10**h))/(x*10**h-x/10**h)
	except ValueError as error:
		print(error)
		return np.nan


def differentiate_discrete(y_vals, x_vals):
	return np.gradient(y_vals, x_vals)


###FIX this inteprolation method does not quite work as expected
def interpolate(x_data, y_data, minAsym=None, maxAsym=None, fill_value=None):
	data_interpolation=pchip(x_data, y_data)
	
	if minAsym is None:
		minAsym=lambda x:fill_value if fill_value is not None else np.nan
		scalingLow= fill_value if fill_value is not None else np.nan
	else:
		scalingLow=data_interpolation(min(x_data))/minAsym(min(x_data))
	if maxAsym is None:
		maxAsym=lambda x:fill_value if fill_value is not None else np.nan
		scalingHigh=fill_value if fill_value is not None else np.nan
	else:
		scalingHigh=data_interpolation(max(x_data))/maxAsym(max(x_data))
	
	def func(x):
		###FIX numerical issue where two identical floats are not considered identical
		x=np.array(x)
		return np.piecewise(x, [np.array(x<min(x_data)), np.array(x>max(x_data)), np.array((x>=min(x_data)) & (x<=max(x_data)))], [lambda x:np.array(scalingLow*minAsym(x)), lambda x:np.array(scalingHigh*maxAsym(x)), lambda x:data_interpolation(x)])

	return func


def glueAsymptode(func, min=None, max=None, minAsym=None, maxAsym=None, fill_value=np.nan):

	if minAsym is None:
		minAsym=lambda x:fill_value
		scalingLow=fill_value
	else:
		scalingLow=func(min)/minAsym(min)
	if maxAsym is None:
		maxAsym=lambda x:fill_value
		scalingHigh=fill_value
	else:
		print(maxAsym(1))
		scalingHigh=func(max)/maxAsym(max)

	if min is not None and max is not None:
		return lambda x: np.piecewise(x, [np.array(x<min), np.array(x>max), np.array((x>=min) & (x<=max))], [lambda x:np.array(scalingLow*minAsym(x)), lambda x:np.array(scalingHigh*maxAsym(x)), lambda x:func(x)])
	elif min is not None:
		return lambda x: np.piecewise(x, [np.array(x<min), np.array(x>min)], [lambda x:np.array(scalingLow*minAsym(x)), lambda x:func(x)])
	elif max is not None:
		return lambda x: np.piecewise(x, [np.array(x>max), np.array(x<max)], [lambda x:np.array(scalingHigh*maxAsym(x)), lambda x:func(x)])


#function to find zero crossing of a given function
def find_zeros(func, x_vals, indexes=None, accuracy=1e-3):
	zero_crossings = np.where(np.diff(np.signbit(func(x_vals))))[0]
	vals=[]
	if indexes is None:
		for index in zero_crossings:
			x1=x_vals[index]
			x2=x_vals[index+1]
			try:
				vals.append(solve(func, guesses=(x1,x2), accuracy=accuracy))
			except:
				continue
	else:
		for index in indexes:
			x1=x_vals[index]
			x2=x_vals[index+1]
			try:
				vals.append(solve(func, guesses=(x1,x2), accuracy=accuracy))
			except:
				continue
	return np.array(vals)

