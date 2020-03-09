import numpy as np
from scipy.interpolate import RectBivariateSpline


class GrowthInterpolation:

	def __init__(self, file, physics=None):
		if physics is None:
			self.__phys=Physics()
		else:
			self.__phys=physics

		def converter(s):
		    try:
		        return float(s)
		    except ValueError:
		        return None

		evol_data=np.loadtxt(file, delimiter=',', converters = {0: converter})
		self.__aVals=evol_data[0,1:]
		self.__kVals=evol_data[1:,0]
		self.__growthData=evol_data[1:,1:]


		normalized_dataset=self.__growthData/np.meshgrid(self.__growthData[:,-1], self.__aVals)[0].T
		#print(normalized_dataset[:, -1])
		interpolation=RectBivariateSpline(np.log10(self.__aVals), np.log10(self.__kVals), normalized_dataset.T, bbox=[min(np.log10(self.__aVals)), max(np.log10(self.__aVals)), min(np.log10(self.__kVals)), max(np.log10(self.__kVals))])
		self.__growth= lambda a, k: np.squeeze(interpolation.ev(np.log10(a), np.log10(k)))
		

		interpolation_unnormalized=RectBivariateSpline(np.log10(self.__aVals), np.log10(self.__kVals), self.__growthData.T, bbox=[min(np.log10(self.__aVals)), max(np.log10(self.__aVals)), min(np.log10(self.__kVals)), max(np.log10(self.__kVals))])
		self.__growth_unnormalized=lambda a, k: np.squeeze(interpolation_unnormalized.ev(np.log10(a), np.log10(k)))

		diff_data=np.diff(normalized_dataset, n=1, axis=-1)/np.diff(np.meshgrid(self.__aVals, self.__kVals)[0])
		interpolation_derivative=RectBivariateSpline(np.log10(self.__aVals[0:-1]), np.log10(self.__kVals), diff_data.T)
		self.__dgrowth=lambda a, k: a*self.__phys.da_over_a((1-a)/a)*np.squeeze(interpolation_derivative.ev(np.log10(a), np.log10(k)))


	def get_growth(self):
		return self.__growth, self.__dgrowth

	def get_unnormalized(self):
		return self.__growth_unnormalized, lambda k: self.__growth_unnormalized(1, k)

	#def get_normalized(self):
	#	return lambda a, k: self.__growth(a, k)/self.__growth(np.full(np.shape(a), 1), k),lambda a, k: self.__dgrowth(a, k)/self.__growth(np.full(np.shape(a), 1), k)

#g=GrowthInterpolation(phys=Physics(), readNew=False)
#g.plot_data()