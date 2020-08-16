from physics import Physics
import dill
import helpers
import numpy as np
from scipy.interpolate import interp1d
import warnings

class ParameterGenerator:
	def __init__(self, AxionMass_Vals, AxionFrac_Vals, fiducial_ax_frac, omegaMh2_Vals, fiducial_omegaMh2, omegaBh2_Vals, fiducial_omegaBh2, h_Vals, fiducial_h, ns_Vals, fiducial_ns, logAs_1010_Vals, fiducial_logAs_1010):

		AxionFrac_Vals=np.clip(AxionFrac_Vals,1.0e-6/(fiducial_omegaMh2-fiducial_omegaBh2), 1-(1.0e-6/(fiducial_omegaMh2-fiducial_omegaBh2)))#axion CAMB does not like zero cdm or axion densities

		self.__params=[]
		self.__values=[]
		self.__fiducial_values=[]

		tmp_params=[]
		tmp_vals=[]
		tmp_fiducial=[]

		for axion_mass in AxionMass_Vals:

			AxionFrac_phys=[]
			for axfrac in AxionFrac_Vals:
				phys=Physics.create_parameter_set(axion_mass, axfrac, fiducial_omegaMh2, fiducial_omegaBh2, fiducial_h, fiducial_ns, fiducial_logAs_1010)
				AxionFrac_phys.append(phys)

			tmp_params.append(AxionFrac_phys)
			tmp_vals.append(AxionFrac_Vals)
			tmp_fiducial.append(fiducial_ax_frac)

		self.__params.append(tmp_params)
		self.__values.append(tmp_vals)
		self.__fiducial_values.append(tmp_fiducial)

		omegaM_phys=[]
		for oMh2 in omegaMh2_Vals:
			phys=Physics.create_parameter_set(1.0e-24, 0.0, oMh2, fiducial_omegaBh2, fiducial_h, fiducial_ns, fiducial_logAs_1010)
			omegaM_phys.append(phys)

		omegaB_phys=[]
		for oBh2 in omegaBh2_Vals:
			phys=Physics.create_parameter_set(1.0e-24, 0.0, fiducial_omegaMh2, oBh2, fiducial_h, fiducial_ns, fiducial_logAs_1010)
			omegaB_phys.append(phys)

		h_phys=[]
		for h in h_Vals:
			phys=Physics.create_parameter_set(1.0e-24, 0.0, fiducial_omegaMh2, fiducial_omegaBh2, h, fiducial_ns, fiducial_logAs_1010)
			h_phys.append(phys)

		ns_phys=[]
		for n in ns_Vals:
			phys = Physics.create_parameter_set(1.0e-24, 0.0, fiducial_omegaMh2, fiducial_omegaBh2, fiducial_h, n, fiducial_logAs_1010)
			ns_phys.append(phys)

		As_phys = []
		for logAs_1010 in logAs_1010_Vals:
			phys = Physics.create_parameter_set(1.0e-24, 0.0, fiducial_omegaMh2, fiducial_omegaBh2, fiducial_h, fiducial_ns, logAs_1010)
			As_phys.append(phys)

		if len(omegaM_phys)>0:
			self.__params.append(omegaM_phys)
			self.__values.append(omegaMh2_Vals)
			self.__fiducial_values.append(fiducial_omegaMh2)
		if len(omegaB_phys)>0:
			self.__params.append(omegaB_phys)
			self.__values.append(omegaBh2_Vals)
			self.__fiducial_values.append(fiducial_omegaBh2)
		if len(h_phys)>0:
			self.__params.append(h_phys)
			self.__values.append(h_Vals)
			self.__fiducial_values.append(fiducial_h)
		if len(ns_phys)>0:
			self.__params.append(ns_phys)
			self.__values.append(ns_Vals)
			self.__fiducial_values.append(fiducial_ns)
		if len(As_phys)>0:
			self.__params.append(As_phys)
			self.__values.append(logAs_1010_Vals)
			self.__fiducial_values.append(fiducial_logAs_1010)


	def save_to_file(self, filename, flat=True):
		with open(filename, 'wb') as file:
			dill.dump(self.get_params(flat), file)

	def load_from_file(self, filename):
		with open(filename, 'rb') as file:
			self.__params = dill.load(file)

	@staticmethod
	def flatten(array):
		flat_params=[]
		for x in array:
			if helpers.is_array(x):
				flat_params += ParameterGenerator.flatten(x)
			else:
				flat_params.append(x)
		return flat_params

	def get_params(self, flat=True):
		if flat:
			return ParameterGenerator.flatten(self.__params)
		else:
			return self.__params

	def get_pickled_params(self):
		return [dill.dumps(x) for x in self.get_params(flat=True)]

	def regroup_results(self, results):
		return self.regroup_results_recursive(results, self.__values)

	@staticmethod
	def regroup_results_recursive(results, values):
		output_results=[]
		pos=0
		for i in range(len(values)):
			if helpers.is_array(values[i]):
				output_results.append(ParameterGenerator.regroup_results_recursive(results[pos:pos+len(ParameterGenerator.flatten(values[i]))],values[i]))
				pos+=len(ParameterGenerator.flatten(values[i]))
			else:
				output_results.append(results[pos])
				pos+=1

		return output_results

	def get_derivatives(self, results, r_vals_eval=None,  deg=3, z_vals=[1.15]):
		return self.get_derivatives_recursive(results, self.__values, self.__fiducial_values, r_vals_eval, deg, z_vals)

	@staticmethod
	def get_derivatives_recursive(results, values, fiducial_value, r_vals_eval, deg=3, z_vals=[1.15]):
		derivatives=[]

		if helpers.is_array(values[0]):
			for i in range(len(values)):
				derivatives.append(ParameterGenerator.get_derivatives_recursive(results[i], values[i], fiducial_value[i], r_vals_eval, deg, z_vals))
		else:

			r_vals=None
			for vals in results:
				try:
					r_vals=vals[0]
					break
				except Exception as e:
					print(e)
					continue

			if r_vals is None:
				derivatives=np.full((len(z_vals), len(r_vals_eval)), np.nan)
				warnings.warn('Derivative could not be computed!')
				return derivatives

			for z in range(len(z_vals)):
				tmp_ds = []
				for k in range(len(r_vals)):
					v_vals = []
					value_vals = []

					"""
					try:
						d=(-3*results[0][1][z][k] + 4*results[1][1][z][k] - results[2][1][z][k])/2/(values[1]-values[0])
					except:
						d=np.nan

					"""
					for v in range(len(values)):
						try:
							v_vals.append(results[v][1][z][k])
							value_vals.append(values[v])
						except:
							pass
					if len(v_vals) > 1:
						actual_degree=min([deg, len(v_vals)-1])
						fit = np.polyfit(value_vals, v_vals, deg=actual_degree)
						d = 0
						for h in range(actual_degree):
							d += (actual_degree - h) * fit[h] * fiducial_value ** (actual_degree - h - 1)
					else:
						d = np.nan
						warnings.warn('Derivative could not be computed!')

					tmp_ds.append(d)
				if r_vals_eval is not None:
					d_vals = interp1d(r_vals, tmp_ds)(r_vals_eval)
					derivatives.append(d_vals)
				else:
					derivatives.append(tmp_ds)

		return derivatives




"""
ax_masses=np.logspace(-26, -20, 10)
ax_fracs=np.linspace(0,1,10)

ax_frac_dstep=0.01

AxionFrac_Vals=[]
for ax in ax_fracs:
	AxionFrac_Vals.append(ax-ax*ax_frac_dstep)
	AxionFrac_Vals.append(ax+ax*ax_frac_dstep)

paramGen=ParameterGenerator(ax_masses, AxionFrac_Vals)

print(np.shape(paramGen.get_params()))
"""

