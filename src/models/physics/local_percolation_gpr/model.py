import joblib
import pathlib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import json

# model_root_path = pathlib.Path('./src/models/physics/local_percolation_gpr/model_path/')


# def load_model(m:float,
# 			   radius_N:float=2.0,
# 			   radius:float=0.45,
# 			   ll_space:float=10.5,
# 			   via_dim_y:float=10.5,
# 			   via_dim_z:float=10.5*2,
# 			   line_dim_x:float=10.5,
# 			   line_dim_y:float=10.5*2,
# 			   line_dim_z:float=10.5*2,
# 			   root_path:pathlib.Path=model_root_path):
# 	dir_name = f'll{ll_space:.2f}_vy{via_dim_y:.2f}_vz{via_dim_z:.2f}_lx{line_dim_x:.2f}_ly{line_dim_y:.2f}_lz{line_dim_z:.2f}_r{radius:.2f}'
# 	load_path = root_path / dir_name / f'weibull_gpr_m{m:.2f}_rN{radius_N:.2f}.pkl'

# 	return WeibullGPR.load(load_path)

# class WeibullGPR:
# 	def __init__(self):
# 		# kernel： constant × RBF + noise
# 		kernel = (
# 			ConstantKernel(1.0, (1e-3, 1e3)) *
# 			RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
# 			+ WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
# 		)

# 		# two independent GP
# 		self.gp_beta = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
# 		self.gp_eta  = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

# 	# def save(self, path) -> None:
# 	# 	joblib.dump(self, path)
	
# 	# @staticmethod
# 	# def load(path):
# 	# 	return joblib.load(path)

# 	def fit(self, vl_space:np.array, beta:np.array, eta:np.array,
# 			train_save_path:str=None) -> None:
# 		"""
# 		vl_space: (N,)
# 		beta:     (N,)
# 		eta:      (N,)
# 		"""
# 		X = np.array(vl_space).reshape(-1, 1)

# 		# log-domain to stablize values and avoid negative values
# 		self.gp_beta.fit(X, np.log(beta))
# 		self.gp_eta.fit(X, np.log(eta))

# 		if train_save_path is not None:
# 			np.savez(
# 				path,
# 				data=vl_space,
# 				beta=beta,
# 				eta=eta
# 			)

# 	def predict(self, vl_query:np.array, return_std:bool=False):
# 		"""
# 		vl_query: (M,)
# 		"""
# 		Xq = np.array(vl_query).reshape(-1, 1)

# 		if return_std:
# 			log_beta, std_beta = self.gp_beta.predict(Xq, return_std=True)
# 			log_eta,  std_eta  = self.gp_eta.predict(Xq, return_std=True)

# 			beta = np.exp(log_beta)
# 			eta  = np.exp(log_eta)

# 			return beta, eta, std_beta, std_eta
# 		else:
# 			beta = np.exp(self.gp_beta.predict(Xq))
# 			eta  = np.exp(self.gp_eta.predict(Xq))

# 			return beta, eta

def load_model(train_save_path):
	data = np.load(train_save_path)
	X = data["X"]
	beta = data["beta"]
	eta = data["eta"]

	model = WeibullGPR()
	model.fit(X=X, beta=beta, eta=eta)

	return model

class WeibullGPR:
	def __init__(self):
		kernel = (
			ConstantKernel(1.0, (1e-3, 1e3)) *
			RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
			+ WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
		)

		self.gp_beta = GaussianProcessRegressor(
			kernel=kernel,
			n_restarts_optimizer=5
		)

		self.gp_eta = GaussianProcessRegressor(
			kernel=kernel,
			n_restarts_optimizer=5
		)

	def fit(self, X:np.ndarray,
			beta:np.ndarray,
			eta:np.ndarray,
			train_save_path:str=None):
		"""
		X [:, 2]
		(vl_space, ll_space)
		"""
		# log-domain to stablize values and avoid negative values
		self.gp_beta.fit(X, np.log(beta))
		self.gp_eta.fit(X, np.log(eta))

		if train_save_path is not None:
			np.savez(
				train_save_path,
				X=X,
				beta=beta,
				eta=eta
			)

	def predict(self, X:np.ndarray, verbose:bool=False, conf_factor:float=2.):
		if verbose:
			log_beta, std_beta = self.gp_beta.predict(X, return_std=True)
			log_eta,  std_eta  = self.gp_eta.predict(X, return_std=True)

			log_beta = np.expand_dim(log_beta, axis=-1)
			log_eta = np.expand_dim(log_eta, axis=-1)

			log_beta_itv = np.array([0., -std_beta, std_beta]) * conf_factor
			log_eta_itv = np.array([0., -std_eta, std_eta]) * conf_factor

			log_beta += log_beta_itv
			log_eta += log_eta_itv

			return (
				np.exp(log_beta),
				np.exp(log_eta),
			)
		else:
			return (
				np.exp(self.gp_beta.predict(Xq)),
				np.exp(self.gp_eta.predict(Xq))
			)