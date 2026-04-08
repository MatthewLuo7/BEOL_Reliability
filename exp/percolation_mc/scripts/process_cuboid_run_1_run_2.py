from tqdm import tqdm
import numpy as np
from scipy.stats import weibull_min
from scipy.optimize import minimize
from pathlib import Path
import itertools
import math

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def weibull_convertion_cuboid(data:np.ndarray, dimx:float, dimy:float, dimz:float)\
	-> (np.ndarray, np.ndarray, np.ndarray):
	# step 1: sort
	data_sorted = np.sort(data)
	n = len(data_sorted)
	
	# step 2: plotting position
	i = np.arange(1, n+1)
	F = (i - 0.3) / (n + 0.4)
	
	# step 3: weibits
	weibits_area = np.log(-np.log(1 - F))
	weibits_cube = weibits_area + 2*math.log(dimz) - math.log(dimx*dimy)
	
	# x-axis for Weibull plot
	x = np.log(data_sorted)

	return data_sorted, x, weibits_cube

def fit_weibull_cuboid(data:np.ndarray, dimx:float, dimy:float, dimz:float) -> (float, float):
	shape, _, scale = weibull_min.fit(data, floc=0)
	beta = shape
	eta_area = scale
	eta_cube = eta_area * (((dimx*dimy)/(dimz**2))**(1/beta))
	return beta, eta_cube

def build_loss_func(dimx:float, dimy:float,\
					fix_W_mu:bool=False,\
					W_mu_default:float=-0.3665) -> callable:
	area = dimx*dimy

	def neg_log_likelihood(params:np.ndarray,\
						   dimz:np.ndarray, D_OT:np.ndarray) -> np.ndarray:
		# get parameters
		if fix_W_mu:
			C_SC, P_SC, mu_SC = params
			W_mu = W_mu_default
		else:
			C_SC, P_SC, mu_SC, W_mu = params
		
		# modeling
		beta = C_SC * (dimz ** P_SC)
		N_cube = area / (dimz ** 2)
		
		# numerical stability
		eps = 1e-12
		D_OT = np.maximum(D_OT, eps)
		beta = np.maximum(beta, eps)
		
		# log-likelihood
		logf = (
			np.log(beta)
			+ (beta - 1) * np.log(D_OT)
			- beta * np.log(mu_SC)
			+ W_mu
			+ np.log(N_cube)
			- N_cube * np.exp(W_mu) * ((D_OT / mu_SC) ** beta)
		)
		return -np.sum(logf)
		
	return neg_log_likelihood


def fit_plot_cuboid(r_list:[float], dim_list:[[float, float, float]],\
					root_path:Path, fit_sample_num:int=1000,\
					plot_sample_num:int=50,\
					seed:int=42, fix_W_mu:bool=False) -> None:
	total_list = []
	r_list_n = len(r_list)
	dim_list_n = len(dim_list)
	rng = np.random.default_rng(seed)
	for (radius, (dimx, dimy, dimz)) in tqdm(itertools.product(r_list, dim_list), total=r_list_n*dim_list_n):
		folder_name = f"x{dimx:.2f}_y{dimy:.2f}_z{dimz:.2f}_r{radius:.2f}"
		folder_path = root_path / folder_name

		break_np = np.load(folder_path / "break_flag.npy")
		defect_np = np.load(folder_path / "defect_num.npy")
		mask = (break_np == True)

		sim_volume = dimx*dimy*dimz
		defect_density = defect_np[mask]/sim_volume


		data_sorted, _, weibits_cube = weibull_convertion_cuboid(defect_density, dimx, dimy, dimz)
		beta, eta_cube = fit_weibull_cuboid(defect_density, dimx, dimy, dimz)
		defect_linspace = np.linspace(start=defect_density.min(), stop=defect_density.max(), num=1000)
		fitted_weibits = beta * (np.log(defect_linspace) - np.log(eta_cube))
		fig, ax = plt.subplots(figsize=(8, 5))
		ax.scatter(data_sorted, weibits_cube, label='Simulation')
		ax.plot(defect_linspace, fitted_weibits, label='Fit')
		ax.set_xscale('log')
		ax.set_ylabel("Weibits_cube")
		ax.set_xlabel("Defect Density")
		ax.set_title(f'x={dimx:.2f}; y={dimy:.2f}; z={dimz:.2f}; r={radius:.2f}')
		ax.grid()
		ax.legend()
		fig.savefig(folder_path / "defect_weibits_cube.png")
		plt.close(fig)

		N = len(data_sorted)
		rand_idxs = rng.integers(0, N, size=fit_sample_num)
		total_list.append((beta, eta_cube, data_sorted[rand_idxs], weibits_cube[rand_idxs]))

	cmap = cm.turbo_r		# cm.viridis
	norm = colors.Normalize(vmin=dim_list[0][-1], vmax=dim_list[-1][-1])

	weibits_min, weibits_max = -3, 1
	for i, radius in enumerate(r_list):
		fig1, ax = plt.subplots(figsize=(8, 5))

		dimz_list = []
		D_OT_list = []
		weibits_cube_list = []
		for j, item in enumerate(total_list[i*dim_list_n:(i+1)*dim_list_n]):
			beta, eta, defect_density, weibits_cube = item
			Dmin = eta * ((math.e)**(weibits_min/beta))
			Dmax = eta * ((math.e)**(weibits_max/beta))

			D_ls = np.linspace(start=Dmin, stop=Dmax, num=1000)
			fit_weibits = beta * (np.log(D_ls) - np.log(eta))
			ax.plot(D_ls, fit_weibits, label=f'z={dim_list[j][-1]:.2f} nm', color=cmap(norm(dim_list[j][-1])))

			dimz_list.extend([dim_list[j][-1]] * fit_sample_num)
			D_OT_list.extend(defect_density.tolist())
			weibits_cube_list.extend(weibits_cube.tolist())

		# ----- fitting -----
		dimz_data = np.array(dimz_list)
		D_OT_data = np.array(D_OT_list)
		weibits_cube_data = np.array(weibits_cube_list)

		perm_idx = rng.permutation(len(dimz_data))
		dimz_data = dimz_data[perm_idx]
		D_OT_data = D_OT_data[perm_idx]
		weibits_cube_data = weibits_cube_data[perm_idx]

		D_OT_default = np.median(D_OT_data)
		W_mu_default = np.log(-np.log(1 - 0.5))
		
		if fix_W_mu:
			# initialization
			init_params = [
				1.0,
				0.5,
				D_OT_default
			]

			# bounds
			bounds=[
				(1e-12, 10),	# C_SC
				(1e-12, 10),	# P_SC
				(1e-12, 10)		# mu_SC
			]
		else:
			# initialization
			init_params = [
				1.0,
				0.5,
				D_OT_default,
				W_mu_default
			]

			# bounds
			bounds=[
				(1e-12, 10),	# C_SC
				(1e-12, 10),	# P_SC
				(1e-12, 10),	# mu_SC
				(-5, 5)			# W_mu
			]
		
		# optimization
		neg_log_likelihood = build_loss_func(dimx=dim_list[j][0], dimy=dim_list[j][1],\
											 fix_W_mu=fix_W_mu, W_mu_default=W_mu_default)
		result = minimize(
			neg_log_likelihood,
			init_params,
			args=(dimz_data, D_OT_data),
			method='L-BFGS-B',
			bounds=bounds
		)

		# output
		if fix_W_mu:
			C_SC_opt, P_SC_opt, mu_SC_opt = result.x
			W_mu_opt = W_mu_default
		else:
			C_SC_opt, P_SC_opt, mu_SC_opt, W_mu_opt = result.x
		# -------------------

		ax.set_xscale('log')
		ax.grid()
		ax.set_ylabel("Weibits_cube")
		ax.set_xlabel("Defect Density")
		ax.set_title(f"C_SC: {C_SC_opt:.4f}, P_SC: {P_SC_opt:.4f}, mu_SC: {mu_SC_opt:.4f}, W_mu: {W_mu_opt:.4f}")
		sm = cm.ScalarMappable(cmap=cmap, norm=norm)
		sm.set_array([])
		cbar = fig1.colorbar(sm, ax=ax)
		cbar.set_label("Thickness (nm)")
		suffix = '_fix_W_mu' if fix_W_mu else ''
		fig1.savefig(root_path / f"mc_fit_r{radius:.2f}_cube{suffix}.png")
		plt.close(fig1)

		# -------------- plot the fitted curves and samples ---------------
		weibits_min, weibits_max = -6, 1
		fig2, ax = plt.subplots(figsize=(8, 5))
		plot_Dmin, plot_Dmax = None, None
		for j, item in enumerate(total_list[i*dim_list_n:(i+1)*dim_list_n]):
			_, _, defect_density, weibits_cube = item
			dimz = dim_list[j][-1]

			# scatter dots
			N = len(defect_density)
			rand_idxs = rng.integers(0, N, size=plot_sample_num)
			ax.scatter(defect_density[rand_idxs], weibits_cube[rand_idxs], label=f'exp z={dimz:.2f} nm', color=cmap(norm(dimz)), s=5)

			# fit curves
			beta = C_SC_opt * (dimz ** P_SC_opt)
			eta_cube = mu_SC_opt * np.exp(-W_mu_opt/beta)

			Dmin = eta * (np.exp(weibits_min/beta))
			Dmax = eta * (np.exp(weibits_max/beta))
			if (plot_Dmin is None) or (plot_Dmax is None):
				plot_Dmin = Dmin
				plot_Dmax = Dmax
			else:
				plot_Dmin = min(plot_Dmin, Dmin)
				plot_Dmax = max(plot_Dmax, Dmax)

			D_ls = np.linspace(start=Dmin, stop=Dmax, num=1000)
			fit_weibits = beta * (np.log(D_ls) - np.log(eta_cube))
			ax.plot(D_ls, fit_weibits, label=f'z={dimz:.2f} nm', color=cmap(norm(dimz)))

		ax.set_xscale('log')
		ax.grid()
		ax.set_ylabel("Weibits_cube")
		ax.set_xlabel("Defect Density")
		ax.set_title(f"C_SC: {C_SC_opt:.4f}, P_SC: {P_SC_opt:.4f}, mu_SC: {mu_SC_opt:.4f}, W_mu: {W_mu_opt:.4f}")
		ax.set_xlim(plot_Dmin, plot_Dmax)
		ax.set_ylim(weibits_min, weibits_max)
		sm = cm.ScalarMappable(cmap=cmap, norm=norm)
		sm.set_array([])
		cbar = fig2.colorbar(sm, ax=ax)
		cbar.set_label("Thickness (nm)")
		suffix = '_fix_W_mu' if fix_W_mu else ''
		fig2.savefig(root_path / f"ml_plot_cmp_r{radius:.2f}_cube{suffix}.png")
		plt.close(fig2)

if __name__ == "__main__":
	root_path = Path("./exp/percolation_mc/gen_data/run_1_run2/")
	r_list = [0.45, 0.75]
	dim_list = [[30, 30, z] for z in np.arange(2, 30+1, 2)]
	fit_sample_num = 10000
	plot_sample_num = 50

	fix_W_mu = True
	fit_plot_cuboid(r_list=r_list, dim_list=dim_list,\
					root_path=root_path,\
					fit_sample_num=fit_sample_num,\
					plot_sample_num=plot_sample_num,\
					seed=42, fix_W_mu=fix_W_mu)