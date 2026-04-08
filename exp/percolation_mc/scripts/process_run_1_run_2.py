from tqdm import tqdm
import numpy as np
from scipy.stats import weibull_min
from pathlib import Path
import itertools
import math

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def weibull_convertion(data:np.ndarray)\
	-> (np.ndarray, np.ndarray, np.ndarray):
	# step 1: sort
	data_sorted = np.sort(data)
	n = len(data_sorted)
	
	# step 2: plotting position
	i = np.arange(1, n+1)
	F = (i - 0.3) / (n + 0.4)
	
	# step 3: weibits
	weibits = np.log(-np.log(1 - F))
	
	# x-axis for Weibull plot
	x = np.log(data_sorted)

	return data_sorted, x, weibits

def fit_weibull(data:np.ndarray) -> (float, float):
	shape, _, scale = weibull_min.fit(data, floc=0)
	beta = shape
	eta = scale
	return beta, eta

def fit_plot(r_list:[float], dim_list:[[float, float, float]],\
			 root_path:Path, sub_sample_num:int=100,\
			 seed:int=42) -> None:
	total_list = []
	r_list_n = len(r_list)
	dim_list_n = len(dim_list)
	rng = np.random.default_rng(seed)
	for (radius, (dimx, dimy, dimz)) in tqdm(itertools.product(r_list, dim_list), total=r_list_n*dim_list_n):
		folder_name = f"x{dimx:.2f}_y{dimy:.2f}_z{dimz:.2f}_r{radius:.2f}"
		folder_path = root_path / folder_name

		break_np = np.load(folder_path / "break_flag.npy")
		defect_np = np.load(folder_path / "defect_num.npy")
		volume_np = np.load(folder_path / "norm_volume.npy")

		mask = (break_np == True)

		fig1, ax = plt.subplots(figsize=(8, 5))
		ax.hist(volume_np[mask], bins=200, density=True)
		ax.grid()
		ax.set_ylabel('Probability Density')
		ax.set_xlabel('Normalized Defect Volume')
		ax.set_title(f'x={dimx:.2f}; y={dimy:.2f}; z={dimz:.2f}; r={radius:.2f}')
		fig1.savefig(folder_path / "norm_defect_vol.png")
		plt.close(fig1)

		sim_volume = dimx*dimy*dimz
		defect_density = defect_np[mask]/sim_volume
		fig2, ax = plt.subplots(figsize=(8, 5))
		ax.hist(defect_density, bins=200, density=True)
		ax.grid()
		ax.set_ylabel('Probability Density')
		ax.set_xlabel('Defect Density')
		ax.set_title(f'x={dimx:.2f}; y={dimy:.2f}; z={dimz:.2f}; r={radius:.2f}')
		fig2.savefig(folder_path / "defect_density.png")
		plt.close(fig2)

		data_sorted, _, weibits = weibull_convertion(defect_density)
		beta, eta = fit_weibull(defect_density)
		defect_linspace = np.linspace(start=defect_density.min(), stop=defect_density.max(), num=1000)
		fitted_weibits = beta * (np.log(defect_linspace) - np.log(eta))
		fig3, ax = plt.subplots(figsize=(8, 5))
		ax.scatter(data_sorted, weibits, label='Simulation')
		ax.plot(defect_linspace, fitted_weibits, label='Fit')
		ax.set_xscale('log')
		ax.set_ylabel("Weibit")
		ax.set_xlabel("Defect Density")
		ax.set_title(f'x={dimx:.2f}; y={dimy:.2f}; z={dimz:.2f}; r={radius:.2f}')
		ax.grid()
		ax.legend()
		fig3.savefig(folder_path / "defect_weibits.png")
		plt.close(fig3)

		N = len(data_sorted)
		rand_idxs = rng.integers(0, N+1, size=sub_sample_num)
		total_list.append((beta, eta, data_sorted[rand_idxs], weibits[rand_idxs]))

	cmap = cm.viridis
	norm = colors.Normalize(vmin=dim_list[0][-1], vmax=dim_list[-1][-1])

	weibits_min, weibits_max = -3, 1
	for i, radius in enumerate(r_list):
		fig, ax = plt.subplots(figsize=(8, 5))
		for j, item in enumerate(total_list[i*dim_list_n:(i+1)*dim_list_n]):
			beta, eta, defect_density, weibits = item
			# weibits = b(ln(D) - ln(n))
			# ln(D) = ln(n) + weibits/b
			# D = n * e^(weibits/b)

			Dmin = eta * ((math.e)**(weibits_min/beta))
			Dmax = eta * ((math.e)**(weibits_max/beta))

			D_ls = np.linspace(start=Dmin, stop=Dmax, num=1000)
			fit_weibits = beta * (np.log(D_ls) - np.log(eta))
			ax.plot(D_ls, fit_weibits, label=f'z={dim_list[j][-1]:.2f} nm', color=cmap(norm(dim_list[j][-1])))

		ax.set_xscale('log')
		ax.grid()
		ax.set_ylabel("Weibits_cube")
		ax.set_xlabel("Defect Density")
		sm = cm.ScalarMappable(cmap=cmap, norm=norm)
		sm.set_array([])
		cbar = fig.colorbar(sm, ax=ax)
		cbar.set_label("Thickness (nm)")
		fig.savefig(root_path / f"mc_fit_r{radius:.2f}.png")
		plt.close(fig)

if __name__ == "__main__":
	root_path = Path("./exp/percolation_mc/gen_data/run_1_run2/")
	r_list = [0.45, 0.75]
	dim_list = [[30, 30, z] for z in np.arange(2, 30+1, 2)]
	sub_sample_num = 100

	fit_plot(r_list=r_list, dim_list=dim_list,\
			 root_path=root_path,\
			 sub_sample_num=sub_sample_num, seed=42)