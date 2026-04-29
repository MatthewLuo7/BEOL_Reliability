import argparse
import math
import numpy as np
import pathlib

from scipy.optimize import minimize

from exp.percolation_points.plot_via2line import load_sim_data
from src.distribution.weibull import weibull_convertion

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def build_loss_func(fix_W_mu:bool=True,\
					W_mu_default:float=math.log(-math.log(0.5))) -> callable:

	def neg_log_likelihood(params:np.ndarray,\
						   vl_space:np.ndarray, D_OT:np.ndarray) -> np.ndarray:
		# get parameters
		if fix_W_mu:
			C_SC, P_SC, mu_SC = params
			W_mu = W_mu_default
		else:
			C_SC, P_SC, mu_SC, W_mu = params
		
		# modeling
		beta = C_SC * (vl_space ** P_SC)
		
		# numerical stability
		eps = 1e-12
		D_OT = np.maximum(D_OT, eps)
		beta = np.maximum(beta, eps)

		# val = np.exp(beta * np.log(D_OT / mu_SC) + W_mu)
		# val = np.exp(W_mu)
		# val = np.exp(np.log(D_OT / mu_SC))
		# val = np.exp(beta)
		# val = beta
		# val = vl_space ** P_SC
		# val = vl_space
		# val = P_SC
		# print(val.min(), val.max())
		
		# log-likelihood
		logf = (
			np.log(beta)
			+ (beta - 1) * np.log(D_OT)
			- beta * np.log(mu_SC)
			+ np.exp(beta * np.log(D_OT / mu_SC) + W_mu)
			# + W_mu
			# - ((D_OT / mu_SC) ** beta) * np.exp(W_mu)
		)
		return -np.sum(logf)
		
	return neg_log_likelihood

def weibit_fitting(vl_space:np.ndarray, defect_density:np.ndarray,
				   fix_W_mu:bool=True, seed:int=0):
	rng = np.random.default_rng(seed)

	perm_idx = rng.permutation(len(defect_density))
	vl_space = vl_space[perm_idx]
	defect_density = defect_density[perm_idx]

	D_OT_default = np.median(defect_density)
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
			(1e-12, 1),		# C_SC
			(1e-12, 1),		# P_SC
			(0.8, 1.5)		# mu_SC
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
			(0.1, 2),		# C_SC
			(0.1, 2),		# P_SC
			(0.9, 1.1),		# mu_SC
			(1, 3)			# W_mu
		]
	
	# optimization
	neg_log_likelihood = build_loss_func(fix_W_mu=fix_W_mu, W_mu_default=W_mu_default)
	result = minimize(
		neg_log_likelihood,
		init_params,
		args=(vl_space, defect_density),
		method='L-BFGS-B',
		bounds=bounds
	)

	# output
	if fix_W_mu:
		C_SC_opt, P_SC_opt, mu_SC_opt = result.x
		W_mu_opt = W_mu_default
	else:
		C_SC_opt, P_SC_opt, mu_SC_opt, W_mu_opt = result.x

	return C_SC_opt, P_SC_opt, mu_SC_opt, W_mu_opt
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog='via2line_dd_weibit_hinge.py')
	parser.add_argument('--vm-offset-list', nargs="+", type=float,
						default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
						help='List of vm offset')
	parser.add_argument('--ll-space', type=float, default=10.5, help='Line-line spacing, along the x dim')
	parser.add_argument('--via-dim-y', type=float, default=10.5, help='Via size (y dim)')
	parser.add_argument('--via-dim-z', type=float, default=21.0, help='Via size (z dim)')
	parser.add_argument('--line-dim-x', type=float, default=10.5, help='Line size (x dim)')
	parser.add_argument('--line-dim-y', type=float, default=21.0, help='Line size (y dim)')
	parser.add_argument('--line-dim-z', type=float, default=21.0, help='Line size (z dim)')
	parser.add_argument('-r', '--radius', type=float, default=0.45, help='Radius of defects')
	parser.add_argument('--plot-sample-num', type=int, default=100, help='The number of dots for plotting')
	parser.add_argument('--seed', type=int, default=73)
	parser.add_argument('--verbose', action='store_true')
	args = parser.parse_args()
	print(args)

	rng = np.random.default_rng(args.seed)
	plot_sample_num = args.plot_sample_num
	vm_offset_list = args.vm_offset_list

	ll_space = args.ll_space
	via_dim_y = args.via_dim_y
	via_dim_z = args.via_dim_z
	line_dim_x = args.line_dim_x
	line_dim_y = args.line_dim_y
	line_dim_z = args.line_dim_z
	radius = args.radius

	cmap = cm.turbo_r		# cm.viridis
	norm = colors.Normalize(vmin=ll_space-vm_offset_list[-1], vmax=ll_space-vm_offset_list[0])

	V = (line_dim_x*2+ll_space) * line_dim_y * (line_dim_z + via_dim_z) -\
		line_dim_x*line_dim_y*line_dim_z*2 - (max(line_dim_x+vm_offset, 0.) - max(vm_offset, 0.))*via_dim_y*via_dim_z

	fig, ax = plt.subplots(figsize=(8, 5))

	all_vl_space = []
	all_defect_density = []

	for vm_offset in vm_offset_list:
		args.vm_offset = vm_offset
		vl_space = ll_space - vm_offset
		_, defect_num, _, sucs_sim_idxs, _ = load_sim_data(args=args)
		sucs_defect_num = [defect_num[idx] for idx in sucs_sim_idxs]
		sucs_defect_density = np.array(sucs_defect_num).astype(np.float32) / V

		# convert to weibits
		data_sorted, _, weibits = weibull_convertion(data=sucs_defect_density)

		# scatter dots
		N = len(data_sorted)
		rand_idxs = rng.integers(0, N, size=plot_sample_num)
		ax.scatter(data_sorted[rand_idxs], weibits[rand_idxs], label=f'exp vl={vl_space:.2f} nm', color=cmap(norm(vl_space)), s=5)

		all_vl_space.extend([vl_space] * N)
		all_defect_density.extend(sucs_defect_density.tolist())


	# fit and plot
	# print(f'num: {len(all_vl_space)}')
	C_SC_opt, P_SC_opt, mu_SC_opt, W_mu_opt = weibit_fitting(vl_space=np.array(all_vl_space),
															 defect_density=np.array(all_defect_density),
															 fix_W_mu=False, seed=args.seed)

	# fit curves
	weibits_min, weibits_max = -6, 2
	plot_Dmin, plot_Dmax = None, None
	for vm_offset in vm_offset_list:
		vl_space = ll_space - vm_offset

		beta = C_SC_opt * (vl_space ** P_SC_opt)
		eta = mu_SC_opt * np.exp(-W_mu_opt/beta)

		Dmin = eta * (np.exp(weibits_min/beta))
		Dmax = eta * (np.exp(weibits_max/beta))
		if (plot_Dmin is None) or (plot_Dmax is None):
			plot_Dmin = Dmin
			plot_Dmax = Dmax
		else:
			plot_Dmin = min(plot_Dmin, Dmin)
			plot_Dmax = max(plot_Dmax, Dmax)

		D_ls = np.linspace(start=Dmin, stop=Dmax, num=100)
		fit_weibits = beta * (np.log(D_ls) - np.log(eta))
		ax.plot(D_ls, fit_weibits, label=f'fit vl={vl_space:.2f} nm', color=cmap(norm(vl_space)))

	ax.set_xscale('log')
	ax.grid()
	ax.set_ylabel("Weibits")
	ax.set_xlabel("Defect Density")
	ax.set_title(f"C_SC: {C_SC_opt:.4f}, P_SC: {P_SC_opt:.4f}, mu_SC: {mu_SC_opt:.4f}, W_mu: {W_mu_opt:.4f}")
	sm = cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])
	cbar = fig.colorbar(sm, ax=ax)
	cbar.set_label("vl space (nm)")
	
	plt.show()


	# ax.set_xscale('log')
	# ax.grid()
	# ax.set_ylabel("Weibits_cube")
	# ax.set_xlabel("Defect Density")
	# ax.set_title(f"C_SC: {C_SC_opt:.4f}, P_SC: {P_SC_opt:.4f}, mu_SC: {mu_SC_opt:.4f}, W_mu: {W_mu_opt:.4f}")
	# sm = cm.ScalarMappable(cmap=cmap, norm=norm)
	# sm.set_array([])
	# cbar = fig1.colorbar(sm, ax=ax)
	# cbar.set_label("Thickness (nm)")
	# suffix = '_fix_W_mu' if fix_W_mu else ''
	# fig1.savefig(root_path / f"mc_fit_r{radius:.2f}_cube{suffix}.png")
	# plt.close(fig1)
	