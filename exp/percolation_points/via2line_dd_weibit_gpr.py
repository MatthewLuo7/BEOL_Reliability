import argparse
import math
import numpy as np
import pathlib

from exp.percolation_points.plot_via2line import load_sim_data
from src.distribution.weibull import weibull_convertion, fit_weibull

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.models.physics.local_percolation_gpr.model import WeibullGPR


if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog='via2line_dd_weibit_gpr.py')
	parser.add_argument('--vm-offset-list', nargs="+", type=float,
						default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
						# default=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
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

	fig, ax = plt.subplots(1, 3, figsize=(12, 4))

	vl_space_list = []
	beta_list = []
	eta_list = []

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
		ax[0].scatter(data_sorted[rand_idxs], weibits[rand_idxs], label=f'exp vl={vl_space:.2f} nm', color=cmap(norm(vl_space)), s=5)

		beta, eta = fit_weibull(data=sucs_defect_density)
		vl_space_list.append(vl_space)
		beta_list.append(beta)
		eta_list.append(eta)

	vl_space_np = np.array(vl_space_list)
	beta_np = np.array(beta_list)
	eta_np = np.array(eta_list)

	wb_gpr = WeibullGPR()
	wb_gpr.fit(vl_space_np, beta_np, eta_np)

	vl_linspace = np.linspace(0., 10., 100)
	beta_pd, eta_pd, std_beta, std_eta = wb_gpr.predict(vl_query=vl_linspace, return_std=True)

	ax[1].plot(vl_linspace, beta_pd, label='mean')
	ax[1].plot(vl_linspace, beta_pd + 3*std_beta, label='upper', ls='--')
	ax[1].plot(vl_linspace, beta_pd - 3*std_beta, label='lower', ls='--')
	ax[1].grid()
	ax[1].set_ylabel("Beta")
	ax[1].set_xlabel("Via-Line Space (nm)")

	ax[2].plot(vl_linspace, eta_pd, label='mean')
	ax[2].plot(vl_linspace, eta_pd + 3*std_eta, label='upper', ls='--')
	ax[2].plot(vl_linspace, eta_pd - 3*std_eta, label='lower', ls='--')
	ax[2].grid()
	ax[2].set_ylabel("Eta")
	ax[2].set_xlabel("Via-Line Space (nm)")

	# fit curves
	weibits_min, weibits_max = -6, 2
	plot_Dmin, plot_Dmax = None, None
	for vm_offset in vm_offset_list:
		vl_space = ll_space - vm_offset

		beta, eta = wb_gpr.predict(vl_query=vl_space, return_std=False)

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
		ax[0].plot(D_ls, fit_weibits, label=f'fit vl={vl_space:.2f} nm', color=cmap(norm(vl_space)))

	ax[0].set_xscale('log')
	ax[0].grid()
	ax[0].set_ylabel("Weibits")
	ax[0].set_xlabel("Defect Density")
	ax[0].set_ylim(weibits_min, weibits_max)
	# ax.set_title(f"C_SC: {C_SC_opt:.4f}, P_SC: {P_SC_opt:.4f}, mu_SC: {mu_SC_opt:.4f}, W_mu: {W_mu_opt:.4f}")
	sm = cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])
	cbar = fig.colorbar(sm, ax=ax[0])
	cbar.set_label("vl space (nm)")
	plt.tight_layout()
	plt.show()