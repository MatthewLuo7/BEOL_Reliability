import argparse
import numpy as np
import pathlib
import numpy as np
from tqdm import tqdm
import joblib

from src.percolation.via2line import Via2LineSim_sumup_time_intervals_create_wrapper
from src.distribution.weibull import weibull_convertion, fit_weibull, weibull_plot

from exp.percolation_points.plot_via2line import load_sim_data
from exp.percolation_points.post_process_via2line import mp_post_process
from exp.percolation_points.via2line_dd_weibit_gpr import WeibullGPR

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
os.environ['KMP_DUPLICATE_LIB_OK']='True'

data_root = pathlib.Path('./exp/percolation_points/gen_data/via2line/')
save_root = pathlib.Path('./exp/percolation_points/gen_data/via2line_bt_fitting_gpr/')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog='via2line_bt_fitting_gpr.py')
	parser.add_argument('--vm-offset-list', nargs="+", type=float,
						default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
						# default=[1.0, 3.0, 5.0, 7.0],
						help='List of vm offset')
	parser.add_argument('--ll-space', type=float, default=10.5, help='Line-line spacing, along the x dim')
	parser.add_argument('--via-dim-x', type=float, default=10.5, help='Via size (x dim)')
	parser.add_argument('--via-dim-y', type=float, default=10.5, help='Via size (y dim)')
	parser.add_argument('--via-dim-z', type=float, default=21.0, help='Via size (z dim)')
	parser.add_argument('--line-dim-y', type=float, default=21.0, help='Line size (y dim)')
	parser.add_argument('--line-dim-z', type=float, default=21.0, help='Line size (z dim)')
	parser.add_argument('-r', '--radius', type=float, default=0.45, help='Radius of defects')

	parser.add_argument('--m-list', nargs="+", type=float, help='A list of parameter m (> 1.0)')
	parser.add_argument('-rN', '--radius-N', type=float, default=2.0, help='Radius for checking neighbouring defects (r_N > r)')

	parser.add_argument('--max-defects', type=int, default=100000, help='The max allowed defects')
	parser.add_argument('--rebuild-thresh', type=int, default=50, help='Iteration interval to rebuild KD-Tree')
	parser.add_argument('--plot-sample-num', type=int, default=100, help='The number of dots for plotting')
	parser.add_argument('--fit-sample-num', type=int, default=-1, help='The number of samples for fitting')

	parser.add_argument('--chunk-size', type=int, default=4, help='Chunk size of simulations')
	parser.add_argument('--cpu-num', type=int, default=-1, help='The number of CPUs')
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument('--verify', action='store_true', help='Verify breakdown')
	parser.add_argument('--seed', type=int, default=73)
	args = parser.parse_args()
	print(args)

	rng = np.random.default_rng(args.seed)

	vm_offset_list = args.vm_offset_list
	ll_space = args.ll_space
	via_dim_x = args.via_dim_x
	via_dim_y = args.via_dim_y
	via_dim_z = args.via_dim_z
	line_dim_y = args.line_dim_y
	line_dim_z = args.line_dim_z
	radius = args.radius
	workers = 1

	m_list = args.m_list
	radius_N = args.radius_N
	m_num = len(m_list)

	max_defects = args.max_defects
	rebuild_thresh = args.rebuild_thresh
	plot_sample_num = args.plot_sample_num

	cmap = cm.turbo_r		# cm.viridis
	norm = colors.Normalize(vmin=ll_space-vm_offset_list[-1], vmax=ll_space-vm_offset_list[0])

	if not save_root.exists():
		save_root.mkdir()
	dir_name = f'll{ll_space:.2f}_vx{via_dim_x:.2f}_vy{via_dim_y:.2f}_vz{via_dim_z:.2f}_ly{line_dim_y:.2f}_lz{line_dim_z:.2f}_r{radius:.2f}'
	save_path = save_root / dir_name
	if not save_path.exists():
		save_path.mkdir()

	figs, axs = zip(*[plt.subplots(1, 3, figsize=(12, 4)) for _ in range(m_num)])

	vl_space_list = []
	beta_list = []
	eta_list = []

	for vm_offset in tqdm(vm_offset_list, total=len(vm_offset_list)):
		args.vm_offset = vm_offset
		vl_space = ll_space - vm_offset

		_, _, _, _, sucs_sim_points = load_sim_data(args=args)
		pp_wrapper = Via2LineSim_sumup_time_intervals_create_wrapper(
						m_np=np.array(m_list).astype(np.float64),
						radius_N=radius_N,
						vm_offset=vm_offset,
						ll_space=ll_space,
						via_dim_x=via_dim_x,	
						via_dim_y=via_dim_y,	
						via_dim_z=via_dim_z,
						line_dim_y=line_dim_y,
						line_dim_z=line_dim_z,
						radius=radius,
						max_defects=max_defects,
						rebuild_thresh=rebuild_thresh,
						workers=workers,
						verify=args.verify)

		sucs_breakdown_time = mp_post_process(pp_wrapper, sucs_sim_points[:args.fit_sample_num],\
											  chunk_size=args.chunk_size,\
											  cpu_num=args.cpu_num,\
											  show_progress=False)
		
		sucs_breakdown_time = sucs_breakdown_time.reshape(-1, m_num)

		# scatter dots
		N = sucs_breakdown_time.shape[0]
		rand_idxs = rng.integers(0, N, size=plot_sample_num)
		for idx in range(m_num):
			# convert to weibits
			data_sorted, _, weibits = weibull_convertion(data=sucs_breakdown_time[:, idx])
			# scatter data
			axs[idx][0].scatter(data_sorted[rand_idxs], weibits[rand_idxs], label=f'exp vl={vl_space:.2f} nm', color=cmap(norm(vl_space)), s=5)

		betas, etas = zip(*[fit_weibull(data=sucs_breakdown_time[:, idx]) for idx in range(m_num)])
		vl_space_list.append(vl_space)
		beta_list.append(betas)
		eta_list.append(etas)

	vl_space_np = np.array(vl_space_list)
	beta_np = np.array(beta_list).reshape(-1, m_num)
	eta_np = np.array(eta_list).reshape(-1, m_num)

	vl_linspace = np.linspace(0., 10., 100)
	wb_gprs = [WeibullGPR() for _ in range(m_num)]
	weibits_min, weibits_max = -6, 3
	for idx in range(m_num):
		wb_gprs[idx].fit(vl_space_np, beta_np[:, idx], eta_np[:, idx])	
		beta_pd, eta_pd, std_beta, std_eta = wb_gprs[idx].predict(vl_query=vl_linspace, return_std=True)

		axs[idx][1].plot(vl_linspace, beta_pd, label='mean')
		axs[idx][1].plot(vl_linspace, beta_pd + 3*std_beta, label='upper', ls='--')
		axs[idx][1].plot(vl_linspace, beta_pd - 3*std_beta, label='lower', ls='--')
		axs[idx][1].grid()
		axs[idx][1].set_ylabel("Beta")
		axs[idx][1].set_xlabel("Via-Line Space (nm)")
		axs[idx][1].set_title("GP Regression for Beta")
		axs[idx][1].legend()

		axs[idx][2].plot(vl_linspace, eta_pd, label='mean')
		axs[idx][2].plot(vl_linspace, eta_pd + 3*std_eta, label='upper', ls='--')
		axs[idx][2].plot(vl_linspace, eta_pd - 3*std_eta, label='lower', ls='--')
		axs[idx][2].grid()
		axs[idx][2].set_ylabel("Eta")
		axs[idx][2].set_xlabel("Via-Line Space (nm)")
		axs[idx][2].set_title("GP Regression for Eta")
		axs[idx][2].legend()

		# fit curves
		plot_Dmin, plot_Dmax = None, None
		for vm_offset in vm_offset_list:
			vl_space = ll_space - vm_offset
			beta, eta = wb_gprs[idx].predict(vl_query=vl_space, return_std=False)

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
			axs[idx][0].plot(D_ls, fit_weibits, label=f'fit vl={vl_space:.2f} nm', color=cmap(norm(vl_space)))

		axs[idx][0].set_xscale('log')
		axs[idx][0].grid()
		axs[idx][0].set_ylabel("Weibits")
		axs[idx][0].set_xlabel("Breakdown Time")
		axs[idx][0].set_ylim(weibits_min, weibits_max)
		axs[idx][0].set_title(f"Exp and Fit Weibits")
		sm = cm.ScalarMappable(cmap=cmap, norm=norm)
		sm.set_array([])
		cbar = figs[idx].colorbar(sm, ax=axs[idx][0])
		cbar.set_label("vl space (nm)")
		figs[idx].tight_layout()
		figs[idx].savefig(save_path / f"m{m_list[idx]:.2f}_rN{radius_N:.2f}.png")
		plt.close(figs[idx])

		# save model
		joblib.dump(wb_gprs[idx], save_path / f"weibull_gpr_m{m_list[idx]:.2f}_rN{radius_N:.2f}.pkl")