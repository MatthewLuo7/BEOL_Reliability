import argparse
import pathlib
import itertools
import numpy as np
from tqdm import tqdm
import joblib

from src.percolation.via2line import Via2LineSim_sumup_time_intervals_create_wrapper
from src.distribution.weibull import weibull_convertion, fit_weibull, weibull_plot
from src.models.physics.local_percolation_gpr.model import WeibullGPR

from exp.percolation_points.plot_via2line import load_sim_data
from exp.percolation_points.post_process_via2line import mp_post_process

import os
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

data_root = pathlib.Path('./exp/percolation_points/gen_data/via2line/')
save_root = pathlib.Path('./exp/percolation_points/gen_data/via2line_bt_fitting_gpr/')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog='via2line_bt_fitting_gpr.py')
	parser.add_argument('--vm-offset-list', nargs="+", type=float,
						default=[-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
						# default=[0.0, 2.0],
						help='List of vm offset')
	parser.add_argument('--ll-space-list', nargs="+", type=float,
						default=[5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
						# default=[5.5],
						help='Line-line spacing, along the x dim')
	parser.add_argument('--via-dim-y', type=float, default=10.5, help='Via size (y dim)')
	parser.add_argument('--via-dim-z', type=float, default=21.0, help='Via size (z dim)')
	parser.add_argument('--line-dim-x', type=float, default=10.5, help='Line size (x dim)')
	parser.add_argument('--line-dim-y', type=float, default=21.0, help='Line size (y dim)')
	parser.add_argument('--line-dim-z', type=float, default=21.0, help='Line size (z dim)')
	parser.add_argument('-r', '--radius', type=float, default=0.45, help='Radius of defects')

	parser.add_argument('--m-list', nargs="+", type=float, help='A list of parameter m (> 1.0)')
	parser.add_argument('-rN', '--radius-N', type=float, default=2.0, help='Radius for checking neighbouring defects (r_N > r)')

	parser.add_argument('--max-defects', type=int, default=100000, help='The max allowed defects')
	parser.add_argument('--rebuild-thresh', type=int, default=50, help='Iteration interval to rebuild KD-Tree')
	parser.add_argument('--conf-factor', type=float, default=2.0)

	parser.add_argument('--chunk-size', type=int, default=4, help='Chunk size of simulations')
	parser.add_argument('--cpu-num', type=int, default=-1, help='The number of CPUs')
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument('--verify', action='store_true', help='Verify breakdown')
	parser.add_argument('--reload-breakdown-time', action='store_true')
	args = parser.parse_args()
	print(args)

	vm_offset_list = args.vm_offset_list
	ll_space_list = args.ll_space_list
	via_dim_y = args.via_dim_y
	via_dim_z = args.via_dim_z
	line_dim_x = args.line_dim_x
	line_dim_y = args.line_dim_y
	line_dim_z = args.line_dim_z
	radius = args.radius
	workers = 1

	m_list = args.m_list
	radius_N = args.radius_N
	m_num = len(m_list)
	m_np = np.array(m_list)

	max_defects = args.max_defects
	rebuild_thresh = args.rebuild_thresh

	if not save_root.exists():
		save_root.mkdir()
	dir_name = f'vy{via_dim_y:.2f}_vz{via_dim_z:.2f}_lx{line_dim_x:.2f}_ly{line_dim_y:.2f}_lz{line_dim_z:.2f}_r{radius:.2f}'
	save_path = save_root / dir_name
	if not save_path.exists(): save_path.mkdir()
	bt_save_path = save_path / 'breakdown_time'
	if not bt_save_path.exists(): bt_save_path.mkdir()
	db_save_path = save_path / 'distribution'
	if not db_save_path.exists(): db_save_path.mkdir()


	# ----------------- collect data -----------------
	breakdown_time_list = []
	x_list = []		# (vl_space, ll_space)
	beta_list = []
	eta_list = []

	for (vm_offset, ll_space) in itertools.product(vm_offset_list, ll_space_list):
		args.vm_offset = vm_offset
		args.ll_space = ll_space
		vl_space = ll_space - vm_offset

		try:
			bt_filename = f"vl{vl_space:.2f}_ll{ll_space:.2f}_rN{radius_N:.2f}.npz"
			
			try:
				assert args.reload_breakdown_time
				data = np.load(bt_save_path / bt_filename)
				sucs_breakdown_time = data["sucs_breakdown_time"]
				ld_m_np = data["m_np"]
				assert np.power(ld_m_np - m_np, 2).sum() == 0.
				print(f'Loaded breakdown time from {bt_save_path / bt_filename}')

			except:
				_, _, _, _, sucs_sim_points = load_sim_data(args=args)
				pp_wrapper = Via2LineSim_sumup_time_intervals_create_wrapper(
								m_np=m_np,
								radius_N=radius_N,
								vm_offset=vm_offset,
								ll_space=ll_space,
								via_dim_y=via_dim_y,	
								via_dim_z=via_dim_z,
								line_dim_x=line_dim_x,
								line_dim_y=line_dim_y,
								line_dim_z=line_dim_z,
								radius=radius,
								max_defects=max_defects,
								rebuild_thresh=rebuild_thresh,
								workers=workers,
								verify=args.verify)
		
				sucs_breakdown_time = mp_post_process(pp_wrapper, sucs_sim_points,\
													  chunk_size=args.chunk_size,\
													  cpu_num=args.cpu_num,\
													  show_progress=False)
				sucs_breakdown_time = sucs_breakdown_time.reshape(-1, m_num)

				np.savez(
					bt_save_path / bt_filename,
					sucs_breakdown_time=sucs_breakdown_time,
					m_np=m_np
				)

	
			betas, etas = zip(*[fit_weibull(data=sucs_breakdown_time[:, idx]) for idx in range(m_num)])
			x_list.append([vl_space, ll_space])
			beta_list.append(betas)
			eta_list.append(etas)
			breakdown_time_list.append(sucs_breakdown_time)
		except:
			continue


	x_np = np.array(x_list).reshape(-1, 2)
	beta_np = np.array(beta_list).reshape(-1, m_num)
	eta_np = np.array(eta_list).reshape(-1, m_num)

	vl = np.linspace(0., 22., 100)
	ll = np.linspace(5., 12., 100)

	VL, LL = np.meshgrid(vl, ll)
	mask = VL <= (LL + 6)

	weibits_min, weibits_max = -6, 3
	for idx in range(m_num):
		wb_gpr = WeibullGPR()
		file_name = f"weibull_gpr_m{m_list[idx]:.2f}_rN{radius_N:.2f}"
		wb_gpr.fit(x_np, beta_np[:, idx], eta_np[:, idx], train_save_path=save_path / (file_name + '.npz'))

		x_pred = np.stack([VL.ravel(), LL.ravel()], axis=1)
		beta, eta = wb_gpr.predict(X=x_pred, verbose=True, conf_factor=args.conf_factor)

		beta_mean  = beta[..., 0].reshape(VL.shape)
		beta_lower = beta[..., 1].reshape(VL.shape)
		beta_upper = beta[..., 2].reshape(VL.shape)

		eta_mean  = eta[..., 0].reshape(VL.shape)
		eta_lower = eta[..., 1].reshape(VL.shape)
		eta_upper = eta[..., 2].reshape(VL.shape)

		beta_mean[~mask]  = np.nan
		beta_lower[~mask] = np.nan
		beta_upper[~mask] = np.nan

		eta_mean[~mask]  = np.nan
		eta_lower[~mask] = np.nan
		eta_upper[~mask] = np.nan

		fig = plt.figure(figsize=(9, 4))
		ax1 = fig.add_subplot(1, 2, 1, projection='3d')
		ax2 = fig.add_subplot(1, 2, 2, projection='3d')

		ax1.plot_surface(VL, LL, beta_mean , alpha=0.8, color='green', label='mean')
		ax1.plot_surface(VL, LL, beta_lower, alpha=0.3, color='red'  , label='lower')
		ax1.plot_surface(VL, LL, beta_upper, alpha=0.3, color='blue' , label='upper')
		ax1.set_xlabel("vl_space")
		ax1.set_ylabel("ll_space")
		ax1.set_title(f"Weibull beta surface with ±{args.conf_factor}σ uncertainty")
		ax1.legend()

		ax2.plot_surface(VL, LL, eta_mean,  alpha=0.8, color='green', label='mean')
		ax2.plot_surface(VL, LL, eta_lower, alpha=0.3, color='red'  , label='lower')
		ax2.plot_surface(VL, LL, eta_upper, alpha=0.3, color='blue' , label='upper')
		ax2.set_xlabel("vl_space")
		ax2.set_ylabel("ll_space")
		ax2.set_title(f"Weibull eta surface with ±{args.conf_factor}σ uncertainty")
		ax2.legend()
		
		fig.savefig(save_path / f"m{m_list[idx]:.2f}_rN{radius_N:.2f}.png")
		plt.close(fig)

		# save distributions
		dist_save_path = db_save_path / f"{m_list[idx]:.2f}_rN{radius_N:.2f}"
		if not dist_save_path.exists(): dist_save_path.mkdir()
		for j, (vl_space, ll_space) in enumerate(x_list):
			fig, ax = plt.subplots(1, 2, figsize=(9, 4))
			ax[0].hist(breakdown_time_list[j][:, idx], bins=200, density=True)
			ax[0].grid()
			ax[0].set_ylabel('Probability Density')
			ax[0].set_xlabel('Breakdown Time')
			ax[0].set_title(f'Breakdown Time Distribution')

			weibull_plot(ax[1], breakdown_time_list[j][:, idx])
			ax[1].set_xlabel('Breakdown Time')
			ax[1].set_title(f'Breakdown Time Weibits')

			fig.savefig(dist_save_path / f"vl{vl_space:.2f}_ll{ll_space:.2f}.png")
			plt.tight_layout()
			plt.close(fig)