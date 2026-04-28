import argparse
import numpy as np
import pathlib
from src.percolation.via2line import Via2LineSim_sumup_time_intervals, Via2LineSim_sumup_time_intervals_create_wrapper
from exp.percolation_points.plot_via2line import load_sim_data
import numpy as np
import functools
import multiprocessing
from tqdm import tqdm

from src.distribution.weibull import weibull_plot

import os
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

data_root = pathlib.Path('./exp/percolation_points/gen_data/via2line/')
save_root = pathlib.Path('./exp/percolation_points/gen_data/via2line_pp/')

def mp_post_process(pp_wrapper:functools.partial,
					sim_points:[np.ndarray],
					chunk_size:int=4,
					cpu_num:int=-1,
					show_progress:bool=False) -> np.ndarray:
	cpu_num = multiprocessing.cpu_count() if cpu_num <= 0 else cpu_num

	try:
		with multiprocessing.Pool(cpu_num) as pool:

			if show_progress:
				results = list(
					tqdm(
						pool.imap(pp_wrapper, sim_points, chunk_size),
						total=len(sim_points)
					)
				)
			else:
				results = list(pool.imap(pp_wrapper, sim_points, chunk_size))

		return np.array(results)

	except KeyboardInterrupt:
		pool.terminate()
		pool.join()
		return


if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog='post_process_via2line.py')
	parser.add_argument('--vm-offset', type=float, default=4.0, help='Via misalignment, along the x dim')
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

	parser.add_argument('--chunk-size', type=int, default=4, help='Chunk size of simulations')
	parser.add_argument('--cpu-num', type=int, default=-1, help='The number of CPUs')
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument('--verify', action='store_true', help='Verify breakdown')
	args = parser.parse_args()
	print(args)

	vm_offset = args.vm_offset
	ll_space = args.ll_space
	via_dim_x = args.via_dim_x
	via_dim_y = args.via_dim_y
	via_dim_z = args.via_dim_z
	line_dim_y = args.line_dim_y
	line_dim_z = args.line_dim_z
	radius = args.radius

	dir_name = f'vm{vm_offset:.2f}_ll{ll_space:.2f}_vx{via_dim_x:.2f}_vy{via_dim_y:.2f}_vz{via_dim_z:.2f}_ly{line_dim_y:.2f}_lz{line_dim_z:.2f}_r{radius:.2f}'

	m_list = args.m_list
	radius_N = args.radius_N

	max_defects = args.max_defects
	rebuild_thresh = args.rebuild_thresh
	workers = 1

	_, defect_num, _, sucs_sim_idxs, sucs_sim_points = load_sim_data(args=args)

	V = (via_dim_x*2+ll_space) * line_dim_y * (line_dim_z + via_dim_z) -\
		via_dim_x*line_dim_y*line_dim_z*2 - via_dim_x*via_dim_y*via_dim_z
	sucs_defect_num = [defect_num[idx] for idx in sucs_sim_idxs]
	sucs_defect_density = np.array(sucs_defect_num).astype(np.float32) / V
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

	sucs_breakdown_time = mp_post_process(pp_wrapper, sucs_sim_points,\
										  chunk_size=args.chunk_size,\
										  cpu_num=args.cpu_num,\
										  show_progress=True)
	m_num = len(m_list)
	sucs_breakdown_time = sucs_breakdown_time.reshape(-1, m_num)

	if not save_root.exists():
		save_root.mkdir()
	save_path = save_root / dir_name
	if not save_path.exists():
		save_path.mkdir()

	fig1, ax = plt.subplots(figsize=(8, 5))
	ax.hist(sucs_defect_density, bins=200, density=True)
	ax.grid()
	ax.set_ylabel('Probability Density')
	ax.set_xlabel('Defect Desnity')
	ax.set_title(f'Defect Density Distribution')
	fig1.savefig(save_path / "defect_density.png")
	plt.close(fig1)

	fig2, ax = plt.subplots(figsize=(8, 5))
	weibull_plot(ax, sucs_defect_density)
	ax.set_xlabel('Defect Density')
	ax.set_title(f'Defect Density Weibits')
	fig2.savefig(save_path / "defect_density_weibits.png")
	plt.close(fig2)

	for idx, m in enumerate(m_list):
		fig3, ax = plt.subplots(figsize=(8, 5))
		ax.hist(sucs_breakdown_time[:, idx], bins=200, density=True)
		ax.grid()
		ax.set_ylabel('Probability Density')
		ax.set_xlabel('Breakdown Time')
		ax.set_title(f'Breakdown Time Distribution; m={m:.2f}; rN={radius_N:.2f}')
		fig3.savefig(save_path / f"breakdown_time_m{m:.2f}_rN{radius_N:.2f}.png")
		plt.close(fig3)

		fig4, ax = plt.subplots(figsize=(8, 5))
		weibull_plot(ax, sucs_breakdown_time[:, idx])
		ax.set_xlabel('Breakdown Time')
		ax.set_title(f'Breakdown Time Weibits')
		fig4.savefig(save_path / f"defect_density_weibits_m{m:.2f}_rN{radius_N:.2f}.png")
		plt.close(fig4)