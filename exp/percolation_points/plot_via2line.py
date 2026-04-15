import argparse
import numpy as np
import pathlib
import pyvista as pv
from src.percolation.via2line import Via2LineSim_retrieve_percolation_path

data_root = pathlib.Path('./exp/percolation_points/gen_data/via2line/')

from exp.percolation_points.plot_line2line import plot_sphere

def load_sim_data(args):
	vm_offset = args.vm_offset
	ll_space = args.ll_space
	via_dim_x = args.via_dim_x
	via_dim_y = args.via_dim_y
	via_dim_z = args.via_dim_z
	line_dim_y = args.line_dim_y
	line_dim_z = args.line_dim_z
	radius = args.radius

	max_defects = args.max_defects
	rebuild_thresh = args.rebuild_thresh
	workers = 1

	dir_name = f'vm{vm_offset:.2f}_ll{ll_space:.2f}_vx{via_dim_x:.2f}_vy{via_dim_y:.2f}_vz{via_dim_z:.2f}_ly{line_dim_y:.2f}_lz{line_dim_z:.2f}_r{radius:.2f}'
	data_dir = data_root / dir_name

	break_flag = np.load(data_dir / "break_flag.npy")
	defect_num = np.load(data_dir / "defect_num.npy")
	sim_points = np.load(data_dir / "points.npy")

	sucs_break_sims = break_flag.sum().item()
	if args.verbose: print(f'Simulations with successful breakdown: {sucs_break_sims} / {break_flag.shape[0]}')
	assert sucs_break_sims > 0
	all_defect_num = defect_num.sum().item()
	if all_defect_num == sim_points.shape[0]:
		if args.verbose: print(f"Number of defects is consistent: {all_defect_num} / {sim_points.shape[0]}")
	else:
		if args.verbose: print(f"Warning! Number of defects is not consistent: {all_defect_num} / {sim_points.shape[0]}")

	point_start = 0
	sucs_sim_idxs = []
	sucs_sim_points = []
	for idx in range(break_flag.shape[0]):
		if break_flag[idx]:
			sucs_sim_idxs.append(idx)
			sucs_sim_points.append(sim_points[point_start:point_start+defect_num[idx], :])
		point_start += defect_num[idx]

	assert point_start == all_defect_num

	return break_flag, defect_num, sim_points, sucs_sim_idxs, sucs_sim_points

def rendering(args, seed:int):
	rng = np.random.default_rng(seed=seed)

	vm_offset = args.vm_offset
	ll_space = args.ll_space
	via_dim_x = args.via_dim_x
	via_dim_y = args.via_dim_y
	via_dim_z = args.via_dim_z
	line_dim_y = args.line_dim_y
	line_dim_z = args.line_dim_z
	radius = args.radius

	max_defects = args.max_defects
	rebuild_thresh = args.rebuild_thresh
	workers = 1

	break_flag, defect_num, sim_points, sucs_sim_idxs, sucs_sim_points = load_sim_data(args=args)

	# random plot
	rand_sucs_idx = rng.integers(len(sucs_sim_idxs))
	rand_idx = sucs_sim_idxs[rand_sucs_idx]

	if args.verbose: print(f'Simulation Index: {rand_idx}')
	if args.verbose: print(f'Number of Defects: {defect_num[rand_idx]}')
	points = sucs_sim_points[rand_sucs_idx]

	perco_points = Via2LineSim_retrieve_percolation_path(
						defect_points=points,
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
						seed=seed)

	total_defect_num = points.shape[0]
	perco_defect_num = perco_points.shape[0]
	if args.verbose: print(f'Percolation / Total Defect Number: {perco_defect_num} / {total_defect_num}')
	if args.verbose: print(f'Percolation Defect Ratio: {100*perco_defect_num/total_defect_num:.2f} %')

	plotter = pv.Plotter()

	# ---- cuboid ----
	line_a = pv.Box(bounds=(0., via_dim_x, 0., line_dim_y, 0., line_dim_z))
	plotter.add_mesh(line_a, color='purple')
	line_b = pv.Box(bounds=(via_dim_x+ll_space, via_dim_x+ll_space+via_dim_x, 0., line_dim_y, 0., line_dim_z))
	plotter.add_mesh(line_b, color='purple')
	via = pv.Box(bounds=(vm_offset, vm_offset+via_dim_x, (line_dim_y - via_dim_y)/2, (line_dim_y - via_dim_y)/2 + via_dim_y, line_dim_z, line_dim_z + via_dim_z))
	plotter.add_mesh(via, color='purple')

	plot_sphere(plotter=plotter, points=perco_points, radius=radius)
	if args.show: plotter.show()

	if args.save:
		plotter.export_html(data_dir / f"rendering_{rand_idx}.vtk")
		if args.verbose: print(f'Saved to {data_dir / f"rendering_{rand_idx}.vtk"}')

	plotter.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog='plot_via2line.py')
	parser.add_argument('--vm-offset', type=float, default=4.0, help='Via misalignment, along the x dim')
	parser.add_argument('--ll-space', type=float, default=10.5, help='Line-line spacing, along the x dim')
	parser.add_argument('--via-dim-x', type=float, default=10.5, help='Via size (x dim)')
	parser.add_argument('--via-dim-y', type=float, default=10.5, help='Via size (y dim)')
	parser.add_argument('--via-dim-z', type=float, default=21.0, help='Via size (z dim)')
	parser.add_argument('--line-dim-y', type=float, default=21.0, help='Line size (y dim)')
	parser.add_argument('--line-dim-z', type=float, default=21.0, help='Line size (z dim)')
	parser.add_argument('-r', '--radius', type=float, default=0.45, help='Radius of defects')

	parser.add_argument('--max-defects', type=int, default=100000, help='The max allowed defects')
	parser.add_argument('--rebuild-thresh', type=int, default=50, help='Iteration interval to rebuild KD-Tree')

	parser.add_argument('--seed-list', nargs="+", type=str, help="List of random seeds")
	parser.add_argument('--save', action='store_true', help='Save the rendering')
	parser.add_argument('--show', action='store_true')
	parser.add_argument('--verbose', action='store_true')
	args = parser.parse_args()
	print(args)

	for seed in args.seed_list:
		seed = int(seed)
		if args.verbose: print(f'---------- Rendering with seed {seed} ----------')
		rendering(args=args, seed=seed)
		if args.verbose: print('')