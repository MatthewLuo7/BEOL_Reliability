import argparse
import numpy as np
import pathlib
import pyvista as pv
from tqdm import tqdm

from src.percolation.planar_capacitor import PlanarCapSim_retrieve_percolation_path

data_root = pathlib.Path('./exp/percolation_points/gen_data/line2line/')

def plot_sphere(plotter, points:np.ndarray, radius:float, color:str='grey', opacity:float=0.5):
	for p in tqdm(points, total=points.shape[0], desc="Rendering spheres"):
		sphere = pv.Sphere(radius=radius, center=p)
		plotter.add_mesh(sphere, color=color, opacity=opacity)

def load_sim_data(args):
	dimx = args.dimx
	dimy = args.dimy
	dimz = args.dimz
	radius = args.radius

	max_defects = args.max_defects
	rebuild_thresh = args.rebuild_thresh
	workers = 1

	dir_name = f'x{dimx:.2f}_y{dimy:.2f}_z{dimz:.2f}_r{radius:.2f}'
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

	dimx = args.dimx
	dimy = args.dimy
	dimz = args.dimz
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

	perco_points = PlanarCapSim_retrieve_percolation_path(
						defect_points=points,
						dimx=dimx,
						dimy=dimy,
						dimz=dimz,
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
	mthick = 2.
	box_bot = pv.Box(bounds=(0., dimx, 0., dimy, 0., mthick))
	plotter.add_mesh(box_bot, color='purple')
	box_top = pv.Box(bounds=(0., dimx, 0., dimy, mthick+dimz, mthick+dimz+mthick))
	plotter.add_mesh(box_top, color='purple')

	pos_offset = np.array([0., 0., mthick]).reshape(1, 3)
	plot_sphere(plotter=plotter, points=perco_points+pos_offset, radius=radius)
	if args.show: plotter.show()

	if args.save:
		plotter.export_html(data_dir / f"rendering_{rand_idx}.vtk")
		if args.verbose: print(f'Saved to {data_dir / f"rendering_{rand_idx}.vtk"}')

	plotter.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog='plot_line2line.py')
	parser.add_argument('-x', '--dimx', type=float, default=30.0, help='Dimension X')
	parser.add_argument('-y', '--dimy', type=float, default=30.0, help='Dimension Y')
	parser.add_argument('-z', '--dimz', type=float, default=10.0, help='Dimension Z')
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