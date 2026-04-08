import argparse
from src.percolation.planar_capacitor import PlanarCapSim_simulation, PlanarCapSim_create_wrapper
from src.percolation.mc_simulation import mc_simulate
import json
import pathlib
import numpy as np

if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog='line2line_sim.py')
	parser.add_argument('-x', '--dimx', type=float, default=30.0, help='Dimension X')
	parser.add_argument('-y', '--dimy', type=float, default=30.0, help='Dimension Y')
	parser.add_argument('-z', '--dimz', type=float, default=10.0, help='Dimension Z')
	parser.add_argument('-r', '--radius', type=float, default=0.45, help='Radius of defects')
	parser.add_argument('-n', '--sample-num', type=int, default=1000, help='The number of simulation smaples')

	parser.add_argument('--max-defects', type=int, default=100000, help='The max allowed defects')
	parser.add_argument('--rebuild-thresh', type=int, default=50, help='Iteration interval to rebuild KD-Tree')
	parser.add_argument('--randseq-entropy', type=int, default=1234, help='The entropy to generate random seed sequence')
	parser.add_argument('--chunk-size', type=int, default=16, help='Chunk size of simulations')
	parser.add_argument('--cpu-num', type=int, default=-1, help='The number of CPUs')

	parser.add_argument('--save-path', type=str, default=None, help='Path to save the results')
	parser.add_argument('--single-sim', action='store_true', help='Just run a single simulation')
	args = parser.parse_args()
	print(args)

	# # ------------- single simulation -------------
	if args.single_sim:
		sim_res = PlanarCapSim_simulation(
					seed=args.randseq_entropy,
					dimx=args.dimx,
					dimy=args.dimy,
					dimz=args.dimz,
					radius=args.radius,
					max_defects=args.max_defects,
					rebuild_thresh=args.rebuild_thresh,
					workers=args.cpu_num)
		is_broken, defect_num, p_list = sim_res
		print(f'Breakdown: {is_broken}')
		print(f'Defects at Breakdown: {defect_num}')
		print(f'Point list')
		print(p_list[:16])
		exit(0)

	dimx = args.dimx
	dimy = args.dimy
	dimz = args.dimz
	radius = args.radius
	sample_num = args.sample_num

	max_defects = args.max_defects
	rebuild_thresh = args.rebuild_thresh
	randseq_entropy = args.randseq_entropy
	chunk_size = args.chunk_size
	cpu_num = args.cpu_num
	workers = 1 							# use one worker for each batch
	
	sim_wrapper = PlanarCapSim_create_wrapper(
					dimx=dimx,
					dimy=dimy,
					dimz=dimz,
					radius=radius,
					max_defects=max_defects,
					rebuild_thresh=rebuild_thresh,
					workers=workers)
	sim_volume = dimx*dimy*dimz

	results = mc_simulate(sim_wrapper=sim_wrapper,
						  sample_num=sample_num,
						  randseq_entropy=randseq_entropy,
						  chunk_size=chunk_size,
						  cpu_num=cpu_num)
	break_np, defect_np, points_np = results

	if args.save_path is not None:
		dir_name = f'x{dimx:.2f}_y{dimy:.2f}_z{dimz:.2f}_r{radius:.2f}'
		save_path = pathlib.Path(args.save_path) / dir_name
		if not save_path.is_dir(): save_path.mkdir()

		# save data
		np.save(save_path / "break_flag.npy", break_np)
		np.save(save_path / "defect_num.npy", defect_np)
		np.save(save_path / "points.npy", points_np)