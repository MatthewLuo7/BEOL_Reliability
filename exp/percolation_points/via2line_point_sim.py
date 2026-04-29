import argparse
from src.percolation.via2line import Via2LineSim_simulation, Via2LineSim_create_wrapper
from src.percolation.mc_simulation import mc_simulate
import json
import pathlib
import numpy as np

if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog='via2line_sim.py')
	parser.add_argument('--vm-offset', type=float, default=4.0, help='Via misalignment, along the x dim')
	parser.add_argument('--ll-space', type=float, default=10.5, help='Line-line spacing, along the x dim')
	parser.add_argument('--via-dim-y', type=float, default=10.5, help='Via size (y dim)')
	parser.add_argument('--via-dim-z', type=float, default=21.0, help='Via size (z dim)')
	parser.add_argument('--line-dim-x', type=float, default=10.5, help='Line size (x dim)')
	parser.add_argument('--line-dim-y', type=float, default=21.0, help='Line size (y dim)')
	parser.add_argument('--line-dim-z', type=float, default=21.0, help='Line size (z dim)')
	parser.add_argument('-r', '--radius', type=float, default=0.45, help='Radius of defects')
	parser.add_argument('-n', '--sample-num', type=int, default=1000, help='The number of simulation smaples')

	parser.add_argument('--max-defects', type=int, default=100000, help='The max allowed defects')
	parser.add_argument('--rebuild-thresh', type=int, default=50, help='Iteration interval to rebuild KD-Tree')
	parser.add_argument('--randseq-entropy', type=int, default=1234, help='The entropy to generate random seed sequence')
	parser.add_argument('--chunk-size', type=int, default=4, help='Chunk size of simulations')
	parser.add_argument('--cpu-num', type=int, default=-1, help='The number of CPUs')

	parser.add_argument('--save-path', type=str, default=None, help='Path to save the results')
	parser.add_argument('--single-sim', action='store_true', help='Just run a single simulation')
	args = parser.parse_args()
	print(args)

	# # ------------- single simulation -------------
	if args.single_sim:
		sim_res = Via2LineSim_simulation(
					seed=args.randseq_entropy,
					vm_offset=args.vm_offset,
					ll_space=args.ll_space,	
					via_dim_y=args.via_dim_y,	
					via_dim_z=args.via_dim_z,
					line_dim_x=args.line_dim_x,
					line_dim_y=args.line_dim_y,
					line_dim_z=args.line_dim_z,
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

	vm_offset = args.vm_offset
	ll_space = args.ll_space
	via_dim_y = args.via_dim_y
	via_dim_z = args.via_dim_z
	line_dim_x = args.line_dim_x
	line_dim_y = args.line_dim_y
	line_dim_z = args.line_dim_z
	radius = args.radius
	sample_num = args.sample_num

	max_defects = args.max_defects
	rebuild_thresh = args.rebuild_thresh
	randseq_entropy = args.randseq_entropy
	chunk_size = args.chunk_size
	cpu_num = args.cpu_num
	workers = 1 							# use one worker for each batch

	sim_wrapper = Via2LineSim_create_wrapper(
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
					workers=workers)
	sim_volume = (line_dim_x*2+ll_space)*line_dim_y*line_dim_z -\
				 line_dim_x*line_dim_y*line_dim_z*2 -\
				 (max(line_dim_x+vm_offset, 0.) - max(vm_offset, 0.))*via_dim_y*via_dim_z

	results = mc_simulate(sim_wrapper=sim_wrapper,
						  sample_num=sample_num,
						  randseq_entropy=randseq_entropy,
						  chunk_size=chunk_size,
						  cpu_num=cpu_num)
	break_np, defect_np, points_np = results

	if args.save_path is not None:
		dir_name = f'vm{vm_offset:.2f}_ll{ll_space:.2f}_vy{via_dim_y:.2f}_vz{via_dim_z:.2f}_lx{line_dim_x:.2f}_ly{line_dim_y:.2f}_lz{line_dim_z:.2f}_r{radius:.2f}'
		save_path = pathlib.Path(args.save_path) / dir_name
		if not save_path.is_dir(): save_path.mkdir()

		# save data
		np.save(save_path / "break_flag.npy", break_np)
		np.save(save_path / "defect_num.npy", defect_np)
		np.save(save_path / "points.npy", points_np)