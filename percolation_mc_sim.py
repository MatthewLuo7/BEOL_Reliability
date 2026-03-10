import argparse
from src.percolation.planar_capacitor import PlanarCapSim_create_wrapper
from src.percolation.mc_simulation import mc_simulate
import json
import pathlib
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog='percolation_mc_sim.py')
	parser.add_argument('-x', '--dimx', type=float, default=30.0, help='Dimension X')
	parser.add_argument('-y', '--dimy', type=float, default=30.0, help='Dimension Y')
	parser.add_argument('-z', '--dimz', type=float, default=10.0, help='Dimension Z')
	parser.add_argument('-r', '--radius', type=float, default=0.45, help='Radius of defects')
	parser.add_argument('-n', '--sample-num', type=int, default=10000, help='The number of simulation smaples')

	parser.add_argument('--max-defects', type=int, default=100000, help='The max allowed defects')
	parser.add_argument('--rebuild-thresh', type=int, default=50, help='Iteration interval to rebuild KD-Tree')
	parser.add_argument('--vol-est-samples', type=int, default=500000, help='MC samples for estimating volume')
	parser.add_argument('--randseq-entropy', type=int, default=1234, help='The entropy to generate random seed sequence')
	parser.add_argument('--chunk-size', type=int, default=16, help='Chunk size of simulations')
	parser.add_argument('--cpu-num', type=int, default=-1, help='The number of CPUs')

	parser.add_argument('--show', action='store_true', help='Show the density diagram')
	parser.add_argument('--save-path', type=str, default=None, help='Path to save the results')
	args = parser.parse_args()
	print(args)

	# # ------------- single simulation -------------
	# sim = PlanarCapSim()
	# sim.simulate()
	# is_broken, defect_num, defect_volume_norm = sim.get_sim_results()
	# print(f'Breakdown: {is_broken}')
	# print(f'Defects at Breakdown: {defect_num}')
	# print(f'Normalized Defect Volume: {defect_volume_norm*100:.2f} %')

	dimx = args.dimx
	dimy = args.dimy
	dimz = args.dimz
	radius = args.radius
	sample_num = args.sample_num

	max_defects = args.max_defects
	rebuild_thresh = args.rebuild_thresh
	vol_est_samples = args.vol_est_samples
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
					workers=workers,
					vol_est_samples=vol_est_samples)

	results = mc_simulate(sim_wrapper=sim_wrapper,
						  sample_num=sample_num,
						  randseq_entropy=randseq_entropy,
						  chunk_size=chunk_size,
						  cpu_num=cpu_num)
	break_np, defect_np, volume_np = results

	mask = (break_np == True)
	fig, ax = plt.subplots(figsize=(8, 5))
	ax.hist(volume_np[mask], bins=200, density=True)
	ax.grid()
	ax.set_ylabel('Probability Density')
	ax.set_xlabel('Normalized Defect Density')
	ax.set_title(f'x={dimx:.2f}; y={dimy:.2f}; z={dimz:.2f}; r={radius:.2f}')

	if args.save_path is not None:
		dir_name = f'x{dimx:.2f}_y{dimy:.2f}_z{dimz:.2f}_r{radius:.2f}'
		save_path = pathlib.Path(args.save_path) / dir_name
		if not save_path.is_dir(): save_path.mkdir()

		# save picture
		plt.savefig(save_path / "distribution.png")

		# save data
		np.save(save_path / "break_flag.npy", break_np)
		np.save(save_path / "defect_num.npy", defect_np)
		np.save(save_path / "norm_volume.npy", volume_np)

		# save info dict
		info_dict = {'args': vars(args),\
					 'break_rate': mask.sum() / sample_num}
		with open(save_path / "info.json", 'w', encoding='utf-8') as f:
			json.dump(info_dict, f, indent=4)

	if args.show: plt.show()	